#!/usr/bin/env python3
"""
Benchmark runner for CuTeDSL Blackwell (SM100/GB200) kernels.

Reads benchmark.csv, runs every (type, M, N, K, group) shape through all
relevant experimental/blackwell kernels, and writes GFLOPS results back.

Kernel coverage:
  gemm / groupgemm  fp8   → dense_block_scaled_gemm  (Float8E4M3FN + E8M0 scale)
  gemm / groupgemm  bf16  → dense_gemm                (BFloat16, 1-CTA)
  gemm / groupgemm  bf16  → dense_gemm_cute_pipeline  (BFloat16, 1-CTA or 2-CTA)
  gemm / groupgemm  bf16  → dense_gemm_ptr_array       (BFloat16, ptr-array batched)
  gemm / groupgemm  bf16  → dense_gemm_2sm             (BFloat16, 2-SM cluster)

Tile tuning:
  Before benchmarking each (kernel, M, N, K) the script sweeps candidate
  mma_tiler_mn values and picks the fastest one (--tune / --no_tune).

Usage:
  python bench_csv.py                         # full run, tuning ON
  python bench_csv.py --no_tune               # skip tuning, use default tile
  python bench_csv.py --warmup 3 --iters 10  # custom iteration counts
  python bench_csv.py --dry_run              # print commands without executing
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).parent.resolve()
BLACKWELL_DIR = REPO_ROOT / "examples/python/CuTeDSL/experimental/blackwell"
DEFAULT_CSV   = REPO_ROOT / "benchmark.csv"

# ─── ANSI ─────────────────────────────────────────────────────────────────────
RESET = "\033[0m"; GREEN = "\033[92m"; RED = "\033[91m"
YELLOW = "\033[93m"; CYAN = "\033[96m"; BOLD = "\033[1m"
def c(t, col): return f"{col}{t}{RESET}"

# ─── GFLOPS ───────────────────────────────────────────────────────────────────
def to_gflops(m, n, k, l, us):
    """2*M*N*K*L FLOPs, latency in µs → GFLOPS."""
    return 2.0 * m * n * k * l / (us * 1e-6) / 1e9

# ─── Subprocess helper ─────────────────────────────────────────────────────────
def _run(cmd, timeout=600):
    env = os.environ.copy()
    # Propagate current sys.path so subprocesses can find 'cutlass' installed
    # in the same venv/conda env as this script (fixes ModuleNotFoundError).
    extra = os.pathsep.join(p for p in sys.path if p)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (extra + os.pathsep + existing) if existing else extra
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout, cwd=str(REPO_ROOT), env=env,
    )
    return result.returncode, result.stdout, result.stderr

def _parse_us(stdout):
    """'Execution time: X microseconds per iteration' → float µs."""
    m = re.search(r"Execution time:\s*([0-9.e+\-]+)\s*microseconds", stdout)
    return float(m.group(1)) if m else None

def _parse_sec(stdout):
    """'Execution time: X seconds' → float µs."""
    m = re.search(r"Execution time:\s*([0-9.e+\-]+)\s*seconds", stdout)
    return float(m.group(1)) * 1e6 if m else None

def _parse_fp8_us(stdout):
    """'exec_time_us=X' from inline fp8 timer → float µs."""
    m = re.search(r"exec_time_us=([0-9.e+\-]+)", stdout)
    return float(m.group(1)) if m else None

# ─── Tuning helpers ───────────────────────────────────────────────────────────
# Valid mma_tiler_mn candidates for 1-CTA kernels (M_tile ∈ {64,128,256}, N_tile ∈ {128,256})
# Constraint: M_tile must be ≤ M, N_tile must be ≤ N, and both must divide evenly.
_TILE_CANDIDATES_1CTA = [
    (128, 128), (128, 256),
    (256, 128), (256, 256),
]
# 2-CTA kernel (dense_gemm_2sm): tile is fixed by mma_inst — no tiler sweep needed.
# dense_gemm_cute_pipeline with 2-CTA: tile is (256,256) fixed.


def _valid_tiles(m, n, candidates):
    """Return tile candidates that evenly divide M and N."""
    return [(tm, tn) for (tm, tn) in candidates if m % tm == 0 and n % tn == 0]


# ─── Kernel runners ────────────────────────────────────────────────────────────

def _subprocess_run(script, extra_args, parse_fn, timeout=600):
    """Call script, parse timing, return (us, error_str)."""
    cmd = [sys.executable, str(BLACKWELL_DIR / script)] + extra_args
    rc, out, err = _run(cmd, timeout)
    us = parse_fn(out)
    if us is None or rc != 0:
        last = ((err or out) or "").strip().splitlines()
        return None, (last[-1] if last else f"rc={rc}")
    return us, None


# ── dense_gemm.py (BF16, 1-CTA) ────────────────────────────────────────────

def _dense_gemm_args(m, n, k, l, tile, warmup, iters, skip_ref):
    tm, tn = tile
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--mma_tiler_mn", f"{tm},{tn}",
        "--cluster_shape_mn", "1,1",
        "--ab_dtype", "BFloat16", "--d_dtype", "BFloat16", "--acc_dtype", "Float32",
        "--warmup_iterations", str(warmup), "--iterations", str(iters),
        "--use_cold_l2",
    ]
    if skip_ref: args.append("--skip_ref_check")
    return args

def bench_dense_gemm(m, n, k, l, warmup, iters, skip_ref, tune, dry_run):
    if dry_run:
        tile = _valid_tiles(m, n, _TILE_CANDIDATES_1CTA)[0] if _valid_tiles(m, n, _TILE_CANDIDATES_1CTA) else (128, 128)
        return None, "DRY: " + " ".join(_dense_gemm_args(m, n, k, l, tile, warmup, iters, skip_ref))

    tiles = _valid_tiles(m, n, _TILE_CANDIDATES_1CTA) if tune else [(128, 128)]
    if not tiles:
        return None, f"No valid tile for M={m}, N={n}"
    best_us, best_tile = None, tiles[0]
    for tile in tiles:
        us, err = _subprocess_run("dense_gemm.py",
                                  _dense_gemm_args(m, n, k, l, tile, 1, 3, True),
                                  _parse_us)
        if us is not None and (best_us is None or us < best_us):
            best_us, best_tile = us, tile

    if best_us is None:
        return None, "all tiles failed in tune sweep"
    us, err = _subprocess_run("dense_gemm.py",
                              _dense_gemm_args(m, n, k, l, best_tile, warmup, iters, skip_ref),
                              _parse_us)
    if us is None: return None, err
    return to_gflops(m, n, k, l, us), None


# ── dense_block_scaled_gemm.py (FP8) ────────────────────────────────────────

def _fp8_timing_script(m, n, k, l, warmup, iters):
    """Generate a self-contained FP8 timing script (CUDA event based)."""
    # Embed parent sys.path so the temp script finds 'cutlass' in the same env
    parent_path = repr(list(sys.path))
    return textwrap.dedent(f"""\
        import sys
        for _p in {parent_path}:
            if _p and _p not in sys.path:
                sys.path.insert(0, _p)
        sys.path.insert(0, r'{BLACKWELL_DIR}')
        import cutlass, torch
        import cutlass.cute as cute
        from dense_block_scaled_gemm import BlockScaledGemmTestbed, BlockScaledDenseGemmKernel

        mnkl = ({m}, {n}, {k}, {l})
        ab_dtype, sf_dtype, sf_vec_size = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32
        d_dtype, acc_dtype = cutlass.Float16, cutlass.Float32
        mma_inst_mn = (128, 128)

        torch.manual_seed(42)
        tb = BlockScaledGemmTestbed(mnkl, (ab_dtype, ab_dtype, acc_dtype),
                                    d_dtype, sf_dtype, sf_vec_size, "k", "k", "n")
        kernel = BlockScaledDenseGemmKernel(mma_inst_mn=mma_inst_mn,
                                            mma_dtype=(ab_dtype, acc_dtype),
                                            sf_dtype=sf_dtype, sf_vec_size=sf_vec_size)
        compiled = cute.experimental.compile(
            kernel, tb.a_tensor, tb.sfa_tensor, tb.b_tensor, tb.sfb_tensor, tb.d_tensor)

        for _ in range({warmup}):
            compiled(tb.a_tensor, tb.sfa_tensor, tb.b_tensor, tb.sfb_tensor, tb.d_tensor)
        torch.cuda.synchronize()
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        for _ in range({iters}):
            compiled(tb.a_tensor, tb.sfa_tensor, tb.b_tensor, tb.sfb_tensor, tb.d_tensor)
        ev_e.record()
        torch.cuda.synchronize()
        print(f"exec_time_us={{ev_s.elapsed_time(ev_e) / {iters} * 1000:.4f}}")
    """)

def bench_fp8(m, n, k, l, warmup, iters, dry_run):
    if m % 128 != 0:
        return None, f"M={m} not divisible by 128 (mma_inst constraint)"
    if n % 128 != 0:
        return None, f"N={n} not divisible by 128 (mma_inst constraint)"
    if k % 64 != 0:
        return None, f"K={k} not divisible by 64 (sf_vec_size constraint)"

    script_text = _fp8_timing_script(m, n, k, l, warmup, iters)
    if dry_run:
        return None, "DRY: inline fp8 timing script"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, dir=str(REPO_ROOT)) as f:
        f.write(script_text); tmp = f.name
    try:
        rc, out, err = _run([sys.executable, tmp])
    finally:
        os.unlink(tmp)

    us = _parse_fp8_us(out)
    if us is None or rc != 0:
        last = ((err or out) or "").strip().splitlines()
        return None, (last[-1] if last else f"rc={rc}")
    return to_gflops(m, n, k, l, us), None


# ── dense_gemm_cute_pipeline.py (BF16, 1-CTA or 2-CTA) ─────────────────────

def _pipeline_args(m, n, k, l, tile, cluster, use_2cta, warmup, iters, skip_ref):
    tm, tn = tile
    cm, cn = cluster
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--mma_tiler_mn", f"{tm},{tn}",
        "--cluster_shape_mn", f"{cm},{cn}",
        "--ab_dtype", "BFloat16", "--c_dtype", "BFloat16", "--acc_dtype", "Float32",
        "--warmup_iterations", str(warmup), "--iterations", str(iters),
        "--benchmark", "default", "--use_cold_l2",
    ]
    if use_2cta: args.append("--use_2cta_instrs")
    if skip_ref: args.append("--skip_ref_check")
    return args

def bench_pipeline(m, n, k, l, use_2cta, warmup, iters, skip_ref, tune, dry_run):
    # 2-CTA: tile must be (256,256), cluster (2,1)
    if use_2cta:
        if m % 256 != 0: return None, f"M={m} not divisible by 256 (2-CTA tile)"
        if n % 256 != 0: return None, f"N={n} not divisible by 256 (2-CTA tile)"
        tile, cluster = (256, 256), (2, 1)
        if dry_run:
            return None, "DRY: " + " ".join(
                _pipeline_args(m, n, k, l, tile, cluster, True, warmup, iters, skip_ref))
        us, err = _subprocess_run("dense_gemm_cute_pipeline.py",
                                  _pipeline_args(m, n, k, l, tile, cluster,
                                                 True, warmup, iters, skip_ref),
                                  _parse_sec)
        if us is None: return None, err
        return to_gflops(m, n, k, l, us), None

    # 1-CTA: sweep tiles
    tiles = _valid_tiles(m, n, _TILE_CANDIDATES_1CTA) if tune else [(128, 128)]
    if not tiles: return None, f"No valid 1-CTA tile for M={m}, N={n}"
    if dry_run:
        return None, "DRY: " + " ".join(
            _pipeline_args(m, n, k, l, tiles[0], (1, 1), False, warmup, iters, skip_ref))

    best_us, best_tile = None, tiles[0]
    for tile in tiles:
        us, _ = _subprocess_run("dense_gemm_cute_pipeline.py",
                                _pipeline_args(m, n, k, l, tile, (1, 1), False, 1, 3, True),
                                _parse_sec)
        if us is not None and (best_us is None or us < best_us):
            best_us, best_tile = us, tile
    if best_us is None: return None, "all tiles failed in tune sweep"

    us, err = _subprocess_run("dense_gemm_cute_pipeline.py",
                              _pipeline_args(m, n, k, l, best_tile, (1, 1),
                                             False, warmup, iters, skip_ref),
                              _parse_sec)
    if us is None: return None, err
    return to_gflops(m, n, k, l, us), None


# ── dense_gemm_ptr_array.py (BF16, ptr-array batched) ───────────────────────

def _ptr_array_args(m, n, k, l, tile, warmup, iters, skip_ref):
    tm, tn = tile
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--mma_tiler_mn", f"{tm},{tn}",
        "--cluster_shape_mn", "1,1",
        "--ab_dtype", "BFloat16", "--d_dtype", "BFloat16", "--acc_dtype", "Float32",
        "--warmup_iterations", str(warmup), "--iterations", str(iters),
        "--use_cold_l2",
    ]
    if skip_ref: args.append("--skip_ref_check")
    return args

def bench_ptr_array(m, n, k, l, warmup, iters, skip_ref, tune, dry_run):
    tiles = _valid_tiles(m, n, _TILE_CANDIDATES_1CTA) if tune else [(128, 128)]
    if not tiles: return None, f"No valid tile for M={m}, N={n}"
    if dry_run:
        return None, "DRY: " + " ".join(
            _ptr_array_args(m, n, k, l, tiles[0], warmup, iters, skip_ref))

    best_us, best_tile = None, tiles[0]
    for tile in tiles:
        us, _ = _subprocess_run("dense_gemm_ptr_array.py",
                                _ptr_array_args(m, n, k, l, tile, 1, 3, True),
                                _parse_us)
        if us is not None and (best_us is None or us < best_us):
            best_us, best_tile = us, tile
    if best_us is None: return None, "all tiles failed in tune sweep"

    us, err = _subprocess_run("dense_gemm_ptr_array.py",
                              _ptr_array_args(m, n, k, l, best_tile, warmup, iters, skip_ref),
                              _parse_us)
    if us is None: return None, err
    return to_gflops(m, n, k, l, us), None


# ── dense_gemm_2sm.py (BF16, 2-SM cluster) ──────────────────────────────────

def _2sm_args(m, n, k, l, use_2cta, warmup, iters, skip_ref):
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--ab_dtype", "BFloat16", "--d_dtype", "BFloat16", "--acc_dtype", "Float32",
        "--warmup_iterations", str(warmup), "--iterations", str(iters),
    ]
    if use_2cta: args.append("--use_2cta_instrs")
    else:        args.append("--no_2cta_instrs")
    if skip_ref: args.append("--skip_ref_check")
    return args

def bench_2sm(m, n, k, l, warmup, iters, skip_ref, dry_run):
    # Prefer 2-CTA (higher throughput); fall back to 1-CTA if M not ÷256
    use_2cta = (m % 256 == 0 and n % 256 == 0 and k % 64 == 0)
    use_1cta = (m % 128 == 0 and n % 256 == 0 and k % 64 == 0)
    if not use_2cta and not use_1cta:
        return None, f"M={m} or N={n} or K={k} violates 2SM tile alignment"

    chosen_2cta = use_2cta  # prefer 2-CTA
    if dry_run:
        return None, "DRY: " + " ".join(
            _2sm_args(m, n, k, l, chosen_2cta, warmup, iters, skip_ref))

    us, err = _subprocess_run("dense_gemm_2sm.py",
                              _2sm_args(m, n, k, l, chosen_2cta, warmup, iters, skip_ref),
                              _parse_us)
    if us is None: return None, err
    return to_gflops(m, n, k, l, us), None


# ─── CSV I/O ──────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))

def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)

def ensure_col(rows, name):
    header = rows[0]
    for i, h in enumerate(header):
        if h.strip().lower() == name.lower():
            return i
    header.append(name)
    for row in rows[1:]:
        row.append("")
    return len(header) - 1


# ─── Output columns ───────────────────────────────────────────────────────────
#   One column per (kernel, variant) combination we benchmark.
OUTPUT_COLS = [
    # (csv_column_name,          bench_fn_key)
    ("gflops_fp8_block_scaled",    "fp8"),
    ("gflops_bf16_dense_gemm",     "dense_gemm"),
    ("gflops_bf16_pipeline_1cta",  "pipeline_1cta"),
    ("gflops_bf16_pipeline_2cta",  "pipeline_2cta"),
    ("gflops_bf16_ptr_array",      "ptr_array"),
    ("gflops_bf16_2sm",            "2sm"),
]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv",    default=str(DEFAULT_CSV), help="CSV path")
    ap.add_argument("--warmup", type=int, default=3,  help="Warmup iterations (default 3)")
    ap.add_argument("--iters",  type=int, default=10, help="Benchmark iterations (default 10)")
    ap.add_argument("--skip_ref_check", action="store_true", default=True)
    ap.add_argument("--tune",    dest="tune", action="store_true",  default=True,
                    help="Sweep tile sizes and pick fastest (default: on)")
    ap.add_argument("--no_tune", dest="tune", action="store_false",
                    help="Skip tile tuning, use default tile (128,128)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--kernel", nargs="+",
                    choices=["fp8","dense_gemm","pipeline_1cta","pipeline_2cta",
                             "ptr_array","2sm"],
                    default=None, help="Run only these kernel(s)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found: {csv_path}")

    rows = load_csv(csv_path)
    if len(rows) < 2:
        sys.exit("ERROR: CSV has no data rows.")

    header = rows[0]
    col = {h.strip().lower(): i for i, h in enumerate(header)}
    for req in ("type", "m", "n", "k"):
        if req not in col:
            sys.exit(f"ERROR: CSV missing column '{req}'")

    # Determine which kernels to run
    wanted = set(args.kernel) if args.kernel else {k for _, k in OUTPUT_COLS}

    # Ensure output columns exist
    out_idx = {}
    for col_name, key in OUTPUT_COLS:
        if key in wanted:
            out_idx[key] = ensure_col(rows, col_name)

    # Print header
    print(); print(c("=" * 110, BOLD))
    print(c(f"  CuTeDSL Blackwell Benchmark  ←  experimental/blackwell"
            f"  (tune={'ON' if args.tune else 'OFF'})", BOLD + CYAN))
    print(c("=" * 110, BOLD))
    hdr = f"  {'type':<12} {'M':>6} {'N':>6} {'K':>6} {'grp':>4}"
    col_keys = [k for _, k in OUTPUT_COLS if k in wanted]
    for k in col_keys:
        hdr += f"  {k:>18}"
    print(c(hdr, BOLD)); print(c("-" * 110, BOLD))

    def fmt(val, err):
        if val is not None: return c(f"{val:>10.1f}", GREEN) + "        "
        short = (err or "")[:18]
        return c(f"{'N/A':>10}", YELLOW) + f" [{short}]"

    total = fail = 0
    for row in rows[1:]:
        if not row or not row[0].strip(): continue
        rtype = row[col["type"]].strip()
        try:
            M = int(row[col["m"]]); N = int(row[col["n"]]); K = int(row[col["k"]])
            group = int(row[col["group"]]) if "group" in col and row[col["group"]].strip() else 1
        except (ValueError, IndexError): continue

        total += 1
        # For groupgemm: each kernel sees M_per_group and L=group
        is_group = (rtype == "groupgemm")
        m_ker = M // group if is_group else M
        l_ker = group      if is_group else 1
        m_total = M  # for GFLOPS calculation: always use total M

        results = {}

        if "fp8" in wanted:
            results["fp8"] = bench_fp8(
                m_ker, N, K, l_ker, args.warmup, args.iters, args.dry_run)

        if "dense_gemm" in wanted:
            results["dense_gemm"] = bench_dense_gemm(
                m_ker, N, K, l_ker, args.warmup, args.iters,
                args.skip_ref_check, args.tune, args.dry_run)

        if "pipeline_1cta" in wanted:
            results["pipeline_1cta"] = bench_pipeline(
                m_ker, N, K, l_ker, use_2cta=False,
                warmup=args.warmup, iters=args.iters,
                skip_ref=args.skip_ref_check, tune=args.tune, dry_run=args.dry_run)

        if "pipeline_2cta" in wanted:
            results["pipeline_2cta"] = bench_pipeline(
                m_ker, N, K, l_ker, use_2cta=True,
                warmup=args.warmup, iters=args.iters,
                skip_ref=args.skip_ref_check, tune=args.tune, dry_run=args.dry_run)

        if "ptr_array" in wanted:
            results["ptr_array"] = bench_ptr_array(
                m_ker, N, K, l_ker, args.warmup, args.iters,
                args.skip_ref_check, args.tune, args.dry_run)

        if "2sm" in wanted:
            results["2sm"] = bench_2sm(
                m_ker, N, K, l_ker, args.warmup, args.iters,
                args.skip_ref_check, args.dry_run)

        # Print row
        line = f"  {rtype:<12} {M:>6} {N:>6} {K:>6} {group:>4}"
        all_failed = True
        for key in col_keys:
            gf, err = results.get(key, (None, "skipped"))
            if gf is not None: all_failed = False
            line += f"  {fmt(gf, err)}"
        print(line)
        if all_failed: fail += 1

        # Write back (incremental)
        if not args.dry_run:
            for key, idx in out_idx.items():
                gf, _ = results.get(key, (None, None))
                if gf is not None:
                    row[idx] = f"{gf:.1f}"
            save_csv(csv_path, rows)

    print(c("-" * 110, BOLD))
    status = c("ALL OK", GREEN) if fail == 0 else c(f"{fail}/{total} rows all-failed", RED)
    print(f"  {total} shapes  {status}")
    if not args.dry_run:
        print(f"  Results written to: {csv_path}")
    print(c("=" * 110, BOLD)); print()
    sys.exit(1 if fail == total and total > 0 else 0)


if __name__ == "__main__":
    main()
