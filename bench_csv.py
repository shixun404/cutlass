#!/usr/bin/env python3
"""
Benchmark runner for CuTeDSL Blackwell (SM100/GB200) kernels.

Reads benchmark.csv, runs every (type, M, N, K, group) shape through all
relevant experimental/blackwell kernels, and writes GFLOPS results back.

Kernel / dtype coverage:
  fp8_e4m3          → dense_block_scaled_gemm  (Float8E4M3FN + E8M0 scale)
  fp8_e5m2          → dense_block_scaled_gemm  (Float8E5M2   + E8M0 scale)
  dense_gemm_bf16   → dense_gemm               (BFloat16, 1-CTA)
  dense_gemm_fp16   → dense_gemm               (Float16,  1-CTA)
  pipeline_1cta_bf16→ dense_gemm_cute_pipeline (BFloat16, 1-CTA)
  pipeline_1cta_fp16→ dense_gemm_cute_pipeline (Float16,  1-CTA)
  pipeline_2cta_bf16→ dense_gemm_cute_pipeline (BFloat16, 2-CTA, tile 256×256)
  pipeline_2cta_fp16→ dense_gemm_cute_pipeline (Float16,  2-CTA, tile 256×256)
  ptr_array_bf16    → dense_gemm_ptr_array      (BFloat16, ptr-array batched)
  ptr_array_fp16    → dense_gemm_ptr_array      (Float16,  ptr-array batched)
  2sm_bf16          → dense_gemm_2sm            (BFloat16, 2-SM cluster)
  2sm_fp16          → dense_gemm_2sm            (Float16,  2-SM cluster)

MFU reference (B200 dense, no sparsity):
  FP8  Tensor Core: ~4,500 TFLOPS
  BF16/FP16 Tensor Core: ~2,250 TFLOPS

Tile tuning:
  Before benchmarking each (kernel, M, N, K) the script sweeps candidate
  mma_tiler_mn values and picks the fastest one (--tune / --no_tune).

Usage:
  python bench_csv.py                          # full run, tuning ON
  python bench_csv.py --no_tune                # skip tuning, use tile (128,128)
  python bench_csv.py --warmup 3 --iters 10   # custom iteration counts
  python bench_csv.py --dry_run               # print commands without executing
  python bench_csv.py --kernel fp8 dense_gemm # run only these kernel groups
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

# ─── B200 peak GFLOPS (dense, no sparsity) ────────────────────────────────────
PEAK_GFLOPS = {
    "fp8":  4_500_000,   # FP8 Tensor Core
    "fp16": 2_250_000,   # FP16 / BF16 Tensor Core
}

# ─── ANSI ─────────────────────────────────────────────────────────────────────
RESET = "\033[0m"; GREEN = "\033[92m"; RED = "\033[91m"
YELLOW = "\033[93m"; CYAN = "\033[96m"; BOLD = "\033[1m"
def c(t, col): return f"{col}{t}{RESET}"

# ─── GFLOPS / MFU ─────────────────────────────────────────────────────────────
def to_gflops(m, n, k, l, us):
    """2*M*N*K*L FLOPs, latency in µs → GFLOPS."""
    return 2.0 * m * n * k * l / (us * 1e-6) / 1e9

# ─── Subprocess helper ─────────────────────────────────────────────────────────
def _run(cmd, timeout=600):
    env = os.environ.copy()
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
    """pipeline prints µs with a mislabeled 'seconds' unit → return value as-is (µs)."""
    m = re.search(r"Execution time:\s*([0-9.e+\-]+)\s*seconds", stdout)
    return float(m.group(1)) if m else None   # value IS µs despite the label

def _parse_fp8_us(stdout):
    """'exec_time_us=X' from inline fp8 timer → float µs."""
    m = re.search(r"exec_time_us=([0-9.e+\-]+)", stdout)
    return float(m.group(1)) if m else None

# ─── Tuning helpers ───────────────────────────────────────────────────────────
# BF16/FP16 MMA (MmaF16BF16Op) requires M_tile ∈ {64, 128} — NOT 256.
# Using M_tile=256 raises "expects the M-mode to be 64 or 128" at runtime.
# (256 is only valid for 2-CTA SM100_MMA_2SM instructions.)
_TILE_CANDIDATES_1CTA = [
    (128, 128), (128, 256),
]


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
        lines = ((err or out) or "").strip().splitlines()
        msg = lines[-1] if lines else f"rc={rc}"
        print(f"\n[ERROR] {script}: {msg}", file=sys.stderr)
        if len(lines) > 1:
            print("\n".join(f"  {l}" for l in lines[-5:]), file=sys.stderr)
        return None, msg
    return us, None


# ── dense_block_scaled_gemm.py (FP8 E4M3 or E5M2) ───────────────────────────

def _fp8_timing_script(m, n, k, l, warmup, iters, ab_dtype_str="Float8E4M3FN"):
    """Generate a self-contained FP8 timing script using CUDA Events."""
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
        ab_dtype   = cutlass.{ab_dtype_str}
        sf_dtype   = cutlass.Float8E8M0FNU
        sf_vec_size = 32
        d_dtype    = cutlass.Float16
        acc_dtype  = cutlass.Float32
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

def bench_fp8(m, n, k, l, warmup, iters, dry_run, ab_dtype="Float8E4M3FN"):
    if m % 128 != 0:
        return None, f"M={m} not divisible by 128"
    if n % 128 != 0:
        return None, f"N={n} not divisible by 128"
    if k % 64 != 0:
        return None, f"K={k} not divisible by 64 (sf_vec_size×2)"

    if dry_run:
        return None, f"DRY: inline fp8 timing script ({ab_dtype})"

    script_text = _fp8_timing_script(m, n, k, l, warmup, iters, ab_dtype)
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


# ── dense_gemm.py (BF16 or FP16, 1-CTA) ─────────────────────────────────────

def _dense_gemm_args(m, n, k, l, tile, warmup, iters, skip_ref, ab_dtype="BFloat16"):
    tm, tn = tile
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--mma_tiler_mn", f"{tm},{tn}",
        "--cluster_shape_mn", "1,1",
        "--ab_dtype", ab_dtype, "--d_dtype", ab_dtype, "--acc_dtype", "Float32",
        "--warmup_iterations", str(warmup), "--iterations", str(iters),
        "--use_cold_l2",
    ]
    if skip_ref: args.append("--skip_ref_check")
    return args

def bench_dense_gemm(m, n, k, l, warmup, iters, skip_ref, tune, dry_run,
                     ab_dtype="BFloat16"):
    tiles = _valid_tiles(m, n, _TILE_CANDIDATES_1CTA) if tune else [(128, 128)]
    if not tiles:
        return None, f"No valid tile for M={m}, N={n}"
    if dry_run:
        return None, "DRY: " + " ".join(
            _dense_gemm_args(m, n, k, l, tiles[0], warmup, iters, skip_ref, ab_dtype))

    best_us, best_tile = None, tiles[0]
    for tile in tiles:
        us, _ = _subprocess_run("dense_gemm.py",
                                _dense_gemm_args(m, n, k, l, tile, 1, 3, True, ab_dtype),
                                _parse_us)
        if us is not None and (best_us is None or us < best_us):
            best_us, best_tile = us, tile

    if best_us is None:
        return None, "all tiles failed in tune sweep"
    us, err = _subprocess_run("dense_gemm.py",
                              _dense_gemm_args(m, n, k, l, best_tile, warmup, iters,
                                               skip_ref, ab_dtype),
                              _parse_us)
    if us is None: return None, err
    return to_gflops(m, n, k, l, us), None


# ── dense_gemm_cute_pipeline.py (BF16 or FP16, 1-CTA or 2-CTA) ──────────────

def _pipeline_args(m, n, k, l, tile, cluster, use_2cta, warmup, iters, skip_ref,
                   ab_dtype="BFloat16"):
    tm, tn = tile
    cm, cn = cluster
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--mma_tiler_mn", f"{tm},{tn}",
        "--cluster_shape_mn", f"{cm},{cn}",
        "--ab_dtype", ab_dtype, "--c_dtype", ab_dtype, "--acc_dtype", "Float32",
        "--warmup_iterations", str(warmup), "--iterations", str(iters),
        "--benchmark", "default", "--use_cold_l2",
    ]
    if use_2cta: args.append("--use_2cta_instrs")
    if skip_ref: args.append("--skip_ref_check")
    return args

def bench_pipeline(m, n, k, l, use_2cta, warmup, iters, skip_ref, tune, dry_run,
                   ab_dtype="BFloat16"):
    if use_2cta:
        if m % 256 != 0: return None, f"M={m} not divisible by 256 (2-CTA)"
        if n % 256 != 0: return None, f"N={n} not divisible by 256 (2-CTA)"
        tile, cluster = (256, 256), (2, 1)
        if dry_run:
            return None, "DRY: " + " ".join(
                _pipeline_args(m, n, k, l, tile, cluster, True, warmup, iters,
                               skip_ref, ab_dtype))
        us, err = _subprocess_run("dense_gemm_cute_pipeline.py",
                                  _pipeline_args(m, n, k, l, tile, cluster,
                                                 True, warmup, iters, skip_ref, ab_dtype),
                                  _parse_sec)
        if us is None: return None, err
        return to_gflops(m, n, k, l, us), None

    tiles = _valid_tiles(m, n, _TILE_CANDIDATES_1CTA) if tune else [(128, 128)]
    if not tiles: return None, f"No valid 1-CTA tile for M={m}, N={n}"
    if dry_run:
        return None, "DRY: " + " ".join(
            _pipeline_args(m, n, k, l, tiles[0], (1, 1), False, warmup, iters,
                           skip_ref, ab_dtype))

    best_us, best_tile = None, tiles[0]
    for tile in tiles:
        us, _ = _subprocess_run("dense_gemm_cute_pipeline.py",
                                _pipeline_args(m, n, k, l, tile, (1, 1),
                                               False, 1, 3, True, ab_dtype),
                                _parse_sec)
        if us is not None and (best_us is None or us < best_us):
            best_us, best_tile = us, tile
    if best_us is None: return None, "all tiles failed in tune sweep"

    us, err = _subprocess_run("dense_gemm_cute_pipeline.py",
                              _pipeline_args(m, n, k, l, best_tile, (1, 1),
                                             False, warmup, iters, skip_ref, ab_dtype),
                              _parse_sec)
    if us is None: return None, err
    return to_gflops(m, n, k, l, us), None


# ── dense_gemm_ptr_array.py (BF16 or FP16, ptr-array batched) ────────────────

def _ptr_array_args(m, n, k, l, tile, warmup, iters, skip_ref, ab_dtype="BFloat16"):
    tm, tn = tile
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--mma_tiler_mn", f"{tm},{tn}",
        "--cluster_shape_mn", "1,1",
        "--ab_dtype", ab_dtype, "--d_dtype", ab_dtype, "--acc_dtype", "Float32",
        "--warmup_iterations", str(warmup), "--iterations", str(iters),
        "--use_cold_l2",
    ]
    if skip_ref: args.append("--skip_ref_check")
    return args

def bench_ptr_array(m, n, k, l, warmup, iters, skip_ref, tune, dry_run,
                    ab_dtype="BFloat16"):
    tiles = _valid_tiles(m, n, _TILE_CANDIDATES_1CTA) if tune else [(128, 128)]
    if not tiles: return None, f"No valid tile for M={m}, N={n}"
    if dry_run:
        return None, "DRY: " + " ".join(
            _ptr_array_args(m, n, k, l, tiles[0], warmup, iters, skip_ref, ab_dtype))

    best_us, best_tile = None, tiles[0]
    for tile in tiles:
        us, _ = _subprocess_run("dense_gemm_ptr_array.py",
                                _ptr_array_args(m, n, k, l, tile, 1, 3, True, ab_dtype),
                                _parse_us)
        if us is not None and (best_us is None or us < best_us):
            best_us, best_tile = us, tile
    if best_us is None: return None, "all tiles failed in tune sweep"

    us, err = _subprocess_run("dense_gemm_ptr_array.py",
                              _ptr_array_args(m, n, k, l, best_tile, warmup, iters,
                                              skip_ref, ab_dtype),
                              _parse_us)
    if us is None: return None, err
    return to_gflops(m, n, k, l, us), None


# ── dense_gemm_2sm.py (BF16 or FP16, 2-SM cluster) ───────────────────────────
# NOTE: known kernel bug for K > 256 (pipeline deadlock on peer CTA).
# All benchmark shapes have K ≥ 2048 so this will return N/A.

def _2sm_args(m, n, k, l, use_2cta, warmup, iters, skip_ref, ab_dtype="BFloat16"):
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--ab_dtype", ab_dtype, "--d_dtype", ab_dtype, "--acc_dtype", "Float32",
        "--warmup_iterations", str(warmup), "--iterations", str(iters),
    ]
    if use_2cta: args.append("--use_2cta_instrs")
    else:        args.append("--no_2cta_instrs")
    if skip_ref: args.append("--skip_ref_check")
    return args

def bench_2sm(m, n, k, l, warmup, iters, skip_ref, dry_run, ab_dtype="BFloat16"):
    use_2cta = (m % 256 == 0 and n % 256 == 0 and k % 64 == 0)
    use_1cta = (m % 128 == 0 and n % 256 == 0 and k % 64 == 0)
    if not use_2cta and not use_1cta:
        return None, f"M={m}/N={n}/K={k} violates 2SM alignment"

    chosen_2cta = use_2cta
    if dry_run:
        return None, "DRY: " + " ".join(
            _2sm_args(m, n, k, l, chosen_2cta, warmup, iters, skip_ref, ab_dtype))

    us, err = _subprocess_run("dense_gemm_2sm.py",
                              _2sm_args(m, n, k, l, chosen_2cta, warmup, iters,
                                        skip_ref, ab_dtype),
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
# (csv_column_name, bench_key, peak_category)
# peak_category is "fp8" or "fp16" — used for MFU% display.
OUTPUT_COLS = [
    ("gflops_fp8_block_scaled",      "fp8_e4m3",           "fp8"),
    ("gflops_fp8e5m2_block_scaled",  "fp8_e5m2",           "fp8"),
    ("gflops_bf16_dense_gemm",       "dense_gemm_bf16",    "fp16"),
    ("gflops_fp16_dense_gemm",       "dense_gemm_fp16",    "fp16"),
    ("gflops_bf16_pipeline_1cta",    "pipeline_1cta_bf16", "fp16"),
    ("gflops_fp16_pipeline_1cta",    "pipeline_1cta_fp16", "fp16"),
    ("gflops_bf16_pipeline_2cta",    "pipeline_2cta_bf16", "fp16"),
    ("gflops_fp16_pipeline_2cta",    "pipeline_2cta_fp16", "fp16"),
    ("gflops_bf16_ptr_array",        "ptr_array_bf16",     "fp16"),
    ("gflops_fp16_ptr_array",        "ptr_array_fp16",     "fp16"),
    ("gflops_bf16_2sm",              "2sm_bf16",           "fp16"),
    ("gflops_fp16_2sm",              "2sm_fp16",           "fp16"),
]

# Logical group names for --kernel shorthand
_KERNEL_GROUPS = {
    "fp8":         {"fp8_e4m3", "fp8_e5m2"},
    "dense_gemm":  {"dense_gemm_bf16", "dense_gemm_fp16"},
    "pipeline_1cta":{"pipeline_1cta_bf16", "pipeline_1cta_fp16"},
    "pipeline_2cta":{"pipeline_2cta_bf16", "pipeline_2cta_fp16"},
    "ptr_array":   {"ptr_array_bf16", "ptr_array_fp16"},
    "2sm":         {"2sm_bf16", "2sm_fp16"},
}


def _run_key(key, m_ker, N, K, l_ker, warmup, iters, skip_ref, tune, dry_run):
    """Dispatch a bench key to the appropriate bench function."""
    if key == "fp8_e4m3":
        return bench_fp8(m_ker, N, K, l_ker, warmup, iters, dry_run,
                         ab_dtype="Float8E4M3FN")
    if key == "fp8_e5m2":
        return bench_fp8(m_ker, N, K, l_ker, warmup, iters, dry_run,
                         ab_dtype="Float8E5M2")
    if key == "dense_gemm_bf16":
        return bench_dense_gemm(m_ker, N, K, l_ker, warmup, iters,
                                skip_ref, tune, dry_run, ab_dtype="BFloat16")
    if key == "dense_gemm_fp16":
        return bench_dense_gemm(m_ker, N, K, l_ker, warmup, iters,
                                skip_ref, tune, dry_run, ab_dtype="Float16")
    if key == "pipeline_1cta_bf16":
        return bench_pipeline(m_ker, N, K, l_ker, use_2cta=False,
                              warmup=warmup, iters=iters, skip_ref=skip_ref,
                              tune=tune, dry_run=dry_run, ab_dtype="BFloat16")
    if key == "pipeline_1cta_fp16":
        return bench_pipeline(m_ker, N, K, l_ker, use_2cta=False,
                              warmup=warmup, iters=iters, skip_ref=skip_ref,
                              tune=tune, dry_run=dry_run, ab_dtype="Float16")
    if key == "pipeline_2cta_bf16":
        return bench_pipeline(m_ker, N, K, l_ker, use_2cta=True,
                              warmup=warmup, iters=iters, skip_ref=skip_ref,
                              tune=tune, dry_run=dry_run, ab_dtype="BFloat16")
    if key == "pipeline_2cta_fp16":
        return bench_pipeline(m_ker, N, K, l_ker, use_2cta=True,
                              warmup=warmup, iters=iters, skip_ref=skip_ref,
                              tune=tune, dry_run=dry_run, ab_dtype="Float16")
    if key == "ptr_array_bf16":
        return bench_ptr_array(m_ker, N, K, l_ker, warmup, iters,
                               skip_ref, tune, dry_run, ab_dtype="BFloat16")
    if key == "ptr_array_fp16":
        return bench_ptr_array(m_ker, N, K, l_ker, warmup, iters,
                               skip_ref, tune, dry_run, ab_dtype="Float16")
    if key == "2sm_bf16":
        return bench_2sm(m_ker, N, K, l_ker, warmup, iters, skip_ref, dry_run,
                         ab_dtype="BFloat16")
    if key == "2sm_fp16":
        return bench_2sm(m_ker, N, K, l_ker, warmup, iters, skip_ref, dry_run,
                         ab_dtype="Float16")
    return None, f"unknown key: {key}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_groups = sorted(_KERNEL_GROUPS)
    all_keys   = [key for _, key, _ in OUTPUT_COLS]

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv",    default=str(DEFAULT_CSV), help="CSV path")
    ap.add_argument("--warmup", type=int, default=3,  help="Warmup iterations (default 3)")
    ap.add_argument("--iters",  type=int, default=10, help="Benchmark iterations (default 10)")
    ap.add_argument("--skip_ref_check", action="store_true", default=True)
    ap.add_argument("--tune",    dest="tune", action="store_true",  default=True)
    ap.add_argument("--no_tune", dest="tune", action="store_false",
                    help="Skip tile tuning, use default tile (128,128)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--kernel", nargs="+",
                    choices=all_groups,
                    default=None,
                    help=("Kernel group(s) to run. Each group covers both BF16 and FP16. "
                          f"Choices: {all_groups}"))
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

    # Expand group names → individual keys
    if args.kernel:
        wanted = set()
        for g in args.kernel:
            wanted |= _KERNEL_GROUPS.get(g, {g})
    else:
        wanted = set(all_keys)

    # Ensure output columns exist in CSV
    out_idx = {}
    for col_name, key, _ in OUTPUT_COLS:
        if key in wanted:
            out_idx[key] = ensure_col(rows, col_name)

    # Peak GFLOPS lookup per key
    peak_map = {key: PEAK_GFLOPS[cat] for _, key, cat in OUTPUT_COLS}

    # Print header
    print(); print(c("=" * 130, BOLD))
    print(c(f"  CuTeDSL Blackwell Benchmark  ←  experimental/blackwell"
            f"  (tune={'ON' if args.tune else 'OFF'})"
            f"  [B200 peaks: FP8={PEAK_GFLOPS['fp8']//1000}T  BF16/FP16={PEAK_GFLOPS['fp16']//1000}T  GFLOPS]",
            BOLD + CYAN))
    print(c("=" * 130, BOLD))
    col_keys = [key for _, key, _ in OUTPUT_COLS if key in wanted]
    hdr = f"  {'type':<12} {'M':>6} {'N':>6} {'K':>6} {'grp':>4}"
    for k in col_keys:
        hdr += f"  {k:>22}"
    print(c(hdr, BOLD)); print(c("-" * 130, BOLD))

    def fmt(val, err, peak):
        if val is not None:
            mfu = val / peak * 100
            return c(f"{val:>10.0f}", GREEN) + f"({mfu:4.1f}%)"
        short = (err or "")[:16]
        return c(f"{'N/A':>10}", YELLOW) + f"({short:<16})"

    total = fail = 0
    for row in rows[1:]:
        if not row or not row[0].strip(): continue
        rtype = row[col["type"]].strip()
        try:
            M = int(row[col["m"]]); N = int(row[col["n"]]); K = int(row[col["k"]])
            group = int(row[col["group"]]) if "group" in col and row[col["group"]].strip() else 1
        except (ValueError, IndexError): continue

        total += 1
        is_group = (rtype == "groupgemm")
        m_ker = M // group if is_group else M
        l_ker = group      if is_group else 1

        results = {}
        for key in col_keys:
            results[key] = _run_key(key, m_ker, N, K, l_ker,
                                    args.warmup, args.iters,
                                    args.skip_ref_check, args.tune, args.dry_run)

        line = f"  {rtype:<12} {M:>6} {N:>6} {K:>6} {group:>4}"
        all_failed = True
        for key in col_keys:
            gf, err = results.get(key, (None, "skipped"))
            if gf is not None: all_failed = False
            line += f"  {fmt(gf, err, peak_map[key])}"
        print(line)
        if all_failed: fail += 1

        if not args.dry_run:
            for key, idx in out_idx.items():
                gf, _ = results.get(key, (None, None))
                if gf is not None:
                    row[idx] = f"{gf:.1f}"
            save_csv(csv_path, rows)

    print(c("-" * 130, BOLD))
    status = c("ALL OK", GREEN) if fail == 0 else c(f"{fail}/{total} rows all-failed", RED)
    print(f"  {total} shapes  {status}")
    if not args.dry_run:
        print(f"  Results written to: {csv_path}")
    print(c("=" * 130, BOLD)); print()
    sys.exit(1 if fail == total and total > 0 else 0)


if __name__ == "__main__":
    main()
