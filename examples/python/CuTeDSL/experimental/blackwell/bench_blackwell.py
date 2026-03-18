#!/usr/bin/env python3
"""
Benchmark script for CuTeDSL Blackwell (GB200/SM100) GEMM kernels.

Usage:
    # Quick smoke test (small sizes, 1 iteration):
    python bench_blackwell.py --mode quick

    # Standard benchmark sweep:
    python bench_blackwell.py --mode bench

    # Full sweep (more sizes, more iterations):
    python bench_blackwell.py --mode full

    # Single kernel only:
    python bench_blackwell.py --mode bench --kernel dense_gemm

    # Skip correctness check (faster):
    python bench_blackwell.py --mode bench --skip_ref_check

Available kernels:
    dense_gemm              - Base 1-SM dense GEMM
    dense_gemm_pipeline     - Persistent 1SM/2SM GEMM with CuTe pipeline
    dense_gemm_pipeline_2cta- Persistent GEMM with 2-CTA instructions + cluster
    dense_block_scaled_gemm - Block-scaled (FP8 quantized) GEMM
    dense_gemm_ptr_array    - Pointer-array batched GEMM
    dense_gemm_2sm          - 2-SM hardcoded demo (M=N=256,K=64)
"""

import argparse
import subprocess
import sys
import os
import time
from typing import List, Tuple, Optional

# ─── Color helpers ──────────────────────────────────────────────────────────
RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"

def color(text, c): return f"{c}{text}{RESET}"

# ─── TFLOPS calculator ───────────────────────────────────────────────────────
def tflops(m, n, k, l, ms):
    """2*M*N*K*L FLOPs for matmul, divided by elapsed ms."""
    if ms <= 0:
        return float("nan")
    return 2.0 * m * n * k * l / (ms * 1e-3) / 1e12

# ─── Subprocess runner ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script: str, args: List[str], timeout: int = 300) -> Tuple[int, str, str, float]:
    """Run a kernel script and return (returncode, stdout, stderr, wall_time_s)."""
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script)] + args
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=SCRIPT_DIR,
        )
        return result.returncode, result.stdout, result.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT", time.time() - t0
    except FileNotFoundError as e:
        return -2, "", str(e), time.time() - t0

def extract_exec_time_ms(stdout: str) -> Optional[float]:
    """Try to parse exec time from stdout.  Kernels print lines like:
       'Exec time: 1.234 ms'  or  'Time: 1.234 ms'  or  'time: 1.234 ms'
    """
    import re
    patterns = [
        r"[Ee]xec\s+time[:\s]+([0-9]+\.?[0-9]*)\s*ms",
        r"[Tt]ime[:\s]+([0-9]+\.?[0-9]*)\s*ms",
        r"([0-9]+\.?[0-9]*)\s*ms",
    ]
    for p in patterns:
        m = re.search(p, stdout)
        if m:
            return float(m.group(1))
    return None

# ─── Result printer ──────────────────────────────────────────────────────────
def print_header():
    print()
    print(color("=" * 110, BOLD))
    print(color("  CuTeDSL Blackwell GEMM Benchmark  (GB200 / SM100)", BOLD + CYAN))
    print(color("=" * 110, BOLD))
    fmt = f"  {'Kernel':<30} {'M':>6} {'N':>6} {'K':>6} {'L':>4} {'dtype':<10} {'ms':>8} {'TFLOPS':>8}  {'Status'}"
    print(color(fmt, BOLD))
    print(color("-" * 110, BOLD))

def print_row(kernel, m, n, k, l, dtype, ms, tf, status, notes=""):
    status_str = color("PASS", GREEN) if status == "PASS" \
        else color("SKIP", YELLOW) if status == "SKIP" \
        else color("FAIL", RED)
    ms_str  = f"{ms:.3f}" if ms is not None else "  —"
    tf_str  = f"{tf:.2f}" if tf is not None else "  —"
    notes_str = f"  [{notes}]" if notes else ""
    print(f"  {kernel:<30} {m:>6} {n:>6} {k:>6} {l:>4} {dtype:<10} {ms_str:>8} {tf_str:>8}  {status_str}{notes_str}")

def print_footer(results):
    print(color("-" * 110, BOLD))
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    print(f"  Total: {passed} PASS  {failed} FAIL  {skipped} SKIP  ({len(results)} cases)")
    if failed:
        print(color("  *** Some kernels FAILED — check stderr above ***", RED))
    print(color("=" * 110, BOLD))
    print()

# ─── Individual kernel runners ───────────────────────────────────────────────

def bench_dense_gemm(m, n, k, l, dtype="Float32", tiler_mn="128,128",
                     warmup=2, iters=5, skip_ref=False, extra_args=None):
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--mma_tiler_mn", tiler_mn,
        "--ab_dtype", dtype,
        "--d_dtype", dtype,
        "--warmup_iterations", str(warmup),
        "--iterations", str(iters),
        "--use_cold_l2",
    ]
    if skip_ref:
        args.append("--skip_ref_check")
    if extra_args:
        args.extend(extra_args)
    rc, out, err, wall = run_script("dense_gemm.py", args)
    ms = extract_exec_time_ms(out)
    tf = tflops(m, n, k, l, ms) if ms else None
    status = "PASS" if rc == 0 and ("PASS" in out or skip_ref) \
             else "SKIP" if "SKIP" in out \
             else "FAIL"
    return dict(kernel="dense_gemm", m=m, n=n, k=k, l=l, dtype=dtype,
                ms=ms, tflops=tf, status=status, stdout=out, stderr=err)

def bench_dense_gemm_pipeline(m, n, k, l, dtype="TFloat32", tiler_mn="128,128",
                               use_2cta=False, cluster_mn="1,1",
                               warmup=2, iters=5, skip_ref=False, extra_args=None):
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--mma_tiler_mn", tiler_mn,
        "--cluster_shape_mn", cluster_mn,
        "--ab_dtype", dtype,
        "--warmup_iterations", str(warmup),
        "--iterations", str(iters),
        "--benchmark", "default",
        "--use_cold_l2",
    ]
    if use_2cta:
        args.append("--use_2cta_instrs")
    if skip_ref:
        args.append("--skip_ref_check")
    if extra_args:
        args.extend(extra_args)
    rc, out, err, wall = run_script("dense_gemm_cute_pipeline.py", args)
    ms = extract_exec_time_ms(out)
    tf = tflops(m, n, k, l, ms) if ms else None
    status = "PASS" if rc == 0 and ("PASS" in out or skip_ref) \
             else "SKIP" if "SKIP" in out \
             else "FAIL"
    name = "pipeline_2cta" if use_2cta else "pipeline_1cta"
    return dict(kernel=name, m=m, n=n, k=k, l=l, dtype=dtype,
                ms=ms, tflops=tf, status=status, stdout=out, stderr=err)

def bench_block_scaled_gemm(m, n, k, l, ab_dtype="Float8E4M3FN", sf_dtype="Float8E8M0FNU",
                             tiler_mn="128,128", skip_ref=False, extra_args=None):
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--mma_tiler_mn", tiler_mn,
        "--ab_dtype", ab_dtype,
        "--sf_dtype", sf_dtype,
        "--d_dtype", "Float16",
    ]
    if skip_ref:
        args.append("--skip_ref_check")
    if extra_args:
        args.extend(extra_args)
    rc, out, err, wall = run_script("dense_block_scaled_gemm.py", args)
    ms = extract_exec_time_ms(out)
    tf = tflops(m, n, k, l, ms) if ms else None
    status = "PASS" if rc == 0 and ("PASS" in out or skip_ref) \
             else "SKIP" if "SKIP" in out \
             else "FAIL"
    dtype_label = f"{ab_dtype}+sf"
    return dict(kernel="block_scaled", m=m, n=n, k=k, l=l, dtype=dtype_label,
                ms=ms, tflops=tf, status=status, stdout=out, stderr=err)

def bench_ptr_array_gemm(m, n, k, l, dtype="Float32", tiler_mn="128,128",
                          warmup=2, iters=5, skip_ref=False, extra_args=None):
    args = [
        "--mnkl", f"{m},{n},{k},{l}",
        "--mma_tiler_mn", tiler_mn,
        "--ab_dtype", dtype,
        "--d_dtype", dtype,
        "--warmup_iterations", str(warmup),
        "--iterations", str(iters),
        "--use_cold_l2",
    ]
    if skip_ref:
        args.append("--skip_ref_check")
    if extra_args:
        args.extend(extra_args)
    rc, out, err, wall = run_script("dense_gemm_ptr_array.py", args)
    ms = extract_exec_time_ms(out)
    tf = tflops(m, n, k, l, ms) if ms else None
    status = "PASS" if rc == 0 and ("PASS" in out or skip_ref) \
             else "SKIP" if "SKIP" in out \
             else "FAIL"
    return dict(kernel="ptr_array", m=m, n=n, k=k, l=l, dtype=dtype,
                ms=ms, tflops=tf, status=status, stdout=out, stderr=err)

def bench_dense_gemm_2sm():
    """dense_gemm_2sm.py has no CLI args — runs fixed M=N=256, K=64."""
    rc, out, err, wall = run_script("dense_gemm_2sm.py", [])
    status = "PASS" if rc == 0 and "PASS" in out \
             else "SKIP" if "SKIP" in out \
             else "FAIL"
    return dict(kernel="dense_gemm_2sm", m=256, n=256, k=64, l=1, dtype="Float16",
                ms=None, tflops=None, status=status, stdout=out, stderr=err)

# ─── Benchmark suites ────────────────────────────────────────────────────────

# Problem shapes: (M, N, K, L)  — covers square, LLM-like, and batched cases
QUICK_SHAPES = [
    (256, 256, 256, 1),
    (1024, 1024, 1024, 1),
]

BENCH_SHAPES = [
    (256,  256,  256,  1),
    (512,  512,  512,  1),
    (1024, 1024, 1024, 1),
    (2048, 2048, 2048, 1),
    (4096, 4096, 4096, 1),
    # LLM-like: typical FFN projections
    (4096, 4096, 16384, 1),
    (8192, 8192, 8192,  1),
    # Batched
    (1024, 1024, 1024, 4),
]

FULL_SHAPES = BENCH_SHAPES + [
    (16384, 16384, 16384, 1),
    (4096,  4096,  4096,  8),
    (512,   512,   512,  16),
]

def run_suite(shapes, skip_ref, warmup, iters, kernels, verbose):
    results = []

    def maybe_run(name, fn):
        if kernels and name not in kernels:
            return
        r = fn()
        results.append(r)
        print_row(r["kernel"], r["m"], r["n"], r["k"], r["l"], r["dtype"],
                  r.get("ms"), r.get("tflops"), r["status"])
        if verbose and (r["status"] == "FAIL"):
            print(color("  STDOUT:", YELLOW))
            for line in r["stdout"].splitlines()[-10:]:
                print("    " + line)
            print(color("  STDERR:", RED))
            for line in r["stderr"].splitlines()[-10:]:
                print("    " + line)

    # ── dense_gemm_2sm (fixed shape) ────────────────────────────────────────
    maybe_run("dense_gemm_2sm", bench_dense_gemm_2sm)

    # ── Per-shape kernels ────────────────────────────────────────────────────
    for (m, n, k, l) in shapes:
        # dense_gemm  — fp32 default
        maybe_run("dense_gemm", lambda m=m,n=n,k=k,l=l: bench_dense_gemm(
            m, n, k, l, dtype="Float32", warmup=warmup, iters=iters, skip_ref=skip_ref))

        # dense_gemm  — fp16
        maybe_run("dense_gemm", lambda m=m,n=n,k=k,l=l: bench_dense_gemm(
            m, n, k, l, dtype="Float16", warmup=warmup, iters=iters, skip_ref=skip_ref))

        # dense_gemm_pipeline — 1-CTA tf32
        maybe_run("pipeline_1cta", lambda m=m,n=n,k=k,l=l: bench_dense_gemm_pipeline(
            m, n, k, l, dtype="TFloat32", use_2cta=False,
            warmup=warmup, iters=iters, skip_ref=skip_ref))

        # dense_gemm_pipeline — 2-CTA fp16, cluster (2,1)
        maybe_run("pipeline_2cta", lambda m=m,n=n,k=k,l=l: bench_dense_gemm_pipeline(
            m, n, k, l, dtype="Float16", use_2cta=True,
            tiler_mn="256,256", cluster_mn="2,1",
            warmup=warmup, iters=iters, skip_ref=skip_ref))

        # block_scaled — fp8 + E8M0 scale
        maybe_run("block_scaled", lambda m=m,n=n,k=k,l=l: bench_block_scaled_gemm(
            m, n, k, l,
            ab_dtype="Float8E4M3FN", sf_dtype="Float8E8M0FNU",
            skip_ref=skip_ref))

        # ptr_array — fp32
        maybe_run("ptr_array", lambda m=m,n=n,k=k,l=l: bench_ptr_array_gemm(
            m, n, k, l, dtype="Float32", warmup=warmup, iters=iters, skip_ref=skip_ref))

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CuTeDSL Blackwell GEMM kernels on GB200.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["quick", "bench", "full"], default="bench",
        help="quick=small sizes/1 iter; bench=standard sweep; full=extended sweep",
    )
    parser.add_argument(
        "--kernel",
        nargs="+",
        choices=["dense_gemm", "pipeline_1cta", "pipeline_2cta",
                 "block_scaled", "ptr_array", "dense_gemm_2sm"],
        default=None,
        help="Run only specified kernel(s). Default: all.",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true",
        help="Skip numerical correctness check (faster).",
    )
    parser.add_argument(
        "--warmup", type=int, default=None,
        help="Warmup iterations (default: 1 for quick, 3 for bench/full).",
    )
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Benchmark iterations (default: 1 for quick, 10 for bench, 20 for full).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print stdout/stderr of failed kernels.",
    )
    args = parser.parse_args()

    # defaults per mode
    mode_defaults = {
        "quick": dict(shapes=QUICK_SHAPES, warmup=1,  iters=1),
        "bench": dict(shapes=BENCH_SHAPES, warmup=3,  iters=10),
        "full":  dict(shapes=FULL_SHAPES,  warmup=5,  iters=20),
    }
    cfg = mode_defaults[args.mode]
    warmup = args.warmup if args.warmup is not None else cfg["warmup"]
    iters  = args.iters  if args.iters  is not None else cfg["iters"]
    shapes = cfg["shapes"]

    print_header()
    results = run_suite(
        shapes=shapes,
        skip_ref=args.skip_ref_check,
        warmup=warmup,
        iters=iters,
        kernels=args.kernel,
        verbose=args.verbose,
    )
    print_footer(results)

    # Exit non-zero if any failure
    sys.exit(1 if any(r["status"] == "FAIL" for r in results) else 0)


if __name__ == "__main__":
    main()
