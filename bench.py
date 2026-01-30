#!/usr/bin/env python3
import os
import subprocess

PROFILER = "./tools/profiler/cutlass_profiler"
OUTDIR = "/root/workspace/zsh/cutlass/profile_result"

# 固定参数
PROFILING_ITERS = 10
WARMUP_ITERS = 3
# CTA_M = 128
# CTA_N = 256
OP = "Gemm"

# 你要的 input distribution: (rand - 0.5)/10  ~= uniform[-0.05, 0.05]
DIST = '"uniform,min:-0.05,max:0.05,scale:-1"'

# 可选：固定随机种子，便于可复现（不想固定就设为 None）
SEED = 2026

os.makedirs(OUTDIR, exist_ok=True)

# 图中所有测试用例 (M, N, K)
CASES = [
    (8192, 7168, 5120),
    (4096, 7168, 5120),
    (8192, 5120, 5120),
    (4096, 5120, 5120),
    (8192, 9216, 7168),
    (4096, 9216, 7168),
    (8192, 7168, 7168),
    (4096, 7168, 7168),
    (8192, 155136, 5120),
]

for M, N, K in CASES:
    out_csv = os.path.join(
        OUTDIR,
        f"gemm_{M}x{N}x{K}_itr={PROFILING_ITERS}_warmup={WARMUP_ITERS}_dist=uminus005_uplus005.csv"
    )

    cmd = [
        PROFILER,
        f"--profiling-iterations={PROFILING_ITERS}",
        f"--warmup-iterations={WARMUP_ITERS}",
        f"--operation={OP}",
        '--kernels="*stream_k*"',
        '--A="bf16:row" --B="bf16:column"',
        f"--m={M}",
        f"--n={N}",
        f"--k={K}",
        "--dist=" + DIST,
        "--sort-results=1"
    ]

    # print(cmd.join(""))
    # print(" ".join(cmd))
    if SEED is not None:
        cmd.append(f"--seed={SEED}")

    cmd.append(f'--output={out_csv}')

    # print(f"\n=== Running GEMM: M={M}, N={N}, K={K}")
    print(" ".join(cmd))

    # subprocess.run(cmd, check=True, cwd="/root/workspace/zsh/cutlass/build_gemm")

print(f"\nAll done. Results saved to: {OUTDIR}")
