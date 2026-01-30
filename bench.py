#!/usr/bin/env python3
import os
import subprocess

PROFILER = "./tools/profiler/cutlass_profiler"
OUTDIR = "/home/tiger/cutlass/profile_result"

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


# Forward (fwd): W @ X
# - A = W has shape [m, k]
# - B = X has shape [k, n]   (n = tokens)
# - C/D = Y has shape [m, n]
#
# In GEMM notation: (m, n, k) = (out_features, tokens, in_features)
FWD_CASES = [
    (i, i, i // 8) for i in [2048, 4096, 8192, 16384, 32768]
] + [(i, i // 8, i) for i in [2048, 4096, 8192, 16384, 32768]]

# dgrad: dX = W^T @ dY
# - dY has shape [m, n]
# - W^T has shape [k, m]
# - dX has shape [k, n]
# GEMM(m, n, k) = (k, n, m)
DGRAD_CASES = [(K, N, M) for (M, N, K) in FWD_CASES]

# wgrad: dW = dY @ X^T
# - dY has shape [m, n]
# - X^T has shape [n, k]
# - dW has shape [m, k]
# GEMM(m, n, k) = (m, k, n)
WGRAD_CASES = [(M, K, N) for (M, N, K) in FWD_CASES]

MODES = [
    ("fwd", FWD_CASES, 0),
    # ("dgrad", DGRAD_CASES, 0),
    # ("wgrad", WGRAD_CASES, 0),
    # ("wgrad_accum", WGRAD_CASES, 1),  # accumulation into existing C (beta=1)
]


for mode, cases, beta in MODES:
    for M, N, K in cases:
        out_csv = os.path.join(
            OUTDIR,
            f"gemm_{mode}_{M}x{N}x{K}_itr={PROFILING_ITERS}_warmup={WARMUP_ITERS}_beta={beta}_dist=uminus005_uplus005.csv"
        )
        if beta == 0:
            cmd = [
                PROFILER,
                f"--profiling-iterations={PROFILING_ITERS}",
                f"--warmup-iterations={WARMUP_ITERS}",
                f"--operation={OP}",
                '--A="bf16:row" --B="bf16:row"',
                f"--beta={beta}",
                f"--m={M}",
                f"--n={N}",
                f"--k={K}",
                "--dist=" + DIST,
            ]
        else:
            cmd = [
                PROFILER,
                f"--profiling-iterations={PROFILING_ITERS}",
                f"--warmup-iterations={WARMUP_ITERS}",
                f"--operation={OP}",
                '--kernels="*bf16_bf16_f32*f32*"',
                '--A="bf16:row" --B="bf16:column" --C="f32:row" --D="f32:row"',
                f"--beta={beta}",
                f"--m={M}",
                f"--n={N}",
                f"--k={K}",
                "--dist=" + DIST,
            ]

        if SEED is not None:
            cmd.append(f"--seed={SEED}")

        cmd.append(f'--output={out_csv}')

        # print(f"\n=== Running GEMM: M={M}, N={N}, K={K}")
        print(" ".join(cmd))

    # subprocess.run(cmd, check=True, cwd="/root/workspace/zsh/cutlass/build_gemm")

print(f"\nAll done. Results saved to: {OUTDIR}")
