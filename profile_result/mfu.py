#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# 所有 csv 文件（按你给的清单）
CSV_FILES = [
    "gemm_4096x5120x5120_itr=10_warmup=3_dist=uminus005_uplus005.gemm.csv",
    "gemm_4096x7168x5120_itr=10_warmup=3_dist=uminus005_uplus005.gemm.csv",
    "gemm_4096x7168x7168_itr=10_warmup=3_dist=uminus005_uplus005.gemm.csv",
    "gemm_4096x9216x7168_itr=10_warmup=3_dist=uminus005_uplus005.gemm.csv",
    "gemm_8192x155136x5120.gemm.csv",
    "gemm_8192x155136x5120_itr=10_warmup=3_dist=uminus005_uplus005.gemm.csv",
    "gemm_8192x155136x5120_itr=10_warmup=3.gemm.csv",
    "gemm_8192x5120x5120_itr=10_warmup=3_dist=uminus005_uplus005.gemm.csv",
    "gemm_8192x7168x5120_itr=10_warmup=3_dist=uminus005_uplus005.gemm.csv",
    "gemm_8192x7168x7168_itr=10_warmup=3_dist=uminus005_uplus005.gemm.csv",
    "gemm_8192x9216x7168_itr=10_warmup=3_dist=uminus005_uplus005.gemm.csv",
    "moe_profiling_example92_exp=16_tk=30720_results.csv",
    "moe_profiling_example92_exp=256_results.csv",
    "moe_profiling_example92_exp=256_tk=30720_results.csv",
    "moe_profiling_example92_exp=96_tk=32768_results.csv",
    "moe_profiling_example92_rcgrouped_exp=16_tk=30720_results.csv",
    "moe_profiling_example92_rcgrouped_exp=96_tk=32768_results.csv",
    "moe_profiling_example92_results.csv",
    "moe_profiling_grouped_gemm_decode_exp=16_tk=30720_results.csv",
    "moe_profiling_grouped_gemm_decode_exp=256_results.csv",
    "moe_profiling_grouped_gemm_decode_exp=96_tk=32768_results.csv",
    "moe_profiling_grouped_gemm_decode_results.csv",
    "moe_profiling_results.csv",
]

DIVISOR = 2500000.0

for csv_path in CSV_FILES:
    path = Path(csv_path)

    if not path.exists():
        print(f"[SKIP] Not found: {path}")
        continue

    print(f"[PROCESS] {path}")

    df = pd.read_csv(path)

    if df.shape[1] == 0:
        print(f"  -> Empty CSV, skip")
        continue

    if "mfu" not in df.keys():
        last_col_name = df.columns[-1]

        # 计算 MFU
        df["mfu"] = df[last_col_name] / DIVISOR

        df = df.sort_values(by="mfu", ascending=False)

        # 原地保存（如果你想另存，改成 path.with_name(...)）
        df.to_csv(path, index=False)

print("✅ All CSV files processed. MFU column added.")
