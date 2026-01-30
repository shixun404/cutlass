#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import os

DIVISOR = 2500000.0

for fname in os.listdir("."):
    if "fwd" not in fname:
        continue
    if not fname.endswith(".csv"):
        continue
    csv_path = fname
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
