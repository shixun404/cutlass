import os
import pandas as pd

MODES = ["wgrad_accum", "wgrad", "dgrad", "fwd"]
OUT_FILE = "summary.csv"

rows = []

for fname in os.listdir("."):

    if not fname.endswith(".csv"):
        continue

    mode = None
    for m in MODES:
        if m in fname:
            mode = m
            break
    if mode is None:
        continue

    try:
        df = pd.read_csv(fname)
    except Exception as e:
        print(f"[skip] failed to read {fname}: {e}")
        continue

    if "mfu" not in df.columns:
        print(f"[skip] no mfu column in {fname}")
        continue

    row = df.loc[df["mfu"].idxmax()].copy()
    row["mode"] = mode
    row = row.reindex(["mode"] + [c for c in row.index if c != "mode"])

    rows.append(row)

if not rows:
    print("No valid csv files found.")
    exit(0)

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_FILE, index=False)

print(f"Saved summary to {OUT_FILE}")
