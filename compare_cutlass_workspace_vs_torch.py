#!/usr/bin/env python3
"""
Parse CUTLASS profiler workspace dumps (the *.mat text files produced by --save-workspace=always)
and compare basic statistics against (torch.rand() - 0.5) / 10.

Notes:
- These files are NOT MATLAB .mat. They are plain text produced by CUTLASS TensorViewWrite:
  values separated by commas, with newlines between rows.
- The RNG sequence will NOT match PyTorch, so elementwise equality is not expected.
  This script compares distribution statistics (min/max/mean/std and histogram).
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Tuple


NUM_PATTERN = re.compile(
    r"(?i)(?:"
    r"nan|"
    r"[+-]?inf|"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r")"
)


def iter_floats_from_text_mat(path: str) -> Iterator[float]:
    """Yields floats parsed from a CUTLASS profiler *.mat text dump."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for m in NUM_PATTERN.finditer(line):
                yield float(m.group(0))


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: float = math.inf
    max: float = -math.inf
    n_nan: int = 0

    def update(self, x: float) -> None:
        if math.isnan(x):
            self.n_nan += 1
            return
        self.count += 1
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else float("nan")

    @property
    def std(self) -> float:
        v = self.var
        return math.sqrt(v) if not math.isnan(v) else v


def histogram_bins(lo: float, hi: float, bins: int) -> Tuple[float, float, int]:
    if bins <= 0:
        raise ValueError("bins must be > 0")
    if hi <= lo:
        raise ValueError("hi must be > lo")
    width = (hi - lo) / bins
    return lo, width, bins


def hist_index(x: float, lo: float, width: float, bins: int) -> Optional[int]:
    if math.isnan(x):
        return None
    idx = int((x - lo) / width)
    if idx < 0:
        return -1
    if idx >= bins:
        return bins
    return idx


def summarize_file(
    path: str,
    *,
    expect_count: Optional[int],
    range_check: Optional[Tuple[float, float]],
    hist_range: Optional[Tuple[float, float]],
    hist_bins: int,
    max_values: Optional[int],
) -> Dict[str, object]:
    stats = RunningStats()
    below = 0
    above = 0
    seen = 0

    hist = None
    if hist_range is not None:
        lo, hi = hist_range
        hlo, width, bins = histogram_bins(lo, hi, hist_bins)
        hist = [0] * (bins + 2)  # [below, bins..., above]

    for x in iter_floats_from_text_mat(path):
        stats.update(x)
        if not math.isnan(x):
            seen += 1
            if range_check is not None:
                lo, hi = range_check
                if x < lo:
                    below += 1
                elif x > hi:
                    above += 1
            if hist is not None:
                idx = hist_index(x, hlo, width, bins)
                if idx is not None:
                    hist[idx + 1] += 1  # shift: -1->0, 0->1, ..., bins->bins+1
        if max_values is not None and (stats.count + stats.n_nan) >= max_values:
            break

    if expect_count is not None and seen != expect_count:
        raise RuntimeError(f"{path}: parsed {seen} numeric values, expected {expect_count}")

    out: Dict[str, object] = {
        "path": path,
        "count": stats.count,
        "nan": stats.n_nan,
        "min": stats.min,
        "max": stats.max,
        "mean": stats.mean,
        "std": stats.std,
        "below": below,
        "above": above,
        "hist": hist,
        "hist_range": hist_range,
        "hist_bins": hist_bins,
        "truncated": (max_values is not None and (stats.count + stats.n_nan) >= max_values),
    }
    return out


def torch_reference_stats(
    *,
    sample: int,
    dtype: str,
    seed: int,
    lo: float,
    hi: float,
    bins: int,
) -> Dict[str, object]:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for torch reference comparison. Install torch first.") from e

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x = torch.rand((sample,), generator=g, dtype=torch.float32)
    x = (x - 0.5) / 10.0

    if dtype.lower() in ("bf16", "bfloat16"):
        x = x.to(torch.bfloat16).to(torch.float32)
    elif dtype.lower() in ("f16", "fp16", "float16"):
        x = x.to(torch.float16).to(torch.float32)
    elif dtype.lower() in ("f32", "fp32", "float32"):
        pass
    else:
        raise ValueError(f"Unsupported dtype for torch reference: {dtype}")

    # stats
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    x_mean = float(x.mean().item())
    x_std = float(x.std(unbiased=True).item())

    # histogram
    edges = torch.linspace(lo, hi, bins + 1, dtype=torch.float32)
    # bucketize returns [0..bins]
    idx = torch.bucketize(x, edges, right=False)
    # below: idx==0 and x<lo; above: idx==bins and x>=hi (approx)
    hist = torch.zeros((bins + 2,), dtype=torch.int64)
    below = (x < lo).sum().item()
    above = (x > hi).sum().item()
    hist[0] = below
    hist[-1] = above
    in_range = (x >= lo) & (x <= hi)
    idx_in = torch.bucketize(x[in_range], edges, right=True)  # 1..bins
    # clamp to 1..bins
    idx_in = torch.clamp(idx_in, 1, bins)
    for i in idx_in:
        hist[int(i.item())] += 1

    return {
        "sample": sample,
        "dtype": dtype,
        "seed": seed,
        "min": x_min,
        "max": x_max,
        "mean": x_mean,
        "std": x_std,
        "hist": [int(v) for v in hist.tolist()],
        "hist_range": (lo, hi),
        "hist_bins": bins,
    }


def fmt_stats(label: str, s: Dict[str, object]) -> str:
    return (
        f"{label}\n"
        f"  path: {s['path']}\n"
        f"  count: {s['count']}  nan: {s['nan']}  truncated: {s['truncated']}\n"
        f"  min/max: {s['min']:.8g} / {s['max']:.8g}\n"
        f"  mean/std: {s['mean']:.8g} / {s['std']:.8g}\n"
        f"  out_of_range: below={s['below']} above={s['above']}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--dtype", type=str, default="bf16", help="bf16/f16/f32 for torch reference quantization")
    ap.add_argument("--a", type=str, required=True, help="path to *_A.mat")
    ap.add_argument("--b", type=str, required=True, help="path to *_B.mat")
    ap.add_argument("--c", type=str, required=True, help="path to *_C.mat")
    ap.add_argument("--d", type=str, required=True, help="path to *_D.mat")
    ap.add_argument("--range-lo", type=float, default=-0.05)
    ap.add_argument("--range-hi", type=float, default=0.05)
    ap.add_argument("--hist-bins", type=int, default=101)
    ap.add_argument("--torch-sample", type=int, default=2_000_000)
    ap.add_argument("--torch-seed", type=int, default=2026)
    ap.add_argument(
        "--max-values",
        type=int,
        default=None,
        help="If set, only parse up to this many numeric values per file (for very large dumps).",
    )
    args = ap.parse_args()

    m, n, k = args.m, args.n, args.k

    # RCR shapes for Gemm:
    a_shape = (m, k)
    b_shape = (k, n)
    c_shape = (m, n)
    d_shape = (m, n)

    lo, hi = args.range_lo, args.range_hi
    hist_range = (lo, hi)
    range_check = (lo, hi)

    files = [
        ("A", args.a, a_shape),
        ("B", args.b, b_shape),
        ("C", args.c, c_shape),
        ("D", args.d, d_shape),
    ]

    print("### CUTLASS workspace (*.mat) format")
    print("This is plain text emitted by CUTLASS TensorViewWrite: numbers separated by commas/newlines.")
    print()

    for name, path, shape in files:
        expect = shape[0] * shape[1]
        s = summarize_file(
            path,
            expect_count=expect if args.max_values is None else None,
            range_check=range_check,
            hist_range=hist_range,
            hist_bins=args.hist_bins,
            max_values=args.max_values,
        )
        print(fmt_stats(f"[{name}]", s))

    print("### PyTorch reference (distribution only)")
    ref = torch_reference_stats(
        sample=args.torch_sample,
        dtype=args.dtype,
        seed=args.torch_seed,
        lo=lo,
        hi=hi,
        bins=args.hist_bins,
    )
    print(
        f"[torch] sample={ref['sample']} dtype={ref['dtype']} seed={ref['seed']}\n"
        f"  min/max: {ref['min']:.8g} / {ref['max']:.8g}\n"
        f"  mean/std: {ref['mean']:.8g} / {ref['std']:.8g}\n"
    )
    print("Note: elementwise equality with CUTLASS is not expected (different RNG implementation/sequence).")


if __name__ == "__main__":
    main()


