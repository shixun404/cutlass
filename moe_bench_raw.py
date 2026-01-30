#!/usr/bin/env python3
"""
Profile MoE GroupedGEMM with multiple data types (FP4, FP8, BF16) and NK configurations
"""

import subprocess
import random
import csv
import os
import sys
from typing import List, Dict, Tuple

def generate_uniform_distribution(total_tokens: int, num_experts: int) -> List[int]:
    """Generate uniform token distribution"""
    base = total_tokens // num_experts
    remainder = total_tokens % num_experts
    distribution = [base] * num_experts
    for i in range(remainder):
        distribution[i] += 1
    return distribution

def generate_imbalanced_distribution(total_tokens: int, num_experts: int) -> List[int]:
    """Generate imbalanced token distribution with constraints:
    - Min capacity = avg/4 (no expert gets 0)
    - Max capacity = 4*avg
    - Ensures at least one expert gets max capacity
    """
    avg = total_tokens // num_experts
    min_capacity = max(1, avg // 4)  # Minimum is avg/4, but at least 1
    max_capacity = avg * 4
    
    # Check if we can satisfy minimum requirements
    min_total_needed = min_capacity * num_experts
    if min_total_needed > total_tokens:
        # If we can't satisfy minimum, distribute evenly
        return generate_uniform_distribution(total_tokens, num_experts)
    
    # Start by giving everyone the minimum
    distribution = [min_capacity] * num_experts
    remaining = total_tokens - (min_capacity * num_experts)
    
    # Ensure at least one expert gets max capacity (if possible)
    if remaining > 0:
        # First expert gets as much as possible up to max_capacity
        extra_for_first = min(remaining, max_capacity - min_capacity)
        distribution[0] += extra_for_first
        remaining -= extra_for_first
        
        # Distribute remaining tokens randomly among other experts
        for i in range(1, num_experts):
            if remaining > 0:
                # Each expert can get up to (max_capacity - current) more tokens
                max_extra = min(remaining, max_capacity - distribution[i])
                if max_extra > 0:
                    # Random amount between 0 and max_extra, biased towards lower values
                    # to create more imbalance
                    if random.random() < 0.6:  # 60% chance of getting less tokens
                        extra = random.randint(0, max_extra // 2)
                    else:
                        extra = random.randint(0, max_extra)
                    distribution[i] += extra
                    remaining -= extra
        
        # If there's still remaining, distribute to experts that haven't reached max
        while remaining > 0:
            for i in range(num_experts):
                if remaining <= 0:
                    break
                if distribution[i] < max_capacity:
                    extra = min(1, remaining, max_capacity - distribution[i])
                    distribution[i] += extra
                    remaining -= extra
    
    # Shuffle to randomize which experts get more/less
    random.shuffle(distribution)
    return distribution

def write_problem_sizes(distribution: List[int], n: int, k: int, filename: str):
    """Write problem sizes to file in rcgrouped format.

    For rcgrouped example, each line should be:
      <idx> MxNxK
    where:
      - idx is the expert index [0, NUM_EXPERTS)
      - M is the hidden size (passed in as n argument here)
      - N is tokens_per_expert for that expert (from distribution)
      - K is the K dimension
    """
    m_hidden = n
    with open(filename, 'w') as f:
        for idx, tokens_per_expert in enumerate(distribution):
            f.write(f"{idx} {m_hidden}x{tokens_per_expert}x{k}\n")
    return filename

def run_profiler(problem_file: str, output_csv: str, kernels: str) -> bool:
    """Run cutlass_profiler and capture output"""
    cmd = [
        "./build_moe/tools/profiler/cutlass_profiler",
        f"--kernels={kernels}",
        "--operation=GroupedGemm",
        f"--problem-sizes-file={problem_file}",
        "--dist=uniform,min=0.0,max=1.0",
        "--profiling-iterations=10",
        "--warmup-iterations=3",
        "--alpha=1",
        "--beta=0",
        f"--output={output_csv}"
    ]
    
    print(f"    Running profiler...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Check for actual output file (profiler adds .grouped_gemm.csv)
        actual_output = output_csv.replace('.csv', '.grouped_gemm.csv')
        if os.path.exists(actual_output):
            file_size = os.path.getsize(actual_output)
            print(f"    Completed ({file_size} bytes)")
            if actual_output != output_csv:
                os.rename(actual_output, output_csv)
            return True
        elif os.path.exists(output_csv):
            file_size = os.path.getsize(output_csv)
            print(f"    Completed ({file_size} bytes)")
            return True
        else:
            print(f"    Warning: No output file created")
            return False
            
    except Exception as e:
        print(f"    Error: {e}")
        return False


def run_rcgrouped_example(problem_file: str,
                          m: int,
                          n_max: int,
                          k: int,
                          groups: int,
                          iterations: int = 10,
                          warmup: int = 3,
                          cluster_m: int = 4,
                          cluster_n: int = 2,
                          raster: str = None,
                          max_sm_count: int = None,
                          use_pdl: bool = False) -> Tuple[float, float]:
    """
    Run 92_blackwell_moe_gemm_rcgrouped example and return GFLOPs.

    The example binary reads problem sizes from --benchmark and prints TFLOPS.
    We parse stdout, extract the TFLOPS line, and convert to GFLOPs.
    """
    exe = "./build/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_rcgrouped"

    cmd = [
        exe,
        f"--m={m}",
        f"--n={n_max}",
        f"--k={k}",
        f"--groups={groups}",
        f"--iterations={iterations}",
        f"--warmup={warmup}",
        f"--benchmark={problem_file}",
        f"--cluster_m={cluster_m}",
        f"--cluster_n={cluster_n}",
    ]

    if raster is not None:
        cmd.append(f"--raster={raster}")
    if max_sm_count is not None:
        cmd.append(f"--max_sm_count={max_sm_count}")
    if use_pdl:
        cmd.append("--use_pdl")

    print(f"    Running rcgrouped example:\n      {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            print(f"    rcgrouped example failed with code {result.returncode}")
            if stderr:
                print(f"    stderr:\n{stderr}")
            return None, None

        # Example prints TFLOPS twice:
        #   once for 1SM config, once for 2SM config.
        # Collect all TFLOPS lines in order.
        tflops_values: List[float] = []
        for line in stdout.splitlines():
            if "TFLOPS" in line:
                parts = line.strip().split()
                for token in reversed(parts):
                    try:
                        tflops = float(token)
                        tflops_values.append(tflops)
                        break
                    except ValueError:
                        continue

        if not tflops_values:
            print("    Warning: failed to parse TFLOPS from rcgrouped output")
            print(stdout)
            return None, None

        # Assign 1SM / 2SM results
        tflops_1sm = tflops_values[0]
        tflops_2sm = tflops_values[1] if len(tflops_values) > 1 else None

        gflops_1sm = tflops_1sm * 1000.0
        gflops_2sm = tflops_2sm * 1000.0 if tflops_2sm is not None else None

        if tflops_2sm is not None:
            print(f"    Parsed performance: "
                  f"1SM={gflops_1sm:.0f} GFLOPS ({tflops_1sm:.2f} TFLOPS), "
                  f"2SM={gflops_2sm:.0f} GFLOPS ({tflops_2sm:.2f} TFLOPS)")
        else:
            print(f"    Parsed performance: "
                  f"1SM={gflops_1sm:.0f} GFLOPS ({tflops_1sm:.2f} TFLOPS)")

        return gflops_1sm, gflops_2sm

    except Exception as e:
        print(f"    Error running rcgrouped example: {e}")
        return None, None


def run_blockscaled_rcgrouped_example(problem_file: str,
                                      m: int,
                                      n_max: int,
                                      k: int,
                                      groups: int,
                                      iterations: int = 10,
                                      warmup: int = 3,
                                      cluster_m: int = 2,
                                      cluster_n: int = 1,
                                      raster: str = 'N',
                                      max_sm_count: int = None,
                                      use_pdl: bool = False,
                                      norm_constant: float = 1.0) -> Tuple[float, float]:
    """
    Run 92_blackwell_moe_gemm_blockscaled_rcgrouped example and return (gflops_1sm, gflops_2sm).

    The blockscaled example also prints TFLOPS twice: once for 1SM, once for 2SM.
    """
    exe = "./build/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped"

    cmd = [
        exe,
        f"--m={m}",
        f"--n={n_max}",
        f"--k={k}",
        f"--groups={groups}",
        f"--iterations={iterations}",
        f"--warmup={warmup}",
        f"--benchmark={problem_file}",
        f"--cluster_m={cluster_m}",
        f"--cluster_n={cluster_n}",
        f"--norm_constant={norm_constant}",
        "--no_verif",
    ]
    if raster is not None:
        cmd.append(f"--raster={raster}")
    if max_sm_count is not None:
        cmd.append(f"--max_sm_count={max_sm_count}")
    if use_pdl:
        cmd.append("--use_pdl")

    print(f"    Running blockscaled rcgrouped example:\n      {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            print(f"    blockscaled rcgrouped example failed with code {result.returncode}")
            if stderr:
                print(f"    stderr:\n{stderr}")
            return None, None

        tflops_values: List[float] = []
        for line in stdout.splitlines():
            if "TFLOPS" in line:
                parts = line.strip().split()
                for token in reversed(parts):
                    try:
                        tflops = float(token)
                        tflops_values.append(tflops)
                        break
                    except ValueError:
                        continue

        if not tflops_values:
            print("    Warning: failed to parse TFLOPS from blockscaled rcgrouped output")
            print(stdout)
            return None, None

        tflops_1sm = tflops_values[0]
        tflops_2sm = tflops_values[1] if len(tflops_values) > 1 else None

        gflops_1sm = tflops_1sm * 1000.0
        gflops_2sm = tflops_2sm * 1000.0 if tflops_2sm is not None else None

        if tflops_2sm is not None:
            print(f"    Parsed performance (blockscaled): "
                  f"1SM={gflops_1sm:.0f} GFLOPS ({tflops_1sm:.2f} TFLOPS), "
                  f"2SM={gflops_2sm:.0f} GFLOPS ({tflops_2sm:.2f} TFLOPS)")
        else:
            print(f"    Parsed performance (blockscaled): "
                  f"1SM={gflops_1sm:.0f} GFLOPS ({tflops_1sm:.2f} TFLOPS)")

        return gflops_1sm, gflops_2sm

    except Exception as e:
        print(f"    Error running blockscaled rcgrouped example: {e}")
        return None, None

def get_best_kernel(csv_file: str) -> Tuple[str, float]:
    """Get the best performing kernel from CSV"""
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows:
            return None, None
        
        # Find GFLOPs column
        gflops_col = None
        for col in ['GFLOPs', 'gflops', 'GFLOPS']:
            if col in rows[0]:
                gflops_col = col
                break
        
        if gflops_col:
            # Sort by GFLOPs descending
            rows.sort(key=lambda x: float(x[gflops_col]) if x[gflops_col] and x[gflops_col] != 'nan' else 0, reverse=True)
            best_row = rows[0]
            
            # Get kernel name from Operation column
            kernel_name = best_row.get('Operation', list(best_row.values())[0])
            
            return kernel_name, float(best_row[gflops_col])
        
        return None, None
        
    except Exception as e:
        print(f"    Error reading CSV: {e}")
        return None, None

def main():
    NUM_EXPERTS = 96
    BASE_TOKENS = 32768
    MULTIPLIERS = [1, 2]
    # NUM_EXPERTS = 256
    # BASE_TOKENS = NUM_EXPERTS
    # MULTIPLIERS = [1, 2, 4, 8, 16]
    
    # # Multiple NK configurations
    # NK_CONFIGS = [
    #     (4096, 6144),   # Config 1
    #     (6144, 2048),   # Config 2
    # ]

    # Multiple NK configurations
    NK_CONFIGS = [
        (3072, 5120),   # Config 1
        (4096, 7168),   # Config 2
    ]
    

    # Configs: rcgrouped FP8, blockscaled FP4
    DTYPE_CONFIGS = {
        "rcgrouped_fp8": {
            "short_name": "RCGrouped-FP8",
            "runner": "rcgrouped",
        },
        # "blockscaled_fp4": {
        #     "short_name": "Blockscaled-FP4",
        #     "runner": "blockscaled",
        # },
    }

    # Sweep space for scheduling / parallel parameters
    # You can customize these lists as needed.
    CLUSTER_SHAPES = [
        (1, 1),
        (2, 1),
        (2, 2),
        (4, 2),
    ]
    RASTERS = [
        'M',
        # 'N',
    ]
    # None means "no limit" (use device SM count)
    MAX_SM_COUNTS = [
        None,
    ]
    USE_PDL_OPTIONS = [
        False,
        True,
    ]

    # Distribution types
    distributions = {
        'uniform': generate_uniform_distribution,
        'imbalanced': generate_imbalanced_distribution,
    }

    # Random seed for reproducibility
    random.seed(42)

    print("=" * 80)
    print("MoE rcgrouped / blockscaled GEMM Profiling (92_blackwell_moe_gemm_rcgrouped / blockscaled_rcgrouped)")
    print("=" * 80)
    print(f"Experts: {NUM_EXPERTS}")
    print(f"Token counts: {[BASE_TOKENS * m for m in MULTIPLIERS]}")
    print(f"MK configs (M,K): {NK_CONFIGS}")
    print(f"Distributions: {list(distributions.keys())}")
    print(f"Cluster shapes: {CLUSTER_SHAPES}")
    print(f"Rasters: {RASTERS}")
    print(f"max_sm_count sweep: {MAX_SM_COUNTS}")
    print(f"use_pdl options: {USE_PDL_OPTIONS}")
    print("=" * 80)

    results_dir = f"moe_profiling_example92_rcgrouped_exp={NUM_EXPERTS}_tk={BASE_TOKENS}_results"
    os.makedirs(results_dir, exist_ok=True)

    # Collect all results for summary
    all_results = []

    # Iterate through all configurations
    for dtype_name, dtype_config in DTYPE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {dtype_config['short_name']}")
        print(f"{'='*60}")

        for m_hidden, k in NK_CONFIGS:
            print(f"\n  M(hidden)={m_hidden}, K={k}")
            print(f"  {'-'*50}")

            for multiplier in MULTIPLIERS:
                total_tokens = BASE_TOKENS * multiplier

                for dist_name, dist_func in distributions.items():
                    print(f"\n  M_total(tokens)={total_tokens}, Distribution={dist_name}")

                    try:
                        # Generate distribution of tokens per expert (N_i)
                        distribution = dist_func(total_tokens, NUM_EXPERTS)

                        # Validate
                        assert sum(distribution) == total_tokens
                        assert len(distribution) == NUM_EXPERTS

                        # Stats
                        min_m = min(distribution)
                        max_m = max(distribution)
                        avg_m = sum(distribution) / len(distribution)
                        print(f"    Token dist: min={min_m}, max={max_m}, avg={avg_m:.0f}")

                        # Write problem sizes for rcgrouped:
                        #   line format: "<idx> MxNxK"
                        #   M = m_hidden, N = tokens_per_expert_i, K = k
                        problem_file = os.path.join(
                            results_dir,
                            f"problems_{dtype_name}_m{m_hidden}k{k}_tokens{total_tokens}_{dist_name}.txt"
                        )
                        write_problem_sizes(distribution, m_hidden, k, problem_file)

                        n_max = max(distribution)

                        # Sweep all scheduling / parallel parameters
                        for (cluster_m, cluster_n) in CLUSTER_SHAPES:
                            for raster in RASTERS:
                                for max_sm in MAX_SM_COUNTS:
                                    for use_pdl in USE_PDL_OPTIONS:
                                        print(f"    cluster=({cluster_m},{cluster_n}), raster={raster}, "
                                              f"max_sm={max_sm if max_sm is not None else 'default'}, "
                                              f"use_pdl={use_pdl}")

                                        runner = dtype_config.get("runner", "rcgrouped")
                                        if runner == "rcgrouped":
                                            gflops_1sm, gflops_2sm = run_rcgrouped_example(
                                                problem_file,
                                                m=m_hidden,
                                                n_max=n_max,
                                                k=k,
                                                groups=NUM_EXPERTS,
                                                cluster_m=cluster_m,
                                                cluster_n=cluster_n,
                                                raster=raster,
                                                max_sm_count=max_sm,
                                                use_pdl=use_pdl,
                                            )
                                        elif runner == "blockscaled":
                                            gflops_1sm, gflops_2sm = run_blockscaled_rcgrouped_example(
                                                problem_file,
                                                m=m_hidden,
                                                n_max=n_max,
                                                k=k,
                                                groups=NUM_EXPERTS,
                                                cluster_m=cluster_m,
                                                cluster_n=cluster_n,
                                                raster=raster,
                                                max_sm_count=max_sm,
                                                use_pdl=use_pdl,
                                            )
                                        else:
                                            raise ValueError(f"Unknown runner type: {runner}")
                                        if not gflops_1sm and not gflops_2sm:
                                            print(f"    Failed to get performance")
                                            continue

                                        # Store 1SM result
                                        if gflops_1sm:
                                            all_results.append({
                                                'dtype': dtype_config['short_name'],
                                                'config': '1SM',
                                                'n': m_hidden,           # using column name 'n' for compatibility
                                                'k': k,
                                                'tokens': total_tokens,
                                                'dist': dist_name,
                                                'min_m': min_m,
                                                'max_m': max_m,
                                                'avg_m': avg_m,
                                                'gflops_1sm': gflops_1sm,
                                                'gflops_2sm': gflops_2sm if gflops_2sm else 0.0,
                                                'gflops': max(gflops_1sm, gflops_2sm or 0.0),
                                                'kernel': "92_blackwell_moe_gemm_rcgrouped",
                                                'cluster_m': cluster_m,
                                                'cluster_n': cluster_n,
                                                'raster': raster,
                                                'max_sm': max_sm if max_sm is not None else -1,
                                                'use_pdl': int(use_pdl),
                                            })
                                        # Store 2SM result (if present)
                                        if gflops_2sm:
                                            all_results.append({
                                                'dtype': dtype_config['short_name'],
                                                'config': '2SM',
                                                'n': m_hidden,
                                                'k': k,
                                                'tokens': total_tokens,
                                                'dist': dist_name,
                                                'min_m': min_m,
                                                'max_m': max_m,
                                                'avg_m': avg_m,
                                                'gflops_1sm': gflops_1sm if gflops_1sm else 0.0,
                                                'gflops_2sm': gflops_2sm,
                                                'gflops': gflops_2sm,
                                                'kernel': "92_blackwell_moe_gemm_rcgrouped",
                                                'cluster_m': cluster_m,
                                                'cluster_n': cluster_n,
                                                'raster': raster,
                                                'max_sm': max_sm if max_sm is not None else -1,
                                                'use_pdl': int(use_pdl),
                                            })

                    except Exception as e:
                        print(f"    Error: {e}")
                        continue

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Group by config (only one in this script, but keep the structure)
    for dtype_name in DTYPE_CONFIGS.keys():
        dtype_results = [r for r in all_results if r['dtype'] == DTYPE_CONFIGS[dtype_name]['short_name']]
        if not dtype_results:
            continue

        print(f"\n{DTYPE_CONFIGS[dtype_name]['short_name']}:")
        print("-" * 120)
        print(f"{'M':<6} {'K':<6} {'Tokens':<8} {'Dist':<10} {'Cfg':<5} "
              f"{'Min N':<6} {'Max N':<6} {'cl_m':<5} {'cl_n':<5} "
              f"{'rast':<5} {'max_sm':<7} {'pdl':<4} "
              f"{'GF1SM':<10} {'GF2SM':<10} {'GFmax':<10}")
        print("-" * 120)

        for r in dtype_results:
            print(f"{r['n']:<6} {r['k']:<6} {r['tokens']:<8} {r['dist']:<10} {r.get('config',''): <5}"
                  f"{r['min_m']:<6} {r['max_m']:<6} "
                  f"{r.get('cluster_m', 0):<5} {r.get('cluster_n', 0):<5} "
                  f"{r.get('raster', ''):<5} {r.get('max_sm', -1):<7} {r.get('use_pdl', 0):<4} "
                  f"{r.get('gflops_1sm', 0.0):<10.0f} {r.get('gflops_2sm', 0.0):<10.0f} {r.get('gflops', 0.0):<10.0f}")

    # Save summary to CSV
    summary_file = os.path.join(results_dir, "summary.csv")
    if all_results:
        with open(summary_file, 'w', newline='') as f:
            fieldnames = [
                'dtype', 'config', 'n', 'k', 'tokens', 'dist',
                'min_m', 'max_m', 'avg_m',
                'cluster_m', 'cluster_n', 'raster', 'max_sm', 'use_pdl',
                'gflops_1sm', 'gflops_2sm', 'gflops', 'kernel',
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSummary saved to: {summary_file}")

    print(f"\nAll results saved in: {results_dir}/")

if __name__ == "__main__":
    main()