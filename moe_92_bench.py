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
    """Write problem sizes to file"""
    with open(filename, 'w') as f:
        for m in distribution:
            f.write(f"{n}x{m}x{k}\n")
    return filename

def run_profiler(problem_file: str, output_csv: str, kernels: str) -> bool:
    """Run cutlass_profiler and capture output"""
    cmd = [
        "./build_moe/tools/profiler/cutlass_profiler",
        f"--kernels={kernels}",
        "--operation=GroupedGemm",
        f"--problem-sizes-file={problem_file}",
        "--dist=uniform,min=0.0,max=1.0",
        "--profiling-iterations=100",
        "--warmup-iterations=1",
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
    # Configuration
    # NUM_EXPERTS = 16
    # BASE_TOKENS = 30720
    # # MULTIPLIERS = [1]
    # MULTIPLIERS = [1, 2, 4, 8, 16]
    
    # # Multiple NK configurations
    # NK_CONFIGS = [
    #     (4096, 6144),   # Config 1
    #     (6144, 2048),   # Config 2
    # ]

    # M13
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
    
    # Data type configurations
    DTYPE_CONFIGS = {
        # "fp4": {
        #     "kernels": "cutlass3x_sm100_bstensorop_moe_gemm_ue8m0xe2m1_ue8m0xe2m1_f32_f16*",
        #     "short_name": "BS-FP4(E2M1)"
        # },
        # "bs_fp8_e4m3": {
        #     "kernels": "cutlass3x_sm100_bstensorop_moe_gemm_ue8m0xe4m3_ue8m0xe4m3_f32_*",
        #     "short_name": "BS-FP8(E4M3)"
        # },
        # "bs_fp8_e5m2": {
        #     "kernels": "cutlass3x_sm100_bstensorop_moe_gemm_ue8m0xe5m2_ue8m0xe5m2_f32_*",
        #     "short_name": "BS-FP8(E5M2)"
        # },
        # "bs_fp8_mix_e4m3_e5m2": {
        #     "kernels": "cutlass3x_sm100_bstensorop_moe_gemm_ue8m0xe4m3_ue8m0xe5m2_f32_*",
        #     "short_name": "BS-FP8(E4M3xE5M2)"
        # },
        # "bs_fp8_mix_e5m2_e4m3": {
        #     "kernels": "cutlass3x_sm100_bstensorop_moe_gemm_ue8m0xe5m2_ue8m0xe4m3_f32_*",
        #     "short_name": "BS-FP8(E5M2xE4M3)"
        # },
        # 'fp8': {
        #     'kernels': 'cutlass3x_sm100_tensorop_moe_gemm_e4m3_e4m3_f32_bf16_e4m3*',
        #     'short_name': 'FP8-E4M3'
        # },
        'bf16': {
            'kernels': 'cutlass3x_sm100_tensorop_moe_gemm_bf16_bf16*f32_bf16*',
            'short_name': 'BF16'
        }
    }
    
    # Distribution types
    distributions = {
        'uniform': generate_uniform_distribution,
        'imbalanced': generate_imbalanced_distribution,
    }
    
    # Random seed for reproducibility
    random.seed(42)
    
    print("=" * 80)
    print("MoE GroupedGEMM Multi-DataType Profiling")
    print("=" * 80)
    print(f"Experts: {NUM_EXPERTS}")
    print(f"Token counts: {[BASE_TOKENS * m for m in MULTIPLIERS]}")
    print(f"NK configs: {NK_CONFIGS}")
    print(f"Data types: {list(DTYPE_CONFIGS.keys())}")
    print(f"Distributions: {list(distributions.keys())}")
    print("=" * 80)
    
    results_dir = f"moe_profiling_example92_exp={NUM_EXPERTS}_tk={BASE_TOKENS}_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Collect all results for summary
    all_results = []
    
    # Iterate through all configurations
    for dtype_name, dtype_config in DTYPE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Data Type: {dtype_config['short_name']}")
        print(f"{'='*60}")
        
        for n, k in NK_CONFIGS:
            print(f"\n  N={n}, K={k}")
            print(f"  {'-'*50}")
            
            for multiplier in MULTIPLIERS:
                total_tokens = BASE_TOKENS * multiplier
                
                for dist_name, dist_func in distributions.items():
                    print(f"\n  M_total={total_tokens}, Distribution={dist_name}")
                    
                    try:
                        # Generate distribution
                        distribution = dist_func(total_tokens, NUM_EXPERTS)
                        
                        # Validate
                        assert sum(distribution) == total_tokens
                        assert len(distribution) == NUM_EXPERTS
                        
                        # Stats
                        min_m = min(distribution)
                        max_m = max(distribution)
                        avg_m = sum(distribution) / len(distribution)
                        print(f"    Token dist: min={min_m}, max={max_m}, avg={avg_m:.0f}")
                        
                        # Write problem sizes
                        problem_file = os.path.join(
                            results_dir, 
                            f"problems_{dtype_name}_n{n}k{k}_m{total_tokens}_{dist_name}.txt"
                        )
                        write_problem_sizes(distribution, n, k, problem_file)
                        
                        # Run profiler
                        output_csv = os.path.join(
                            results_dir,
                            f"results_{dtype_name}_n{n}k{k}_m{total_tokens}_{dist_name}.csv"
                        )
                        
                        if run_profiler(problem_file, output_csv, dtype_config['kernels']):
                        # if True:
                            # Get best kernel
                            kernel_name, gflops = get_best_kernel(output_csv)
                            if kernel_name and gflops:
                                print(f"    Best: {gflops:.0f} GFLOPS")
                                
                                # Store result
                                all_results.append({
                                    'dtype': dtype_config['short_name'],
                                    'n': n,
                                    'k': k,
                                    'tokens': total_tokens,
                                    'dist': dist_name,
                                    'min_m': min_m,
                                    'max_m': max_m,
                                    'avg_m': avg_m,
                                    'gflops': gflops,
                                    'kernel': kernel_name
                                })
                            else:
                                print(f"    Failed to get performance")
                        else:
                            print(f"    Profiler failed")
                            
                    except Exception as e:
                        print(f"    Error: {e}")
                        continue
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    # Group by data type
    for dtype_name in DTYPE_CONFIGS.keys():
        dtype_results = [r for r in all_results if r['dtype'] == DTYPE_CONFIGS[dtype_name]['short_name']]
        if not dtype_results:
            continue
            
        print(f"\n{DTYPE_CONFIGS[dtype_name]['short_name']}:")
        print("-" * 70)
        print(f"{'N':<6} {'K':<6} {'Tokens':<8} {'Dist':<10} {'Min M':<6} {'Max M':<6} {'GFLOPS':<10}")
        print("-" * 70)
        
        for r in dtype_results:
            print(f"{r['n']:<6} {r['k']:<6} {r['tokens']:<8} {r['dist']:<10} "
                  f"{r['min_m']:<6} {r['max_m']:<6} {r['gflops']:<10.0f}")
    
    # Save summary to CSV
    summary_file = os.path.join(results_dir, "summary_mixed.csv")
    if all_results:
        with open(summary_file, 'w', newline='') as f:
            fieldnames = ['dtype', 'n', 'k', 'tokens', 'dist', 'min_m', 'max_m', 'avg_m', 'gflops', 'kernel']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSummary saved to: {summary_file}")
    
    print(f"\nAll results saved in: {results_dir}/")

if __name__ == "__main__":
    main()