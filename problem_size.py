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
    min_capacity = max(1, avg // 16)  # Minimum is avg/4, but at least 1
    max_capacity = avg * 16
    
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

def write_problem_sizes(distribution: List[int], M: int, K: int, filename: str):

    with open(filename, 'w') as f:
        for idx, tokens_per_expert in enumerate(distribution):
            f.write(f"{M}x{tokens_per_expert}x{K}\n")
    return filename

if __name__ == "__main__":
    total_tokens = 32768
    NUM_EXPERTS = 256

    NK_CONFIGS = [
        (3072, 5120),   # Config 1
        (4096, 7168),   # Config 2
    ]
    M, K = NK_CONFIGS[0]
    distribution = generate_imbalanced_distribution(total_tokens, NUM_EXPERTS)

    dist_name = "imbalenced"
    # Validate
    assert sum(distribution) == total_tokens
    assert len(distribution) == NUM_EXPERTS

    # Stats
    min_m = min(distribution)
    max_m = max(distribution)
    avg_m = sum(distribution) / len(distribution)
    print(f"    Token dist: min={min_m}, max={max_m}, avg={avg_m:.0f}")
    problem_file = os.path.join(
                            "build",
                            f"problems_m{M}k{K}_tokens{total_tokens}_{dist_name}.txt"
                        )
    write_problem_sizes(distribution, M, K, problem_file)