#!/usr/bin/env python3
"""
Compare performance between moe_gemm and grouped_gemm with identical parameters.

Usage:
    python compare_moe_vs_grouped.py \
        --moe_csv <path_to_moe_csv> \
        --grouped_csv <path_to_grouped_csv> \
        --output_csv <optional_output_csv>
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

def create_key(row, key_columns):
    """Create a unique key from row based on key columns."""
    return tuple(row[col] for col in key_columns)

def compare_performance(moe_csv_path, grouped_csv_path, output_csv=None):
    """
    Compare performance between moe_gemm and grouped_gemm.
    
    Args:
        moe_csv_path: Path to moe_gemm results CSV
        grouped_csv_path: Path to grouped_gemm results CSV
        output_csv: Optional path to save comparison results
    """
    # Read CSV files
    print(f"Reading moe_gemm results from: {moe_csv_path}")
    moe_df = pd.read_csv(moe_csv_path)
    
    print(f"Reading grouped_gemm results from: {grouped_csv_path}")
    grouped_df = pd.read_csv(grouped_csv_path)
    
    print(f"Moe GEMM entries: {len(moe_df)}")
    print(f"Grouped GEMM entries: {len(grouped_df)}")
    
    # Key columns for matching (excluding operation-specific columns)
    key_columns = [
        'cta_m', 'cta_n', 'cta_k',
        'cluster_m', 'cluster_n', 'cluster_k',
        'stages',
        'inst_m', 'inst_n', 'inst_k',
        'warps_m', 'warps_n', 'warps_k'
    ]
    
    # Filter only successful runs
    moe_df = moe_df[moe_df['Status'] == 'success'].copy()
    grouped_df = grouped_df[grouped_df['Status'] == 'success'].copy()
    
    print(f"\nAfter filtering successful runs:")
    print(f"Moe GEMM entries: {len(moe_df)}")
    print(f"Grouped GEMM entries: {len(grouped_df)}")
    
    # Create keys for matching
    moe_df['key'] = moe_df.apply(lambda row: create_key(row, key_columns), axis=1)
    grouped_df['key'] = grouped_df.apply(lambda row: create_key(row, key_columns), axis=1)
    
    # Performance columns to compare
    perf_columns = ['Runtime', 'GB/s', 'GFLOPs']
    
    # Find matching configurations
    matches = []
    moe_keys = set(moe_df['key'])
    grouped_keys = set(grouped_df['key'])
    common_keys = moe_keys & grouped_keys

    
    print(f"\nCommon configurations (matching keys): {len(common_keys)}")
    print(f"Moe-only configurations: {len(moe_keys - grouped_keys)}")
    print(f"Grouped-only configurations: {len(grouped_keys - moe_keys)}")
    
    # For each common key, compare performance
    for key in common_keys:
        moe_rows = moe_df[moe_df['key'] == key]
        grouped_rows = grouped_df[grouped_df['key'] == key]
        
        # Take the best performing entry from each (highest GFLOPs)
        moe_best = moe_rows.loc[moe_rows['GFLOPs'].idxmax()]
        grouped_best = grouped_rows.loc[grouped_rows['GFLOPs'].idxmax()]
        
        # Create comparison entry
        match_entry = {}
        
        # Add key parameters
        for col in key_columns:
            if col in moe_best:
                match_entry[col] = moe_best[col]
        
        # Add performance metrics
        for col in perf_columns:
            match_entry[f'moe_{col}'] = moe_best[col]
            match_entry[f'grouped_{col}'] = grouped_best[col]
        
        # Calculate speedup/ratio
        # Runtime_speedup: moe/grouped (>1 means moe is slower)
        # GFLOPs_ratio: grouped/moe (>1 means grouped is faster)
        if moe_best['Runtime'] > 0:
            match_entry['Runtime_speedup'] =  grouped_best['Runtime'] / moe_best['Runtime']
        else:
            match_entry['Runtime_speedup'] = float('inf')
            
        if grouped_best['GB/s'] > 0:
            match_entry['GBs_ratio'] = moe_best['GB/s'] / grouped_best['GB/s']
        else:
            match_entry['GBs_ratio'] = float('inf')
            
        if grouped_best['GFLOPs'] > 0:
            match_entry['GFLOPs_ratio'] = moe_best['GFLOPs'] / grouped_best['GFLOPs'] 
        else:
            match_entry['GFLOPs_ratio'] = float('inf')
        
        # Add operation names for reference
        match_entry['moe_Operation'] = moe_best['Operation']
        match_entry['grouped_Operation'] = grouped_best['Operation']
        
        matches.append(match_entry)
    
    # Create comparison DataFrame
    if not matches:
        print("\nNo matching configurations found!")
        return
    
    comparison_df = pd.DataFrame(matches)
    
    # Sort by GFLOPs ratio (grouped/moe)
    comparison_df = comparison_df.sort_values('GFLOPs_ratio', ascending=False)
    
    # Display results
    print("\n" + "="*150)
    print("PERFORMANCE COMPARISON (Grouped GEMM / Moe GEMM)")
    print("="*150)
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"  Average Runtime Speedup (moe/grouped): {comparison_df['Runtime_speedup'].mean():.3f}x")
    print(f"  Average GB/s Ratio (grouped/moe): {comparison_df['GBs_ratio'].mean():.3f}x")
    print(f"  Average GFLOPs Ratio (grouped/moe): {comparison_df['GFLOPs_ratio'].mean():.3f}x")
    
    print(f"\n  Best Runtime Speedup: {comparison_df['Runtime_speedup'].max():.3f}x")
    print(f"  Worst Runtime Speedup: {comparison_df['Runtime_speedup'].min():.3f}x")
    print(f"  Best GFLOPs Ratio: {comparison_df['GFLOPs_ratio'].max():.3f}x")
    print(f"  Worst GFLOPs Ratio: {comparison_df['GFLOPs_ratio'].min():.3f}x")
    
    # Display top comparisons
    print("\n" + "-"*150)
    print("TOP 10 CONFIGURATIONS (by GFLOPs ratio, grouped/moe):")
    print("(Higher ratio means grouped_gemm performs better)")
    print("-"*150)
    
    display_cols = [
        'cluster_m', 'cluster_n', 'cluster_k', 'cta_m', 'cta_n', 'cta_k', 'stages', 'inst_m', 'inst_n', 'inst_k',
        'moe_Runtime', 'grouped_Runtime', 'Runtime_speedup',
        'moe_GFLOPs', 'grouped_GFLOPs', 'GFLOPs_ratio'
    ]
    
    # Filter out infinite values for display
    display_df = comparison_df[display_cols].copy()
    display_df = display_df.replace([float('inf'), -float('inf')], None)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 60)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
    
    print(display_df.head(10).to_string(index=False))
    
    print("\n" + "-"*150)
    print("BOTTOM 10 CONFIGURATIONS (by GFLOPs ratio, grouped/moe):")
    print("(Lower ratio means moe_gemm performs better)")
    print("-"*150)
    print(display_df.tail(10).to_string(index=False))
    
    # Reset pandas display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.float_format')
    
    # Save to CSV if requested
    if output_csv:
        comparison_df.to_csv(output_csv, index=False)
        print(f"\nComparison results saved to: {output_csv}")
    
    return comparison_df

def main():
    parser = argparse.ArgumentParser(
        description='Compare performance between moe_gemm and grouped_gemm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--moe_csv',
        type=str,
        required=True,
        help='Path to moe_gemm results CSV file'
    )
    
    parser.add_argument(
        '--grouped_csv',
        type=str,
        required=True,
        help='Path to grouped_gemm results CSV file'
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Optional path to save comparison results CSV'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.moe_csv).exists():
        print(f"Error: Moe CSV file not found: {args.moe_csv}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.grouped_csv).exists():
        print(f"Error: Grouped CSV file not found: {args.grouped_csv}", file=sys.stderr)
        sys.exit(1)
    
    # Run comparison
    try:
        comparison_df = compare_performance(args.moe_csv, args.grouped_csv, args.output_csv)
        if comparison_df is not None:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
