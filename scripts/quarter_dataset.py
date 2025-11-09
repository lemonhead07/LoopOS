#!/usr/bin/env python3
"""
Quarter Dataset Script
Reduces a text dataset to 25% of its original size by randomly sampling lines.
Useful for faster training experiments.
"""

import sys
import random
import argparse
from pathlib import Path


def quarter_dataset(input_file, output_file=None, seed=42):
    """
    Quarter a dataset by keeping only 25% of lines randomly sampled.
    
    Args:
        input_file: Path to input text file
        output_file: Path to output file (default: input_file.quarter.txt)
        seed: Random seed for reproducibility
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    # Default output filename
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}.quarter{input_path.suffix}"
    else:
        output_file = Path(output_file)
    
    print(f"Quartering dataset: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all lines
    print("Reading input file...")
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines in input: {total_lines:,}")
    
    # Calculate target size (25%)
    target_size = total_lines // 4
    print(f"Target lines (25%): {target_size:,}")
    
    # Randomly sample 25% of lines
    print("Sampling lines...")
    sampled_lines = random.sample(lines, target_size)
    
    # Write to output file
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)
    
    # Calculate file sizes
    input_size_mb = input_path.stat().st_size / (1024 * 1024)
    output_size_mb = output_file.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ Complete!")
    print(f"  Input:  {total_lines:,} lines ({input_size_mb:.2f} MB)")
    print(f"  Output: {target_size:,} lines ({output_size_mb:.2f} MB)")
    print(f"  Reduction: {100 - (output_size_mb/input_size_mb*100):.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Quarter a dataset by randomly sampling 25% of lines'
    )
    parser.add_argument(
        'input',
        help='Input text file path'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: <input>.quarter.txt)',
        default=None
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        help='Random seed for reproducibility (default: 42)',
        default=42
    )
    
    args = parser.parse_args()
    
    quarter_dataset(args.input, args.output, args.seed)


if __name__ == '__main__':
    main()
