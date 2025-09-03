#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run sklearn benchmarks for multiple sample sizes')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs per sample size (default: 5)')
    parser.add_argument('--cpu', action='store_true', help='Run with CPU (default sklearn)')
    parser.add_argument('--gpu', action='store_true', help='Run with GPU (cuML accelerated)')
    args = parser.parse_args()
    
    # Determine mode
    if args.gpu:
        python_cmd = [sys.executable, "-m", "cuml.accel"]
        csv_suffix = "cuml_acc"
    else:
        python_cmd = [sys.executable]
        csv_suffix = "cpu"

    # Create results directory
    os.makedirs("sklearn_results", exist_ok=True)
    
    sample_sizes = [100_000, 1_000_000, 5_000_000]
    
    for n_samples in sample_sizes:
        print(f"Running benchmark for {n_samples:,} samples...")
        
        # Run the benchmark
        cmd = python_cmd + [
            "unified_sklearn_benchmark.py",
            "--suite",
            "--n_samples", str(n_samples),
            "--n_runs", str(args.n_runs),
            "--to_csv", csv_suffix
        ]

        subprocess.run(cmd)

    # move all csv files that start with sklearn_stats_ to sklearn_results
    for file in os.listdir("."):
        if file.startswith("sklearn_stats_"):
            subprocess.run(["mv", file, f"sklearn_results/{file}"])
    
if __name__ == "__main__":
    main()
