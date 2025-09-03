#!/bin/bash

# Script to run sklearn benchmarks with n_runs=5 for both CPU and GPU
# This script will run the benchmarks twice - once for CPU and once for GPU

echo "Starting sklearn benchmarks with n_runs=5..."
echo "=============================================="

# Run CPU benchmarks
echo "Running CPU benchmarks (n_runs=5)..."
python run_sklearn_benchmarks.py --n_runs 5 --cpu

echo ""
echo "CPU benchmarks completed!"
echo "=============================================="

# Run GPU benchmarks
echo "Running GPU benchmarks (n_runs=5)..."
python run_sklearn_benchmarks.py --n_runs 5 --gpu

echo ""
echo "GPU benchmarks completed!"
echo "=============================================="
echo "All benchmarks finished successfully!"
echo ""
echo "Results are saved in the 'sklearn_results' directory"