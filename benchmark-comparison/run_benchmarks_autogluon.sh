#!/bin/bash

# Script to run AutoGluon benchmarks with n_runs=5 for CPU, GPU, and GPU-ALL
# This script will run the benchmarks three times - once for CPU, once for GPU, and once for GPU-ALL

echo "Starting AutoGluon benchmarks with n_runs=5..."
echo "=============================================="

# Run CPU benchmarks
echo "Running CPU benchmarks (n_runs=5)..."
python run_autogluon_benchmarks.py --n_runs 5 --cpu

echo ""
echo "CPU benchmarks completed!"
echo "=============================================="

# Run GPU benchmarks
echo "Running GPU benchmarks (n_runs=5)..."
python run_autogluon_benchmarks.py --n_runs 5 --gpu

echo ""
echo "GPU benchmarks completed!"
echo "=============================================="

# Run GPU-ALL benchmarks
echo "Running GPU-ALL benchmarks (n_runs=5)..."
python run_autogluon_benchmarks.py --n_runs 5 --gpu-all

echo ""
echo "GPU-ALL benchmarks completed!"
echo "=============================================="
echo "All benchmarks finished successfully!"
echo ""
echo "Results are saved in the 'autogluon_results' directory"
