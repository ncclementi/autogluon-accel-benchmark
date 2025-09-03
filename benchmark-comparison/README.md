# Benchmark Comparison Directory

Benchmarks AutoGluon vs scikit-learn performance across CPU/GPU configurations (100K-5M samples, LR/RF/KNN models).

## Files

- `benchmark_analysis.py` - Main analysis script that processes CSV results and generates reports/plots
- `unified_autogluon_benchmark.py` - Core AutoGluon benchmarking script for LR/RF/KNN models
- `run_autogluon_benchmarks.py` - AutoGluon benchmark orchestrator (CPU/GPU/GPU-ALL configs)
- `run_benchmarks_autogluon.sh` - Bash script to run complete AutoGluon benchmark suite
- `unified_sklearn_benchmark.py` - Core scikit-learn benchmarking script for LR/RF/KNN models
- `run_sklearn_benchmarks.py` - Scikit-learn benchmark orchestrator (CPU/GPU configs)
- `run_benchmarks_sklearn.sh` - Bash script to run complete scikit-learn benchmark suite


## Directories

- `autogluon_results/` - AutoGluon benchmark CSV results (CPU/cuml_acc/cuml_cudf_acc configs)
- `sklearn_results/` - Scikit-learn benchmark CSV results (CPU/cuml_acc configs)

## Generated Files

- `benchmark_report_final.md` - Comprehensive analysis report with plots and tables
- `autogluon_performance_comparison.png` - AutoGluon performance visualization (CPU vs cuml vs cuml_cudf)
- `sklearn_performance_comparison.png` - Scikit-learn performance visualization (CPU vs cuml_accel)

## Quick Usage

```bash
# Run all benchmarks
./run_benchmarks_autogluon.sh && ./run_benchmarks_sklearn.sh

# Generate analysis report
python benchmark_analysis.py
```
