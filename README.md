# autogluon-accel-benchmark
Some initial test to attempt to benchmark cuml accel usage with autogluon

### Create env with conda

```bash
conda env create -f autogluon-rapids-nightly.yaml
conda activate rapids-nightly-autogluon
```

### `run_benchmark.py`

It runs a synthetic example for sample sizes of 1K, 10K, 100K, 1M, 10M for the
scikit-learn models that cuml.accel supports at the moment. These are the 
default settings that cuml runs when passing no args to `TablePredictor`. It runs first without cuml.accel then with it activated. See details in `autogluon_sklearn_predictor.py`. 

### `generate_md_report.py`

The script above run and will create some files with results, to get a better understanding of the results run `generate_md_report.py`

### First attempt report

See https://github.com/ncclementi/autogluon-accel-benchmark/issues/1


