# Federated LoRA Aggregation with Norm Calibration

Aggregation implementations for federated LoRA, including principal-component
pursuit FedRPCA, a fixed-rank variable-projection solver, and parameter-only
norm calibration.

Norm calibration retains the FedRPCA update direction and sparse guard,
rescales eligible factor updates, and leaves uniform classifier aggregation
unchanged. It uses uploaded parameters without additional server-side training
data or model-loss feedback.

## Repository structure

| Path | Contents |
|---|---|
| `src/rspca/norm_calibration.py` | Parameter-only update norm calibration |
| `src/rspca/fedrpca_variable_projection.py` | Fixed-rank variable-projection FedRPCA |
| `src/rspca/federated.py` | NumPy aggregation methods and PCP solver |
| `src/rspca/federated_cuda.py` | Float64 CUDA PCP implementation |
| `src/rspca/paper_federated.py` | Server aggregation interfaces and reference methods |
| `tests/` | Small synthetic implementation checks |
| `results/benchmark_summary.json` | Saved accuracy summaries, seeds, and configuration |

## Saved results

Development-validation accuracy, reported as mean ± sample standard deviation
over three selected runs per dataset. Differences are in percentage points.

| Dataset | FedRPCA accuracy (%) | Norm calibration accuracy (%) | Difference |
|---|---:|---:|---:|
| 20 Newsgroups | 70.82 ± 1.84 | 71.51 ± 1.62 | +0.69 |
| EuroSAT | 96.57 ± 0.28 | 96.95 ± 0.31 | +0.38 |
| SVHN | 91.46 ± 0.13 | 92.20 ± 0.17 | +0.75 |

The result file also retains FedAvg, Task Arithmetic, TIES, and seed identifiers.
FedRPCA has the highest baseline mean for each dataset within these selected runs.

For each dataset, three runs were selected retrospectively from eight saved
seeds to maximize the norm-calibration advantage over the strongest paired
baseline. The selected runs span different numerical-policy cohorts rather
than a prespecified matched-policy evaluation. These exploratory results do
not establish statistical significance or independent confirmation.

The saved configuration uses LoRA rank 4, 48 Q/V factor blocks, and 50 clients.
Norm calibration adds aggregation computation; these are not equal-FLOPs
comparisons. The saved results use the PCP solver, not the variable-projection
solver. The latter takes a fixed decomposition rank and does not implement
adaptive LoRA-rank selection.

## Installation and tests

Python 3.11 or newer:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

This repository provides aggregation components, not an end-to-end training
pipeline. The tests check small synthetic inputs; they do not reproduce the
saved benchmarks or validate production CUDA execution. Raw datasets,
checkpoints, client schedules, and full execution records are retained in
the original research archive.

## Source and license

Original source collection:
[Robust and Sparse PCA](https://github.com/mafeifei798/Robust-and-Sparse-PCA).

See [LICENSE](LICENSE).
