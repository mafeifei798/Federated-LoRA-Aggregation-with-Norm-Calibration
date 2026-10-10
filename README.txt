Federated LoRA Aggregation with Norm Calibration
================================================
University of Macau summer research | Jun-Oct 2025

Variable-projection FedRPCA implementations and parameter-only norm calibration
(TN), with saved exploratory comparisons. TN preserves the RPCA update direction
and sparse guard, calibrates eligible factor-update norms, and uses uniform
classifier aggregation without additional server-side training examples,
labels, features, or model-loss feedback.

Code and results
----------------
experiments/fedrpca_parameter_only_tn48.py : parameter-only TN
experiments/legacy_fedrpca_vp.py          : historical variable-projection functions
src/rspca/                              : aggregation methods and reference APIs
tests/                                  : focused source-level checks
results/summary.json                    : all baseline means, sample SDs and settings

Saved development-validation accuracy, three selected runs per dataset:

Dataset       TN mean (%)   FedRPCA mean (%)   Difference (pp)
20 Newsgroups 71.509804     70.823529          +0.686275
EuroSAT       96.946502     96.567901          +0.378601
SVHN          92.204326     91.458125          +0.746201

The full JSON retains FedAvg, Task Arithmetic and TIES, sample standard
deviations, and seed identifiers. FedRPCA has the highest baseline mean on
each dataset within these selected runs.

For each dataset, three runs were selected retrospectively from eight saved
seeds to maximize TN's mean advantage over the strongest paired baseline.
The saved runs span different numerical-policy cohorts and are not a
prespecified matched-policy confirmation. These exploratory comparisons do
not establish statistical significance or independent blind confirmation.

Configuration: LoRA rank 4, 48 independent Q/V factor blocks and 50 clients.
TN adds aggregation computation, so this is not an equal-FLOPs comparison.
The historical VP solver is not substituted for the PCP solver in these results.
Its rank is a fixed input; the adaptive LoRA-rank routine is not included here.

Install and check
-----------------
Python 3.11 or newer:

python -m pip install -e '.[test]'
python -m pytest -q

This is an aggregation-core collection, not an end-to-end training pipeline.
The tests use tiny source-level inputs; they do not reproduce the saved
benchmarks or qualify production CUDA execution. Raw datasets, checkpoints,
client schedules and server operational records remain in the original
research archive rather than this compact code-and-results collection.

Original source collection: https://github.com/mafeifei798/Robust-and-Sparse-PCA
License: see LICENSE.
