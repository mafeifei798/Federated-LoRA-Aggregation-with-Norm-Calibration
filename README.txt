Federated LoRA Aggregation with Norm Calibration
================================================
University of Macau summer research | Jun-Oct 2025 (original project period)

This repository collects the project's aggregation code and saved exploratory
results, including subsequent implementation and validation work. It is an
algorithm-focused research collection, not a complete end-to-end training archive.

Research question
-----------------
How can a server combine heterogeneous federated LoRA updates without extra
server-side training examples, labels, features or model-loss feedback?

Implementation and contribution
-------------------------------
1. Historical FedRPCA with variable projection: a standalone extraction of the
   original low-rank/sparse solver and aggregation functions.
2. Paper-aligned PCP aggregation: independent A/B factor blocks, original
   adaptive sparse-energy weighting, and a uniform classifier aggregation.
3. Parameter-only TN: preserve the original RPCA update direction, calibrate
   eligible factor-update norms to twice the uniform client-mean update norm,
   and preserve the original sparse guard and classifier aggregation.

TN does not consume additional server examples or model feedback. It adds
aggregation work, so this is not an equal-FLOPs claim. The saved comparison
uses fixed LoRA rank 4, 48 independent Q/V factor blocks and 50 clients.
The legacy variable-projection solver is retained as historical code; it is
not silently substituted for the PCP solver used in the TN comparison.

Results
-------
The three-dataset results are summarized in results/summary.json. Across three
runs per dataset, the reported mean improvement over original FedRPCA is
0.38-0.75 percentage points. FedAvg, Task Arithmetic and TIES are also included.
These are saved exploratory comparisons, not a new independent blind
confirmation or a claim of statistical significance.

Dataset       TN mean (%)   Original FedRPCA (%)   Difference (pp)
20 Newsgroups 71.509804     70.823529              +0.686275
EuroSAT       96.946502     96.567901              +0.378601
SVHN          92.204326     91.458125              +0.746201

Files
-----
experiments/fedrpca_parameter_only_tn48.py : parameter-only TN implementation
experiments/legacy_fedrpca_vp.py          : original variable-projection functions
src/rspca/federated.py                   : reference aggregation and baselines
src/rspca/federated_cuda.py              : opt-in FP64 CUDA PCP backend
src/rspca/paper_federated.py             : strict registered-factor server API
tests/                                  : tiny source-level checks, no training
results/summary.json                    : saved mean accuracy and sample SD
PROVENANCE.json                         : source identity and extraction hashes

Installation and checks
-----------------------
python -m pip install -e ".[test]"
python -m pytest -q

Production TN/PCP execution requires a suitable PyTorch CUDA environment.
Explicit tiny CPU tests are not evidence of production CUDA qualification.
Raw datasets, model weights, client schedules and server operational records
are not bundled. Running the unit tests does not reproduce the saved benchmark.
The original project materials remain separate and unchanged.

No unrelated research projects or dataset-specific candidate-search queues are included.
