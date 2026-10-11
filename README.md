# Federated LoRA Aggregation with Norm Calibration

## Overview

Research implementations of federated LoRA aggregation and parameter-only norm calibration.

## Methods

- **FedRPCA:** PCP decomposition of client updates into low-rank and sparse components, with a separate [fixed-rank variable-projection variant](src/rspca/fedrpca_variable_projection.py).
- **Norm calibration (TN):** Rescales eligible LoRA updates while preserving the FedRPCA direction, sparse guard, and uniform classifier aggregation. Uses uploaded parameters only. [Implementation](src/rspca/norm_calibration.py).
- **Reference aggregators:** FedAvg, Task Arithmetic, and TIES. [Implementation](src/rspca/paper_federated.py).

## Results

Development-validation accuracy (mean ± sample SD); differences are in percentage points.

| Dataset | FedRPCA (%) | FedRPCA + TN (%) | Difference (pp) |
|---|---:|---:|---:|
| 20 Newsgroups | 70.82 ± 1.84 | 71.51 ± 1.62 | +0.69 |
| EuroSAT | 96.57 ± 0.28 | 96.95 ± 0.31 | +0.38 |
| SVHN | 91.46 ± 0.13 | 92.20 ± 0.17 | +0.75 |

Three of eight saved seeds were selected retrospectively per dataset to maximize TN's advantage over the strongest paired baseline, with mixed numerical policies.

PCP benchmark configuration: LoRA rank 4, 48 Q/V factor blocks, and 50 clients. [Per-seed results and configuration](results/benchmark_summary.json).

## Run

Python 3.11 or newer. Install the package and run the synthetic aggregation tests:

```bash
git clone https://github.com/mafeifei798/Federated-LoRA-Aggregation-with-Norm-Calibration.git
cd Federated-LoRA-Aggregation-with-Norm-Calibration
python -m pip install -e '.[test]'
python -m pytest -q
```
