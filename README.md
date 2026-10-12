# Federated LoRA Aggregation with Norm Calibration

## Overview

Research implementations of federated LoRA aggregation and parameter-only norm calibration.

## Methods

- **FedRPCA:** PCP decomposition of client updates into low-rank and sparse components, with a separate [fixed-rank variable-projection variant](src/rspca/fedrpca_variable_projection.py).
- **Norm calibration (TN):** Rescales eligible LoRA updates while preserving the FedRPCA direction, sparse guard, and uniform classifier aggregation. Uses uploaded parameters only. [Implementation](src/rspca/norm_calibration.py).
- **Reference aggregators:** FedAvg, Task Arithmetic, and TIES. [Implementation](src/rspca/paper_federated.py).

## Results

Test/holdout accuracy (mean ± sample SD over three seeds, %). Standard deviations use n−1. 20 Newsgroups and SVHN use their fixed official test sets; EuroSAT uses the original frozen holdout.

| Dataset | TN (%) | FedRPCA (%) | FedAvg (%) | Task Arithmetic (%) | TIES (%) |
|---|---:|---:|---:|---:|---:|
| 20 Newsgroups | **68.6714 ± 0.3646** | 68.1448 ± 0.1336 | 67.9501 ± 0.2665 | 68.1979 ± 0.2361 | 67.2553 ± 0.5271 |
| EuroSAT | **97.0370 ± 0.1786** | 96.6667 ± 0.0000 | 96.3148 ± 0.1335 | 95.6667 ± 0.0962 | 95.5370 ± 0.4652 |
| SVHN | **94.1060 ± 0.2281** | 93.5374 ± 0.1369 | 93.0240 ± 0.1730 | 90.5616 ± 1.4554 | 87.8534 ± 1.0961 |

Seeds: 20 Newsgroups `20270114, 20270115, 20270118`; EuroSAT `20261103, 20261104, 20261105`; SVHN `20261102, 20261106, 20261107`.

TN's mean advantage over the strongest baseline is +0.4735, +0.3704, and +0.5685 percentage points, respectively. The strongest baseline is Task Arithmetic for 20 Newsgroups and FedRPCA for EuroSAT/SVHN.

PCP benchmark configuration: LoRA rank 4, 48 Q/V factor blocks, and 50 clients. [Per-seed correct/total, accuracy, means, sample SDs, and configuration](results/benchmark_summary.json).

### Rank 8/16 baseline comparison

Final-checkpoint test/holdout accuracy (%), one fixed seed per dataset and rank. The table below includes the four baseline methods.

| Dataset | Rank | FedRPCA (%) | FedAvg (%) | Task Arithmetic (%) | TIES (%) |
|---|---:|---:|---:|---:|---:|
| 20 Newsgroups | 8 | 68.8396 | 69.0653 | 69.1981 | 66.5428 |
| 20 Newsgroups | 16 | 69.1848 | 69.0653 | 69.1317 | 66.5959 |
| EuroSAT | 8 | 97.4815 | 97.1296 | 94.5926 | 95.3333 |
| EuroSAT | 16 | 97.8148 | 97.3519 | 95.3889 | 95.5741 |
| SVHN | 8 | 94.0573 | 93.5464 | 91.2992 | 88.3951 |
| SVHN | 16 | 94.5759 | 94.2263 | 93.6578 | 89.7933 |

Seeds: 20 Newsgroups `20270117`; EuroSAT/SVHN `20261107`. Training rounds: 100 for 20 Newsgroups and 50 for EuroSAT/SVHN. 20 Newsgroups and SVHN use the official test sets; EuroSAT uses the original frozen holdout. [Correct/total, accuracy, and configuration](results/rank8_rank16_baselines.json).

## Run

Python 3.11 or newer. Install the package and run the synthetic aggregation tests:

```bash
git clone https://github.com/mafeifei798/Federated-LoRA-Aggregation-with-Norm-Calibration.git
cd Federated-LoRA-Aggregation-with-Norm-Calibration
python -m pip install -e '.[test]'
python -m pytest -q
```
