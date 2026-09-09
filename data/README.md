# Data

**Dataset:** IBM Telco Customer Churn
**Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
**Mirror:** https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
**Provider:** IBM (distributed as part of IBM's Sample Data Sets for education/demonstration)
**Licence:** IBM Sample Data Sets — freely usable for training/educational projects
**Redistribution permitted:** yes (small file, well under 25 MB), but it is **not** committed to this
repository so that this repo always points at the current authoritative copy.

## How to obtain it

1. Download `Telco-Customer-Churn.csv` from either link above (Kaggle requires a free account).
2. Place it in this folder as: `data/Telco-Customer-Churn.csv`
3. Run `project.ipynb` from the repository root — the notebook automatically detects and loads it.

If the file is not present, `project.ipynb` falls back to a clearly-labelled **synthetic replica** (same
columns, same approximate churn rate and feature relationships) purely so the notebook still runs for
review. **Do not submit results generated on the synthetic fallback** — always run the notebook with the
real file before your final submission.

## Summary

Rows: 7,043  Columns: 21
Target column: `Churn`  Positive class: `Yes` (customer churned)  Base rate: ≈ 26.5%
