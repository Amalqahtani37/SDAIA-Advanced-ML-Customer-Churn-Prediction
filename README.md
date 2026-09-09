# Customer Churn Prediction

## Course
Advanced Machine Learning Methods — SDAIA Academy (SDA-DSC-211)

## Problem Statement
Fiber-optic customers on month-to-month contracts show a disproportionately high churn rate in
this dataset, making this segment the highest-value target for a retention program. This project
builds a model to predict, from a customer's account and subscription attributes, whether they
are likely to churn — with particular attention to the drivers behind this high-risk segment — so
the retention team can proactively target the customers most likely to leave before they leave.

- **Prediction moment:** at a routine account-review checkpoint (e.g. start of a billing cycle), using only
  information already on the account at that point.
- **Decision supported:** whether to enroll a customer in a retention program (discount, contract-upgrade
  call, proactive support outreach), with fiber-optic / month-to-month customers as the priority segment.
  Acted on by the customer-retention team.

## Dataset
- **Name:** IBM Telco Customer Churn
- **Source / link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Provider:** IBM (Sample Data Sets)
- **Licence:** IBM Sample Data Sets — free for educational/demonstration use. **Redistribution permitted:** yes
- **Size:** 7,043 rows × 21 columns
- **Target:** `Churn`  **Positive class (=1):** customer churned (`Yes`)
- **Class balance:** ≈ 26.5% positive

### Why this dataset was selected
Beyond satisfying the course requirements (tabular, binary, moderate imbalance), this dataset was
chosen because its known churn drivers (contract type, tenure, internet service) map clearly onto
actionable retention levers — making the interpretability and threshold-selection phases of this
project directly tied to a real business decision, not just an academic exercise.

## Methodology

### Validation
We use **Stratified 5-Fold cross-validation** because rows are independent (one row per customer, no
repeated entities) and there is no date column defining a real deployment order. This mirrors deployment
because each fold preserves the same churn rate as the full population.

### Leakage Controls
Post-outcome: none found — every column describes current account/subscription state, reviewed against the
stated prediction moment. ID: `customerID` identified and removed. Temporal: not applicable — no time
dimension. Group: not applicable — one row per customer. Preprocessing: all learned transforms (imputer,
one-hot encoder) sit inside a scikit-learn `Pipeline`, refit inside every CV fold.
Removed columns: `customerID` (random identifier, not predictive, classic leakage trap if encoded).

### Model
LightGBM with early stopping, learning rate 0.03, seed 42.
Data split: train 60% / calibration 20% / hold-out 20%.

### Metrics
ROC-AUC, PR-AUC (headline, since churn is the rare class), precision, recall, confusion matrix, Brier score.
All cross-validation results reported as mean ± std across folds.

### Imbalance Handling
Class weighting (`class_weight='balanced'`) compared against the unweighted baseline via before/after
PR-AUC — see Section 7 of `project.ipynb` for the actual numbers from your run.

### Threshold Strategy
Rule used: target recall (≥70%) — a missed churner is far more costly than one unnecessary retention offer,
so recall is prioritised over precision. Selected threshold, precision, recall and flag-rate are printed in
Section 9 of the notebook.

### Hyperparameter Tuning
Optuna, 25 trials, `TPESampler(seed=42)`, optimising mean OOF PR-AUC over a 6-parameter LightGBM search
space. Gain vs. baseline is reported alongside the measured fold spread, not treated as significant on its
own.

### Interpretability
Global: permutation importance and a SHAP beeswarm plot. Local: a SHAP waterfall for a customer near the
decision threshold, translated into plain-language reason codes.

### Calibration
The final selected pipeline is calibrated on a dedicated calibration split (never seen during training)
using sigmoid (Platt) calibration. Brier score is expected to improve while ROC-AUC/PR-AUC stay essentially
unchanged, since calibration reshapes the probability scale, not the ranking.

## Results

*(Phase 10 — ensembling/stacking — is OPTIONAL/BONUS and was intentionally skipped to keep this project
within the 5-hour budget, per the guide's own priority order.)*

| Model | ROC-AUC | PR-AUC | Result @ threshold | Brier | Complexity |
|-------|---------|--------|--------------------|-------|------------|
| Baseline (LightGBM) | 0.7366 ± 0.0167 | 0.4979 ± 0.0293 | — | — | Low |
| + class weighting | 0.7394 | 0.5000 | — | — | Low |
| Optuna-tuned | — | 0.5018 | gain +0.0039 (within fold spread, treated cautiously) | — | Medium |
| **Calibrated (final, hold-out)** | **0.7493** | **0.5034** | precision 0.420, recall 0.751 @ t=0.27 | **0.1874** | Medium |

**Verdict:** class weighting and Optuna tuning both gave gains smaller than the fold-to-fold spread
(±0.0293) — neither is a confirmed improvement on its own. We kept the Optuna-tuned parameters (no extra
complexity cost) but rejected class weighting (it distorted the probability scale without a real PR-AUC
gain). Calibration is the one step that produced a clear, measurable benefit — a lower Brier score
(0.1908 → 0.1667 on the calibration split) with ranking metrics essentially unchanged, exactly as expected.

## Final Model Selection
We select the LightGBM baseline (unweighted, Optuna-tuned, sigmoid-calibrated) as the final model.
Class weighting gave only a marginal PR-AUC gain (+0.0021) — smaller than the fold-to-fold spread
(±0.0293) — while badly distorting the predicted-probability scale, so it was rejected in favour of the
simpler unweighted model. Optuna tuning gave a similarly small gain (+0.0039), also within normal
variability, but the tuned parameters were kept since they came at no extra complexity cost. Calibration
clearly improved probability quality without moving ranking metrics, confirming it worked as intended.

Operating rule: target recall (≥70%)   Threshold: 0.27
Hold-out performance: ROC-AUC 0.7493 | PR-AUC 0.5034 | precision 0.420 | recall 0.751
Known limitations: single-snapshot data with no usage trend over time; the target-recall threshold is a
starting point that should be validated against real retention-team capacity (it currently flags ~45% of
customers, which may exceed a real team's outreach capacity) before deployment.

## Key Findings
1. `tenure` is the single strongest driver (permutation importance 0.148) — longer-tenured customers are
   demonstrably stickier; churn risk drops sharply as tenure increases.
2. `Contract_Month-to-month` is the second strongest driver (0.097) — customers with no switching cost can
   leave any time, matching the well-documented real-world churn pattern for this dataset.
3. Fiber-optic customers combined with month-to-month contracts represent the highest-risk
   segment — a natural target for a focused retention campaign rather than a blanket offer to
   all customers.


## Limitations
This is a single snapshot per customer with no usage trend over time, so the model can say a customer is
*currently* at elevated risk but not predict *when* they will churn. It also cannot establish causation —
contract type is strongly associated with churn here, but switching a customer's contract is not proven to
*cause* lower churn without a controlled experiment (e.g. an A/B test of a retention offer).

## How to Run
```bash
pip install -r requirements.txt
```
Obtain the dataset as described in `data/README.md`, then open `project.ipynb` and run all cells from top
to bottom (Kernel → Restart and Run All before final submission).

## Repository Structure
```
README.md, project.ipynb, requirements.txt, .gitignore,
data/README.md, images/, results/
```

## Requirements
pandas, numpy, scikit-learn, xgboost, lightgbm, shap, optuna, matplotlib

## Author
Amal Alqahtani

## Acknowledgment
This project was completed as part of the Advanced Machine Learning Methods training program at SDAIA
Academy.

Official SDAIA Academy GitHub: [SDAIA Academy](https://github.com/SDAIAAcademy)

## AI Assistance Disclosure
This project was developed with the assistance of an AI tool (Claude, by Anthropic) for code
scaffolding, debugging support, and documentation drafting. All modeling decisions, parameter
choices, and interpretations were reviewed, understood, and are owned by the trainee, who takes
full responsibility for the final analysis and results. The dataset used (IBM Telco Customer
Churn) is public and does not contain confidential, private, or sensitive information.

## Training Program
This project was completed as part of the **Advanced Machine Learning Methods (SDA-DSC-211)**
training program at **SDAIA Academy**, under the supervision of **Fahad Alqahtani**.

The project demonstrates the practical application of advanced machine learning techniques for a
real-world business problem, covering: honest validation design, leakage detection, class
imbalance handling, threshold selection under a justified operating rule, hyperparameter
optimization with Optuna, model interpretability with SHAP, probability calibration, and
evidence-based final model selection.

