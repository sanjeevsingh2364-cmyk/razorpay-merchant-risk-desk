# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview
A fraud triage service for a fictional payment processor (all data is synthetic).
Part A cleans a messy payment export and engineers features.
Part B trains a random forest and picks an alert threshold on validation data.
Part C retrieves the matching fraud policy (TF-IDF) and asks a mock language model for a JSON case note, checked by three validation gates.
Part D serves it all through FastAPI with input checks, guardrails, logging and metrics.

## Reproduce
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prepare.py
python train.py
python retrieval.py
python casenote.py
uvicorn app:app --port 8000
```
Then, in a second terminal (service freshly started, `service.log` deleted first):
```
python smoke_test.py
```

## Part A results
Output of `python prepare.py`:

| Check | Value |
|---|---|
| Rows, columns as read | 6,090, 14 |
| Exact duplicate rows | 90 |
| Distinct raw spellings: merchant_category / city | 24 / 24 |
| After de-duplication: rows | 6,000 |
| Missing amount_inr / distance_from_home_km | 76 / 193 |
| amount_inr below 1 or above 1,000,000 | 5 |
| Rows after cleaning | 5,919 (191 missing distance values remain) |

Missing `distance_from_home_km` values are not filled in Part A because the fill statistic must be learned from training rows only, otherwise validation and test information leaks into training.

Leakage audit (cleaned table):

| chargeback_filed | rows | fraud rows |
|---|---|---|
| 1 | 373 | 304 |
| 0 | 5,546 | 62 |

`chargeback_filed` records an outcome that becomes known after the payment, so using it as an input would cause target leakage. `txn_id` and `customer_id` are identifiers, not reliable behavioural features, and could encourage the model to learn accidental patterns instead of generalising to new transactions. Therefore, all three columns must be excluded from model inputs.

Fraud represents only 6.18% of the cleaned dataset, so a model that predicts every transaction as non-fraud can achieve 93.82% accuracy while detecting no fraud cases. Precision, recall and F1-score are more informative because they show how well the model detects fraud and how many legitimate transactions it incorrectly flags.


### Part B — Model Training and Evaluation

The dataset was split into 3,551 training rows, 1,184 validation rows and 1,184 test rows. The training split contained 220 fraud cases, while validation and test each contained 73 fraud cases. The median used to impute `distance_from_home_km` was 12.6, and random oversampling increased the training set to 6,662 rows with 3,331 fraud cases.

At a validation threshold of 0.5, plain logistic regression achieved precision 0.625, recall 0.342 and F1-score 0.442. Class-weighted logistic regression achieved precision 0.252, recall 0.781 and F1-score 0.381, while oversampled logistic regression achieved precision 0.257, recall 0.781 and F1-score 0.386.

Five-fold cross-validation produced a mean F1-score of 0.402 (standard deviation 0.015) for balanced logistic regression and 0.453 (standard deviation 0.052) for the untuned random forest. Grid search evaluated 12 parameter combinations across five folds and selected `max_depth=None` and `min_samples_leaf=5`, with a mean CV F1-score of 0.465 (standard deviation 0.042).

The selected threshold was `t_star = 0.4`, based on the validation recall constraint of at least 0.65. On the test set, the model achieved precision 0.281, recall 0.781 and F1-score 0.413, with TN=965, FP=146, FN=16 and TP=57. It flagged 203 payments, corresponding to 2.56 legitimate payments held per fraud caught.

The leakage experiment produced a validation F1-score of 0.441 for the clean tuned forest and 0.797 when `chargeback_filed` was included. This difference demonstrates why a post-outcome field must be excluded from model inputs.

## Part C results
Retrieval (`python retrieval.py`): 8 of 8 in-scope queries hit. Highest out-of-scope score 0.4251, lowest in-scope score 0.4862, `MIN_SCORE = 0.45`.

Scenario run (`python casenote.py`):

| Scenario | Policy | Attempts | Failures | Status | Safe default | Final action | Override |
|---|---|---|---|---|---|---|---|
| S1 (T900001) | P1 | 1 | [] | ok | no | hold_for_review | no |
| S2 (T900002) | P2 | 2 | [1] | ok | no | hold_for_review | no |
| S3 (T900003) | P3 | 2 | [2] | ok | no | hold_for_review | no |
| S4 (T900004) | P4 | 2 | [3] | ok | no | block_and_notify | no |
| S5 (T900005) | P5 | 3 | [1, 1, 1] | ok | yes | hold_for_review | no |
| S6 (T900006) | P2 | 1 | [] | ok | no | hold_for_review | yes |
| S7 (T900007) | P1 | 2 | [3] | ok | no | hold_for_review | no |
| S8 (T900008) | P4 | 1 | [] | ok | no | block_and_notify | no |
| S9 (T900009) | P3 | 3 | [1, 2] | ok | no | hold_for_review | no |
| S10 (T900010) | P2 | 2 | [3] | ok | no | hold_for_review | no |

Totals: 19 model calls; 3 valid on the first call; 6 recovered by retry; 1 safe default; 1 guardrail override.
A language model can return malformed JSON, omit required fields, use unsupported values or recommend an action that violates policy, and retrying the same prompt does not guarantee a valid response. Therefore, each output must pass parsing, schema validation and policy guardrails, with a safe default used when all attempts fail.

## Part D results
A language model can return malformed JSON, omit required fields, use unsupported values or recommend an action that violates policy, and retrying the same prompt does not guarantee a valid response. Therefore, each output must pass parsing, schema validation and policy guardrails, with a safe default used when all attempts fail.
## Limitations
- The data is synthetic and its fraud rule was hand-built, so real fraud would behave differently.
- The retriever is keyword-based TF-IDF and cannot match meaning.
- The language model is a scripted stand-in, so real model failures are only imitated.
- The alert threshold is tuned on only 73 validation fraud cases, so it is noisy.