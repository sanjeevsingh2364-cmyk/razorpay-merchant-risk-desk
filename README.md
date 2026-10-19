# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview

Razorpay Merchant Risk Desk is a fraud triage service for a fictional payment processor using synthetic transaction data. Part A cleans the raw payment export and engineers features. Part B trains and evaluates fraud detection models and selects an alert threshold using validation data. Part C retrieves relevant fraud policies using TF-IDF and generates validated JSON case notes through a mock language model. Part D exposes the workflow through FastAPI with input validation, policy guardrails, logging, metrics, and smoke tests.

## Reproduce

Requires **Python 3.11 or newer**. The numbers in this README were verified on Python 3.12.3 (Linux) with scikit-learn 1.9.1 and pandas 3.0.6, installed from `requirements.txt`. `train.py` and `test_app.py` were also run successfully on Windows with Python 3.14.7, scikit-learn 1.9.1 and pandas 3.0.6.

Run the following commands from the project directory after cloning the repository.

Create and activate a virtual environment, then install the dependencies.

On Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the pipeline scripts in this order (the same commands on every platform):

```bash
python generate_data.py
python prepare.py
python train.py
python retrieval.py
python casenote.py
```

Start the API service. It trains the final model at start-up, which took about 35 seconds in the verification run; the smoke test waits up to 120 seconds for the service.

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Keep the server running. Open a second terminal in the same project directory, activate the environment (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux/macOS) and run:

```bash
python smoke_test.py
```

**The smoke test must run once, against a freshly started server.** `mock_llm.py` remembers how many times it has been called for each `txn_id`, so the scripted defects for `T900002` and `T900005` only occur on the first run. In a verification run, a second `python smoke_test.py` against the same server gave 9 of 11 passes, with Q7 and Q8 failing. `app.py` also appends to `service.log`. Before the final run, stop any running server, delete `service.log`, start a new server and run `python smoke_test.py` one time, so that `service.log` holds exactly the 10 POST lines of that run.

Optional unit tests (they need the extra packages in `requirements-dev.txt`, which are not part of the service itself):

```bash
pip install -r requirements-dev.txt
python -m pytest -q test_app.py
```

The API documentation is available at `http://127.0.0.1:8000/docs` while the server is running.

The data-generation step recreates the synthetic input data. Run the preparation, training, retrieval, and case-note scripts in the order shown above before starting the API.

## Part A results — Data Preparation and Leakage Audit

The raw dataset contained 6,090 rows and 14 columns. After removing 90 exact duplicate rows, 6,000 rows remained.

| Check | Result |
|---|---:|
| Raw rows | 6,090 |
| Raw columns | 14 |
| Exact duplicate rows | 90 |
| Distinct raw spellings of `merchant_category` | 24 |
| Distinct raw spellings of `city` | 24 |
| Rows after de-duplication | 6,000 |
| Missing `amount_inr` values | 76 |
| Missing `distance_from_home_km` values | 193 |
| Amounts below 1 or above 1,000,000 INR | 5 |
| Final cleaned rows | 5,919 |
| Missing distance values remaining | 191 |

Missing `distance_from_home_km` values were not filled in during Part A because the imputation statistic must be learned from the training split only. Learning it from the full dataset could leak validation or test information into training.

### Leakage audit

| `chargeback_filed` | Rows | Fraud rows |
|---:|---:|---:|
| 1 | 373 | 304 |
| 0 | 5,546 | 62 |

`chargeback_filed` is filed after a payment has settled and is 1 for 304 of the 366 fraud rows, so it would hand the model the answer using information that does not exist at decision time (target leakage). `txn_id` and `customer_id` are identifiers: `txn_id` is unique per row and `customer_id` is drawn independently of the fraud outcome, so a model could only memorise rows or customers with them. `is_fraud` is the target and is excluded for the same reason.

| Never-fraud baseline | Result |
|---|---:|
| Fraud rows in the cleaned table | 366 of 5,919 |
| Fraud rate | 6.18% |
| Accuracy of predicting "not fraud" for every payment | 93.82% |
| Recall of that rule | 0 |

Accuracy is the wrong headline metric because the rare class barely moves it, so a rule that catches no fraud at all still scores 93.82%. Recall and precision measure what the risk desk cares about, namely how many frauds are caught and how many held payments are real fraud.

Final model-input columns (before one-hot encoding): `amount_inr`, `hour_of_day`, `txn_count_last_24h`, `avg_amount_30d_inr`, `distance_from_home_km`, `is_new_device`, `account_age_days`, `amount_vs_avg_ratio`, `is_night`, `high_value`, plus `merchant_category`, `city` and `payment_mode`. Excluded columns: `chargeback_filed`, `txn_id`, `customer_id` and `is_fraud`.

## Part B results — Model Training and Evaluation

The cleaned dataset was split into training, validation, and test sets. Missing distance values were imputed using the training-set median, and random oversampling was applied only to the training data.

| Dataset | Rows | Fraud cases |
|---|---:|---:|
| Training | 3,551 | 220 |
| Validation | 1,184 | 73 |
| Test | 1,184 | 73 |

The training-set median used to impute `distance_from_home_km` was 12.6. Random oversampling increased the training set to 6,662 rows, including 3,331 fraud cases.

The 26 model-input columns, in the order used for training, validation, test and the service:

```
['amount_inr', 'hour_of_day', 'txn_count_last_24h', 'avg_amount_30d_inr', 'distance_from_home_km', 'is_new_device', 'account_age_days', 'amount_vs_avg_ratio', 'is_night', 'high_value', 'merchant_category_Electronics', 'merchant_category_Fashion', 'merchant_category_Food Delivery', 'merchant_category_Grocery', 'merchant_category_Travel', 'merchant_category_Utilities', 'city_Bengaluru', 'city_Chennai', 'city_Delhi', 'city_Kolkata', 'city_Mumbai', 'city_Pune', 'payment_mode_Card', 'payment_mode_NetBanking', 'payment_mode_UPI', 'payment_mode_Wallet']
```

### Imbalance treatment comparison

Validation-set results at a probability threshold of 0.5:

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Plain logistic regression | 0.625 | 0.342 | 0.442 |
| Class-weighted logistic regression | 0.252 | 0.781 | 0.381 |
| Oversampled logistic regression | 0.257 | 0.781 | 0.386 |

Confusion matrices on the validation split at 0.5:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Plain | 1096 | 15 | 48 | 25 |
| Class-weighted | 942 | 169 | 16 | 57 |
| Oversampled | 946 | 165 | 16 | 57 |

A risk desk that must not miss fraud would prefer class weighting or over-sampling, because both raise validation recall from 0.342 to 0.781 (57 of 73 frauds caught instead of 25). The cost is precision, which falls from 0.625 to about 0.25, so roughly 165 to 169 legitimate payments are held instead of 15. Class weighting is the simpler of the two because it duplicates no rows.

Five-fold cross-validation produced the following results:

| Model | Mean CV F1-score | Standard deviation |
|---|---:|---:|
| Balanced logistic regression | 0.402 | 0.015 |
| Untuned random forest | 0.453 | 0.052 |

Grid search evaluated 12 parameter combinations across five folds. The selected random forest parameters were `max_depth=None` and `min_samples_leaf=5`, with a mean cross-validation F1-score of 0.465 and a standard deviation of 0.042.

### Threshold selection and test results

The selected threshold was `t_star = 0.4`, based on the validation-set recall constraint of at least 0.65.

Validation threshold table for the tuned forest:

| Threshold | Precision | Recall | F1 | Flagged |
|---:|---:|---:|---:|---:|
| 0.1 | 0.111 | 0.973 | 0.200 | 638 |
| 0.2 | 0.158 | 0.945 | 0.271 | 437 |
| 0.3 | 0.209 | 0.863 | 0.336 | 302 |
| 0.4 | 0.272 | 0.712 | 0.394 | 191 |
| 0.5 | 0.352 | 0.589 | 0.441 | 122 |
| 0.6 | 0.507 | 0.507 | 0.507 | 73 |
| 0.7 | 0.528 | 0.260 | 0.349 | 36 |
| 0.8 | 0.778 | 0.192 | 0.308 | 18 |
| 0.9 | 0.800 | 0.055 | 0.103 | 5 |

| Test metric | Result |
|---|---:|
| Precision | 0.281 |
| Recall | 0.781 |
| F1-score | 0.413 |
| True negatives (TN) | 965 |
| False positives (FP) | 146 |
| False negatives (FN) | 16 |
| True positives (TP) | 57 |
| Total flagged payments | 203 |
| Legitimate payments held per fraud caught | 2.56 |

The test results show that the model detected 57 of 73 fraud cases while missing 16. It also flagged 146 legitimate transactions, illustrating the trade-off between fraud detection and unnecessary payment reviews.

### Target leakage experiment

The clean tuned random forest achieved a validation F1-score of 0.441. Including `chargeback_filed` increased the validation F1-score to 0.797. This apparent improvement is misleading because the field contains post-outcome information that would not be available at the time of a genuine payment decision.

### Handling class imbalance

Only 6.18% of the cleaned rows are fraud (366 of 5,919), and the training split holds 220 fraud cases. On validation at a 0.5 threshold, plain logistic regression reached precision 0.625 but recall 0.342, so it missed 48 of the 73 fraud cases. Class weighting and random oversampling both raised recall to 0.781 (57 of 73) at the cost of precision (about 0.25), and their F1-scores were close (0.381 and 0.386).

The final random forest uses `class_weight="balanced"` (`make_forest` in `train.py`), which makes mistakes on fraud rows count more during training without duplicating any rows. Oversampling was compared only for logistic regression, and it was applied to the training split only, so the validation and test splits contain no duplicated fraud rows. After training, the alert threshold is tuned on the validation split: `pick_threshold` takes the highest grid threshold whose recall is at least 0.65, which gave `t_star = 0.4`.

This suits fraud detection because accuracy rewards ignoring the rare class (the never-fraud baseline scores 93.82% with zero recall), and a missed fraud is usually more costly than a legitimate payment held for review. That cost judgement is a business assumption, not something measured in this project. The price of the approach is low precision, which is quantified in the risk committee note below.

### Final model selection

The tuned random forest was selected on 5-fold cross-validated F1 computed on the training split only:

| Model | Mean CV F1-score | Standard deviation |
|---|---:|---:|
| Balanced logistic regression | 0.402 | 0.015 |
| Untuned random forest | 0.453 | 0.052 |
| Tuned random forest (`max_depth=None`, `min_samples_leaf=5`) | 0.465 | 0.042 |

On accuracy (CV F1), the tuned forest scores 0.465 against 0.402 for balanced logistic regression, a gain of about 0.06. On stability, its fold-to-fold standard deviation of 0.042 is larger than the 0.015 of logistic regression but smaller than the gap between the two models; tuning added only 0.012 over the untuned forest (0.453), which is within the fold spread, so tuning is not clear evidence of improvement. On interpretability, the forest offers feature importances and probabilities that can be thresholded, but no single coefficient per feature. Relative to logistic regression it therefore gives up some transparency and some stability, and logistic regression was not threshold-tuned here, so the comparison is not like for like. The test split was used once, at `t_star = 0.4`.

### Feature importance

The tuned forest's top five features by impurity-based importance (the importances of all 26 encoded columns sum to 1) are:

| Feature | Importance |
|---|---:|
| `amount_vs_avg_ratio` | 0.270 |
| `txn_count_last_24h` | 0.178 |
| `amount_inr` | 0.134 |
| `account_age_days` | 0.084 |
| `distance_from_home_km` | 0.071 |

Together these five account for about 74% of the total importance. `amount_vs_avg_ratio` ranks first, which means the model relies most on how unusual a payment is compared with the customer's own 30-day average rather than on the raw amount alone. `txn_count_last_24h` ranks second, so a burst of recent payments is the next strongest signal. Newer accounts (`account_age_days`) and payments further from home (`distance_from_home_km`) also carry weight. `amount_inr` is correlated with the ratio, so the importance is shared between the two. Only the top five importances were printed by `train.py`, so no claim is made here about the other columns.

These importances describe what this model uses, not what causes fraud. The ratio, velocity, account-age and distance terms are all part of the synthetic fraud rule in `generate_data.py`, so the ranking mainly shows that the model recovered that rule, not that the same drivers dominate real fraud.

### Risk committee note

The alert threshold is 0.4, the highest threshold on the 0.1 to 0.9 validation grid whose validation recall is at least 0.65 (recall 0.712 at 0.4, but only 0.589 at 0.5). On the test split it flagged 203 of 1,184 payments. It caught 57 of the 73 frauds and missed 16, and it held 146 legitimate payments, which is 2.56 legitimate payments held per fraud caught (146 / 57). A stricter threshold of 0.6 would cut the validation flagged count from 191 to 73 and lift precision from 0.272 to 0.507, but recall would fall from 0.712 to 0.507, so many more frauds would be missed. The model is a triage aid for a human analyst, not an automatic blocker. All of this rests on synthetic data and only 73 test frauds, so none of it is a production guarantee.

## Part C results — Policy Retrieval and Case Notes

The retrieval component uses TF-IDF similarity to identify the relevant fraud policy for a transaction scenario. The configured minimum retrieval score is `MIN_SCORE = 0.45`.

Retrieval on the 10 queries in `retrieval_queries.json` (top-1 policy and cosine score):

| Query | Expected | Retrieved | Score |
|---|---|---|---:|
| card payment above INR 50000 from a new device | P1 | P1 | 0.5501 |
| how many UPI payments in 24 hours trigger a velocity alert | P2 | P2 | 0.5622 |
| payment at 3 AM far away from the home city | P3 | P3 | 0.4948 |
| customer reports unauthorised payment and chargeback case | P4 | P4 | 0.5689 |
| electronics merchant payment on an account younger than 30 days | P5 | P5 | 0.5641 |
| which payments are approved automatically without manual review | P6 | P6 | 0.6250 |
| temporary hold on repeated UPI payments within a day | P2 | P2 | 0.4862 |
| what happens after a customer reports an unauthorised payment | P4 | P4 | 0.4867 |
| what is the cafeteria menu today | None | P3 | 0.4251 |
| how do I reset my email password | None | P1 | 0.0000 |

In-scope top-1 hits: 8 of 8. The highest out-of-scope score (0.4251) and the lowest in-scope score (0.4862) bracket `MIN_SCORE = 0.45`.

| Retrieval check | Result |
|---|---:|
| In-scope queries matched | 8 of 8 |
| Highest out-of-scope score | 0.4251 |
| Lowest in-scope score | 0.4862 |
| Minimum retrieval score | 0.45 |

### Scenario results

The mock language model was tested against ten case scenarios. Its output was checked for valid JSON, required fields and allowed values, and compliance with policy guardrails.

| Scenario | Policy | Attempts | Failed attempts | Status | Safe default | Final action | Override |
|---|---|---:|---|---|---|---|---|
| S1 (T900001) | P1 | 1 | [] | ok | No | hold_for_review | No |
| S2 (T900002) | P2 | 2 | [1] | ok | No | hold_for_review | No |
| S3 (T900003) | P3 | 2 | [2] | ok | No | hold_for_review | No |
| S4 (T900004) | P4 | 2 | [3] | ok | No | block_and_notify | No |
| S5 (T900005) | P5 | 3 | [1, 1, 1] | ok | Yes | hold_for_review | No |
| S6 (T900006) | P2 | 1 | [] | ok | No | hold_for_review | Yes |
| S7 (T900007) | P1 | 2 | [3] | ok | No | hold_for_review | No |
| S8 (T900008) | P4 | 1 | [] | ok | No | block_and_notify | No |
| S9 (T900009) | P3 | 3 | [1, 2] | ok | No | hold_for_review | No |
| S10 (T900010) | P2 | 2 | [3] | ok | No | hold_for_review | No |

Summary:

- Total mock model calls: 19.
- Valid outputs on the first attempt: 3.
- Scenarios recovered through retry: 6.
- Scenarios using the safe default: 1.
- Guardrail overrides: 1.

A language model may return malformed JSON, omit required fields, use unsupported values, or recommend an action that violates policy. Retrying does not guarantee a valid result. Parsing, schema validation, and policy guardrails are therefore necessary, with a safe default when all attempts fail.

### Why repeated failures happen and why validation is needed

In `casenote.py` the prompt is built once per payment and the same prompt is sent on every attempt. A real hosted model called at a temperature of 0 tends to return almost the same text for the same prompt, so a retry is not an independent draw and can repeat the same defect. The mock model shows this behaviour on purpose: `mock_llm.py` injects scripted defects per `txn_id`, and its output depends only on how many times that transaction has been called. `T900005` (S5) returns broken JSON three times in a row, so all three attempts fail and the service uses the safe default.

Three layers of protection are therefore needed, and the scenario results show what each one does:

- **Validation** has three gates: valid JSON (gate 1), all five keys present as strings (gate 2), and allowed values plus the retrieved policy ID (gate 3). It caught the broken JSON in S2, the missing key in S3, and the bad values in S4, S7 (wrong policy) and S10 (unknown team).
- **Retries** recovered 6 of the 10 scenarios (S2, S3, S4, S7, S9, S10) because their defects were scripted for the first call or calls only.
- **The safe default** applies when retries run out (S5). It is a `hold_for_review` note with `high` priority, so a failed model call can never turn into an approval.
- **The guardrail** covers a different failure. In S6 the model returned a well-formed note recommending `approve` for a payment with fraud probability 0.58. That note passes all three validation gates because `approve` is an allowed value, so only the guardrail (an `approve` at probability 0.5 or higher becomes `hold_for_review`) stopped it. Validation checks the form of the output, and the guardrail checks that its content is safe.

## Part D results — FastAPI Service and Smoke Tests

The FastAPI service exposes health, scoring, case-note, metrics, and API documentation endpoints. The smoke-test suite exercised the expected success, escalation, and edge-case behaviours.

Beyond the five input checks in the brief, `app.py` also rejects a non-finite `avg_amount_30d_inr` or `distance_from_home_km`, and an `avg_amount_30d_inr` so small that `amount_inr / avg_amount_30d_inr` exceeds 1,000,000, with status 400 (these inputs previously reached the model and caused a 500 or an unchecked score). The logged `txn_id` is reduced to printable ASCII without spaces so a client cannot add fake lines to `service.log`.

The smoke test must run once against a freshly started server with a deleted `service.log`, as described in the Reproduce section. The results, metrics and log excerpts below all come from the same final run.

### Smoke-test results

| ID | Category | Expected status | Observed status | Result |
|---|---|---:|---:|---|
| Q1 | Happy path | 200 | 200 | PASS |
| Q2 | Happy path | 200 | 200 | PASS |
| Q3 | Escalation | 200 | 200 | PASS |
| Q4 | Escalation | 200 | 200 | PASS |
| Q5 | Edge case | 400 | 400 | PASS |
| Q6 | Edge case | 422 | 422 | PASS |
| Q7 | Escalation | 200 | 200 | PASS |
| Q8 | Escalation | 200 | 200 | PASS |
| Q9 | Edge case | 503 | 503 | PASS |
| Q10 | Happy path | 200 | 200 | PASS |
| Q11 | Edge case | 200 | 200 | PASS |

| Test category | Passed | Total |
|---|---:|---:|
| Happy path | 3 | 3 |
| Escalation | 4 | 4 |
| Edge cases | 4 | 4 |
| **Overall** | **11** | **11** |

The smoke-test run passed all 11 checks. The expected HTTP statuses included successful requests, input-validation errors, and a simulated case-note service failure.

### Metrics

The recorded `/metrics` output was:

```json
{
  "requests_total": 10,
  "success_total": 7,
  "error_total": 3,
  "success_rate": 0.7,
  "error_rate": 0.3,
  "avg_latency_ms": 13.47,
  "human_review_total": 5,
  "human_review_rate": 0.5
}
```

These figures represent one recorded test run, not a benchmark of production traffic. The metrics endpoint reports request totals, success and error rates, average latency, and human-review counts.

### Logging examples

The following lines are examples from the recorded `service.log`:

```text
2026-10-18 08:23:13,745 INFO endpoint=/score status=200 latency_ms=16.5 txn_id=T800001 fraud_probability=0.043
2026-10-18 08:23:13,783 WARNING endpoint=/score status=200 latency_ms=17.5 txn_id=T800003 fraud_probability=0.1765 guardrail_override=true
2026-10-18 08:23:13,847 ERROR endpoint=/case status=503 latency_ms=17.5 txn_id=T900099 fraud_probability=0.9597
```

The log examples illustrate successful scoring, a guardrail override, and a simulated case-note service failure. The committed `service.log` contains exactly the 10 POST request lines of this run (5 INFO, 4 WARNING, 1 ERROR), with no stack traces; `GET` requests are not logged.

## Limitations

- The dataset is synthetic and uses a hand-built fraud rule, so real-world fraud patterns may differ substantially.
- The TF-IDF retriever relies on textual similarity and may miss relevant policies expressed using different wording.
- The language model is a scripted mock, so its failure patterns do not fully represent a real language model service.
- The threshold was selected using only 73 validation fraud cases, so the estimate may be unstable.
- The reported metrics and latency are from a limited test run and should not be treated as production performance guarantees.
- The service has not been validated on a real payment processor's production traffic or operational environment.
- An unseen `city` or `payment_mode` value is not rejected: it is one-hot encoded as all zeros and scored anyway, so such a request is scored with less information than a known one.
