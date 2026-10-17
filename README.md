# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview

Razorpay Merchant Risk Desk is a fraud triage service for a fictional payment processor using synthetic transaction data. Part A cleans the raw payment export and engineers features. Part B trains and evaluates fraud detection models and selects an alert threshold using validation data. Part C retrieves relevant fraud policies using TF-IDF and generates validated JSON case notes through a mock language model. Part D exposes the workflow through FastAPI with input validation, policy guardrails, logging, metrics, and smoke tests.

## Reproduce

Run the following commands from the project directory after cloning the repository.

```bash
python -m venv .venv
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
python prepare.py
python train.py
python retrieval.py
python casenote.py
```

Start the API service:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Keep the server running. Open a second terminal in the same project directory and activate the environment if necessary:

```bat
.venv\Scripts\activate
python smoke_test.py
```

The API documentation is available at `http://127.0.0.1:8000/docs` while the server is running.

The data-generation step recreates the synthetic input data. Run the preparation, training, retrieval, and case-note scripts in the order shown above before starting the API.

## Part A — Data Preparation and Leakage Audit

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

The `chargeback_filed` field records an outcome that becomes known after a payment. Using it as a model input would introduce target leakage. The `txn_id` and `customer_id` columns are identifiers rather than reliable behavioural features. These three columns are therefore excluded from the model inputs.

Fraud accounts for approximately 6.18% of the cleaned dataset. A classifier that predicts every transaction as non-fraud could achieve approximately 93.82% accuracy while detecting no fraud cases. Precision, recall, and F1-score are consequently more informative for evaluating fraud detection performance than accuracy alone.

## Part B — Model Training and Evaluation

The cleaned dataset was split into training, validation, and test sets. Missing distance values were imputed using the training-set median, and random oversampling was applied only to the training data.

| Dataset | Rows | Fraud cases |
|---|---:|---:|
| Training | 3,551 | 220 |
| Validation | 1,184 | 73 |
| Test | 1,184 | 73 |

The training-set median used to impute `distance_from_home_km` was 12.6. Random oversampling increased the training set to 6,662 rows, including 3,331 fraud cases.

### Imbalance treatment comparison

Validation-set results at a probability threshold of 0.5:

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Plain logistic regression | 0.625 | 0.342 | 0.442 |
| Class-weighted logistic regression | 0.252 | 0.781 | 0.381 |
| Oversampled logistic regression | 0.257 | 0.781 | 0.386 |

Five-fold cross-validation produced the following results:

| Model | Mean CV F1-score | Standard deviation |
|---|---:|---:|
| Balanced logistic regression | 0.402 | 0.015 |
| Untuned random forest | 0.453 | 0.052 |

Grid search evaluated 12 parameter combinations across five folds. The selected random forest parameters were `max_depth=None` and `min_samples_leaf=5`, with a mean cross-validation F1-score of 0.465 and a standard deviation of 0.042.

### Threshold selection and test results

The selected threshold was `t_star = 0.4`, based on the validation-set recall constraint of at least 0.65.

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

## Part C — Policy Retrieval and Case Notes

The retrieval component uses TF-IDF similarity to identify the relevant fraud policy for a transaction scenario. The configured minimum retrieval score is `MIN_SCORE = 0.45`.

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

## Part D — FastAPI Service and Smoke Tests

The FastAPI service exposes health, scoring, case-note, metrics, and API documentation endpoints. The smoke-test suite exercised the expected success, escalation, and edge-case behaviours.

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
  "avg_latency_ms": 31.85,
  "human_review_total": 5,
  "human_review_rate": 0.5
}
```

These figures represent one recorded test run, not a benchmark of production traffic. The metrics endpoint reports request totals, success and error rates, average latency, and human-review counts.

### Logging examples

The following lines are examples from the recorded `service.log`:

```text
2026-10-17 15:39:30,393 INFO endpoint=/score status=200 latency_ms=44.6 txn_id=T800001 fraud_probability=0.043
2026-10-17 15:39:30,496 WARNING endpoint=/score status=200 latency_ms=34.3 txn_id=T800003 fraud_probability=0.1765 guardrail_override=true
2026-10-17 15:39:30,755 ERROR endpoint=/case status=503 latency_ms=39.1 txn_id=T900099 fraud_probability=0.9597
```

The log examples illustrate successful scoring, a guardrail override, and a simulated case-note service failure.

## Limitations

- The dataset is synthetic and uses a hand-built fraud rule, so real-world fraud patterns may differ substantially.
- The TF-IDF retriever relies on textual similarity and may miss relevant policies expressed using different wording.
- The language model is a scripted mock, so its failure patterns do not fully represent a real language model service.
- The threshold was selected using only 73 validation fraud cases, so the estimate may be unstable.
- The reported metrics and latency are from a limited test run and should not be treated as production performance guarantees.
- The service has not been validated on a real payment processor's production traffic or operational environment.
