# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview

Razorpay Merchant Risk Desk is a fraud triage service for a fictional payment processor using synthetic transaction data. Part A cleans the raw payment export and engineers features. Part B trains and evaluates fraud detection models and selects an alert threshold using validation data. Part C retrieves relevant fraud policies using TF-IDF and generates validated JSON case notes through a mock language model. Part D exposes the workflow through FastAPI with input validation, policy guardrails, logging, metrics, and smoke tests.

## Reproduce

Requires **Python 3.11 or newer**. The numbers in this README were re-verified on Python 3.12.3 (Linux) with scikit-learn 1.9.1 and pandas 3.0.6, installed from `requirements.txt`.

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

The `chargeback_filed` field records an outcome that becomes known after a payment. Using it as a model input would introduce target leakage. The `txn_id` and `customer_id` columns are identifiers rather than reliable behavioural features. These three columns are therefore excluded from the model inputs. `txn_id` is a unique label for each row, and `customer_id` is a random label that `generate_data.py` draws independently of the fraud outcome; neither describes how a payment behaves. A model could only use them to memorise individual rows or customers, which would not carry over to new payments. `is_fraud` is the target label and is not a model input either.

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

The random forest is ahead of balanced logistic regression by about 0.06 F1, which is larger than the fold-to-fold spread of either model. The gain from tuning (0.465 against 0.453) is smaller than the spread across folds, so tuning is not clear evidence of improvement. The random forest also returns probabilities that can be thresholded, and the threshold was then chosen on validation. At the default 0.5 threshold, the clean tuned forest scored a validation F1 of 0.441, about the same as plain logistic regression (0.442, but with recall of only 0.342). Logistic regression was not threshold-tuned in this project, so that comparison is not like for like. The test split was used once, at `t_star = 0.4`.

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

**Purpose.** The model is a triage aid that decides which payments a human analyst should look at first. It is not an automatic blocker.

**Evidence (test split, 1,184 payments, 73 fraud, threshold 0.4).**

- It flagged 203 payments (17.1% of the split) and caught 57 of 73 fraud cases (recall 0.781).
- It missed 16 fraud cases (21.9% of fraud) and held 146 legitimate payments.
- Precision was 0.281, which is about 2.56 legitimate payments held for every fraud caught.

**Controls in the service.**

- A flagged payment is never auto-approved in `/score` or `/case`.
- A payment of INR 100,000 or more from a new device is held even when the model does not flag it (smoke test Q4).
- A case note that fails validation three times falls back to a `hold_for_review` safe default.

**Residual risks.**

- About one fraud case in five was missed in testing, and roughly one payment in six was sent to review.
- The threshold was chosen from only 73 validation fraud cases, so it may not be stable.
- The data is synthetic and uses a hand-built fraud rule, so real-world performance is unknown.
- `chargeback_filed` is excluded because it is only known after the outcome; including it inflated validation F1 from 0.441 to 0.797.
- Analyst capacity must cover the review volume.

**Recommendation (cautious, not a production-readiness claim).** This recommendation rests only on a synthetic dataset with a hand-built fraud rule and on one test split containing 73 fraud cases. Those results do not show that the model is ready for production, and they say nothing certain about how it would perform on real payments. On this evidence alone, the most the results support is limited use as decision support, with every flagged payment reviewed by a human analyst, for example in a controlled pilot. Before any production use, the committee should require at least:

- re-training and re-evaluation on real labelled transaction data;
- re-selection of the alert threshold on enough real fraud cases to be stable;
- re-measurement of the miss rate and the review load against available analyst capacity;
- an acceptance decision by the committee on whether a miss rate near 22% and a review load near 17% are tolerable for the real business.

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

### Why repeated failures happen and why validation is needed

In `casenote.py` the prompt is built once per payment and the same prompt is sent on every attempt. A real hosted model called at a temperature of 0 tends to return almost the same text for the same prompt, so a retry is not an independent draw and can repeat the same defect. The mock model shows this behaviour on purpose: `mock_llm.py` injects scripted defects per `txn_id`, and its output depends only on how many times that transaction has been called. `T900005` (S5) returns broken JSON three times in a row, so all three attempts fail and the service uses the safe default.

Three layers of protection are therefore needed, and the scenario results show what each one does:

- **Validation** has three gates: valid JSON (gate 1), all five keys present as strings (gate 2), and allowed values plus the retrieved policy ID (gate 3). It caught the broken JSON in S2, the missing key in S3, and the bad values in S4, S7 (wrong policy) and S10 (unknown team).
- **Retries** recovered 6 of the 10 scenarios (S2, S3, S4, S7, S9, S10) because their defects were scripted for the first call or calls only.
- **The safe default** applies when retries run out (S5). It is a `hold_for_review` note with `high` priority, so a failed model call can never turn into an approval.
- **The guardrail** covers a different failure. In S6 the model returned a well-formed note recommending `approve` for a payment with fraud probability 0.58. That note passes all three validation gates because `approve` is an allowed value, so only the guardrail (an `approve` at probability 0.5 or higher becomes `hold_for_review`) stopped it. Validation checks the form of the output, and the guardrail checks that its content is safe.

## Part D — FastAPI Service and Smoke Tests

The FastAPI service exposes health, scoring, case-note, metrics, and API documentation endpoints. The smoke-test suite exercised the expected success, escalation, and edge-case behaviours.

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