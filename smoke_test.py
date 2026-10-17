"""Part D smoke test: eleven requests against a freshly started service."""
import json
import time

import requests

BASE_URL = "http://127.0.0.1:8000"

BENIGN = {"merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI", "amount_inr": 1500,
          "hour_of_day": 13, "txn_count_last_24h": 0, "avg_amount_30d_inr": 1600,
          "distance_from_home_km": 3.0, "is_new_device": 0, "account_age_days": 900}
RISKY = {"merchant_category": "Electronics", "city": "Mumbai", "payment_mode": "Card", "amount_inr": 45000,
         "hour_of_day": 2, "txn_count_last_24h": 9, "avg_amount_30d_inr": 3000,
         "distance_from_home_km": 60.0, "is_new_device": 1, "account_age_days": 40}
BIG = {**BENIGN, "merchant_category": "Utilities", "payment_mode": "NetBanking", "amount_inr": 150000,
       "avg_amount_30d_inr": 250000, "distance_from_home_km": 0.0, "is_new_device": 1,
       "account_age_days": 2200}


def wait_for_service(timeout_seconds=120):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if requests.get(f"{BASE_URL}/health", timeout=2).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise SystemExit("service did not come up within 120 seconds")


def has_fields(body, **expected):
    return all(body.get(key) == value for key, value in expected.items())


# Each request: id, category, method, path, payload, expected status, extra check on the body
REQUESTS = [
    ("Q1", "happy", "GET", "/health", None, 200, lambda b: b == {"status": "ok"}),
    ("Q2", "happy", "POST", "/score", {**BENIGN, "txn_id": "T800001"}, 200,
     lambda b: has_fields(b, flagged=False, decision="approve", human_review=False)),
    ("Q3", "escalation", "POST", "/score", {**RISKY, "txn_id": "T800002"}, 200,
     lambda b: has_fields(b, flagged=True, decision="hold_for_review", human_review=True)),
    ("Q4", "escalation", "POST", "/score", {**BIG, "txn_id": "T800003"}, 200,
     lambda b: has_fields(b, flagged=False, decision="hold_for_review", human_review=True)),
    ("Q5", "edge", "POST", "/score", {**BENIGN, "amount_inr": -5, "txn_id": "T800004"}, 400,
     lambda b: "error" in b and b.get("field") == "amount_inr"),
    ("Q6", "edge", "POST", "/score",
     {**{k: v for k, v in BENIGN.items() if k != "hour_of_day"}, "txn_id": "T800005"}, 422,
     lambda b: True),
    ("Q7", "escalation", "POST", "/case",
     {**RISKY, "txn_id": "T900002",
      "question": "more than 6 UPI payments in 24 hours velocity alert"}, 200,
     lambda b: has_fields(b, flagged=True, attempts=2, safe_default=False, human_review=True)
     and (b.get("case_note") or {}).get("policy_id") == "P2"),
    ("Q8", "escalation", "POST", "/case",
     {**RISKY, "txn_id": "T900005",
      "question": "electronics merchant payment on an account younger than 30 days"}, 200,
     lambda b: has_fields(b, attempts=3, safe_default=True, decision="hold_for_review",
                          human_review=True)),
    ("Q9", "edge", "POST", "/case",
     {**RISKY, "txn_id": "T900099",
      "question": "card payment above INR 50000 from a new device"}, 503,
     lambda b: "error" in b),
    ("Q10", "happy", "POST", "/case",
     {**BENIGN, "txn_id": "T800001",
      "question": "card payment above INR 50000 from a new device"}, 200,
     lambda b: has_fields(b, case_note=None, attempts=0, decision="approve", human_review=False)),
    ("Q11", "edge", "POST", "/case",
     {**RISKY, "txn_id": "T800006", "question": "what is the cafeteria menu today"}, 200,
     lambda b: has_fields(b, case_note=None, attempts=0, decision="hold_for_review",
                          human_review=True)),
]


def main():
    wait_for_service()
    passed_by_category = {}
    print(f"{'id':<5}{'category':<12}{'expected':<10}{'observed':<10}result")
    for request_id, category, method, path, payload, expected_status, body_check in REQUESTS:
        response = requests.request(method, BASE_URL + path, json=payload, timeout=30)
        try:
            body = response.json()
        except ValueError:
            body = {}
        passed = response.status_code == expected_status and bool(body_check(body))
        counts = passed_by_category.setdefault(category, [0, 0])
        counts[0] += passed
        counts[1] += 1
        print(f"{request_id:<5}{category:<12}{expected_status:<10}{response.status_code:<10}"
              f"{'PASS' if passed else 'FAIL'}")

    print()
    for category, (passed, total) in passed_by_category.items():
        print(f"{category}: {passed}/{total} passed")
    print(f"overall: {sum(p for p, _ in passed_by_category.values())}/"
          f"{sum(t for _, t in passed_by_category.values())} passed")
    print("\n/metrics:")
    print(json.dumps(requests.get(f"{BASE_URL}/metrics", timeout=10).json(), indent=2))


if __name__ == "__main__":
    main()