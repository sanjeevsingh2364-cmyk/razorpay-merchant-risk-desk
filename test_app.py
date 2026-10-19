
from fastapi.testclient import TestClient

import app as service


client = TestClient(service.app)


VALID_TXN = {
    "txn_id": "TEST1001",
    "merchant_category": "Grocery",
    "city": "Gopalganj",
    "payment_mode": "UPI",
    "amount_inr": 1500,
    "hour_of_day": 14,
    "txn_count_last_24h": 2,
    "avg_amount_30d_inr": 1200,
    "distance_from_home_km": 3.5,
    "is_new_device": 0,
    "account_age_days": 365,
}


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_valid_transaction():
    response = client.post("/score", json=VALID_TXN)

    assert response.status_code == 200

    data = response.json()
    assert data["txn_id"] == VALID_TXN["txn_id"]
    assert 0 <= data["fraud_probability"] <= 1
    assert isinstance(data["flagged"], bool)
    assert data["decision"] in ("approve", "hold_for_review")
    assert isinstance(data["human_review"], bool)


def test_score_rejects_invalid_amount():
    txn = {**VALID_TXN, "amount_inr": 0}

    response = client.post("/score", json=txn)

    assert response.status_code == 400
    assert response.json()["field"] == "amount_inr"


def test_score_rejects_unknown_category():
    txn = {**VALID_TXN, "merchant_category": "UnknownCategory"}

    response = client.post("/score", json=txn)

    assert response.status_code == 400
    assert response.json()["field"] == "merchant_category"


def test_score_rejects_missing_required_field():
    txn = VALID_TXN.copy()
    txn.pop("txn_id")

    response = client.post("/score", json=txn)

    assert response.status_code == 422


def test_case_requires_question():
    response = client.post("/case", json=VALID_TXN)

    assert response.status_code == 422


def test_metrics_returns_expected_fields():
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()
    expected_fields = {
        "requests_total",
        "success_total",
        "error_total",
        "success_rate",
        "error_rate",
        "avg_latency_ms",
        "human_review_total",
        "human_review_rate",
    }

    assert expected_fields.issubset(data.keys())


def test_score_rejects_non_finite_average():
    body = '{"txn_id": "TEST1002", "merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI", ' \
           '"amount_inr": 1500, "hour_of_day": 14, "txn_count_last_24h": 2, "avg_amount_30d_inr": NaN, ' \
           '"distance_from_home_km": 3.5, "is_new_device": 0, "account_age_days": 365}'

    response = client.post("/score", content=body, headers={"Content-Type": "application/json"})

    assert response.status_code == 400
    assert response.json()["field"] == "avg_amount_30d_inr"


def test_score_rejects_implausibly_small_average():
    txn = {**VALID_TXN, "avg_amount_30d_inr": 1e-300}

    response = client.post("/score", json=txn)

    assert response.status_code == 400
    assert response.json()["field"] == "avg_amount_30d_inr"


def test_score_large_payment_from_new_device_is_held():
    txn = {**VALID_TXN, "txn_id": "TEST1003", "merchant_category": "Utilities", "payment_mode": "NetBanking",
           "amount_inr": 150000, "avg_amount_30d_inr": 250000, "distance_from_home_km": 0.0,
           "is_new_device": 1, "account_age_days": 2200}

    response = client.post("/score", json=txn)

    assert response.status_code == 200
    assert response.json()["decision"] == "hold_for_review"
    assert response.json()["human_review"] is True
