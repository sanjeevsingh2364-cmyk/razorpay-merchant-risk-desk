"""Part D: the Fraud Triage Service."""
import logging
import math
import re
import time
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from casenote import run_case
from prepare import add_features
from train import build_final_model

logging.basicConfig(filename="service.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("triage")

KNOWN_CATEGORIES = {"Grocery", "Electronics", "Travel", "Food Delivery", "Fashion", "Utilities"}
LARGE_PAYMENT_INR = 100_000
# amount / 30-day average; the training data peaks near 13, so anything this large is a data error
MAX_AMOUNT_RATIO = 1_000_000

# Trained once at start-up; the server answers requests after this finishes.
MODEL = build_final_model()

app = FastAPI()

metrics = {"requests_total": 0, "success_total": 0, "error_total": 0,
           "latency_total_ms": 0.0, "human_review_total": 0}


class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: Optional[float] = None
    is_new_device: int
    account_age_days: int


class CaseIn(Txn):
    question: str


@app.middleware("http")
async def observe(request: Request, call_next):
    """Single place that logs and counts every POST response, including 422 and 503."""
    started = time.perf_counter()
    request.state.info = {}
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        # No stack trace in the response or the log line; the client just sees a 500.
        response = JSONResponse(status_code=500, content={"error": "internal server error"})
        status = 500

    if request.method == "POST":
        latency_ms = (time.perf_counter() - started) * 1000
        info = request.state.info
        record_request(request.url.path, status, latency_ms, info)
    return response


def record_request(endpoint, status, latency_ms, info):
    metrics["requests_total"] += 1
    metrics["latency_total_ms"] += latency_ms
    metrics["success_total" if status < 400 else "error_total"] += 1
    if info.get("human_review"):
        metrics["human_review_total"] += 1

    message = f"endpoint={endpoint} status={status} latency_ms={latency_ms:.1f}"
    if "txn_id" in info:
        # txn_id is client-supplied: keep it to printable, space-free ASCII so one request is one log line
        message += " txn_id=" + re.sub(r"[^\x21-\x7e]", "_", str(info["txn_id"]))[:64]
    if "fraud_probability" in info:
        message += f" fraud_probability={info['fraud_probability']}"
    if info.get("safe_default"):
        message += " safe_default=true"
    if info.get("override"):
        message += " guardrail_override=true"

    if status >= 500:
        level = logging.ERROR
    elif status >= 400 or info.get("safe_default") or info.get("override"):
        level = logging.WARNING
    else:
        level = logging.INFO
    logger.log(level, message)


def find_invalid_field(txn):
    """Return (field, message) for the first impossible value, or None."""
    if not 1 <= txn.amount_inr <= 1_000_000:
        return "amount_inr", "amount_inr must be between 1 and 1,000,000"
    if not 0 <= txn.hour_of_day <= 23:
        return "hour_of_day", "hour_of_day must be between 0 and 23"
    if txn.txn_count_last_24h < 0:
        return "txn_count_last_24h", "txn_count_last_24h must not be negative"
    if not math.isfinite(txn.avg_amount_30d_inr):
        return "avg_amount_30d_inr", "avg_amount_30d_inr must be a finite number"
    if txn.avg_amount_30d_inr <= 0:
        return "avg_amount_30d_inr", "avg_amount_30d_inr must be greater than 0"
    if txn.amount_inr / txn.avg_amount_30d_inr > MAX_AMOUNT_RATIO:
        return "avg_amount_30d_inr", "avg_amount_30d_inr is implausibly small compared with amount_inr"
    if txn.distance_from_home_km is not None and not math.isfinite(txn.distance_from_home_km):
        return "distance_from_home_km", "distance_from_home_km must be a finite number"
    if txn.merchant_category.strip().title() not in KNOWN_CATEGORIES:
        return "merchant_category", "merchant_category is not a known category"
    return None


def reject(problem, info):
    field, message = problem
    return JSONResponse(status_code=400, content={"error": message, "field": field})


def fraud_probability(txn):
    """Run the same cleaning, feature and encoding steps as training, then score."""
    row = pd.DataFrame([{
        "merchant_category": txn.merchant_category.strip().title(),
        "city": txn.city.strip().title(),
        "payment_mode": txn.payment_mode,
        "amount_inr": txn.amount_inr,
        "hour_of_day": txn.hour_of_day,
        "txn_count_last_24h": txn.txn_count_last_24h,
        "avg_amount_30d_inr": txn.avg_amount_30d_inr,
        "distance_from_home_km": (float("nan") if txn.distance_from_home_km is None
                                  else txn.distance_from_home_km),
        "is_new_device": txn.is_new_device,
        "account_age_days": txn.account_age_days,
    }])
    encoded = pd.get_dummies(add_features(row), dtype=int)
    encoded = encoded.reindex(columns=MODEL["columns"], fill_value=0)
    encoded["distance_from_home_km"] = encoded["distance_from_home_km"].fillna(MODEL["median"])
    return float(MODEL["model"].predict_proba(encoded)[:, 1][0])


def scoring_path(txn, info):
    """Score a payment and apply the large-payment hold. Returns (probability, flagged, decision)."""
    probability = round(fraud_probability(txn), 4)
    flagged = bool(probability >= MODEL["threshold"])
    info["fraud_probability"] = probability
    if flagged:
        decision = "hold_for_review"
    elif txn.amount_inr >= LARGE_PAYMENT_INR and txn.is_new_device == 1:
        decision = "hold_for_review"
        info["override"] = True
    else:
        decision = "approve"
    return probability, flagged, decision


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(txn: Txn, request: Request):
    info = request.state.info
    info["txn_id"] = txn.txn_id
    problem = find_invalid_field(txn)
    if problem:
        return reject(problem, info)

    probability, flagged, decision = scoring_path(txn, info)
    human_review = decision == "hold_for_review"
    info["human_review"] = human_review
    return {"txn_id": txn.txn_id, "fraud_probability": probability, "flagged": flagged,
            "decision": decision, "human_review": human_review}


@app.post("/case")
def case(txn: CaseIn, request: Request):
    info = request.state.info
    info["txn_id"] = txn.txn_id
    problem = find_invalid_field(txn)
    if problem:
        return reject(problem, info)

    probability, flagged, decision = scoring_path(txn, info)
    response = {"txn_id": txn.txn_id, "fraud_probability": probability, "flagged": flagged,
                "decision": decision, "human_review": decision == "hold_for_review",
                "case_note": None, "attempts": 0, "safe_default": False,
                "guardrail_override": False}

    if flagged:
        try:
            result = run_case(txn.txn_id, probability, txn.amount_inr, txn.question)
        except ConnectionError:
            return JSONResponse(status_code=503, content={
                "error": "case-note service temporarily unavailable, try again later"})

        if result["status"] == "no_policy_match":
            response["decision"] = "hold_for_review"
        else:
            note = dict(result["case_note"])
            override = result["guardrail_override"]
            # A flagged payment is never auto-approved, even below Part C's 0.5 rule.
            if note["recommended_action"] == "approve":
                note["recommended_action"] = "hold_for_review"
                override = True
            response.update(case_note=note, attempts=result["attempts"],
                            safe_default=result["safe_default"], guardrail_override=override,
                            decision=note["recommended_action"])
            info["safe_default"] = result["safe_default"]
            if override:
                info["override"] = True

        response["human_review"] = (response["decision"] in ("hold_for_review", "block_and_notify")
                                    or response["safe_default"])

    info["human_review"] = response["human_review"]
    return response


@app.get("/metrics")
def get_metrics():
    total = metrics["requests_total"]
    return {
        "requests_total": total,
        "success_total": metrics["success_total"],
        "error_total": metrics["error_total"],
        "success_rate": round(metrics["success_total"] / total, 4) if total else 0.0,
        "error_rate": round(metrics["error_total"] / total, 4) if total else 0.0,
        "avg_latency_ms": round(metrics["latency_total_ms"] / total, 2) if total else 0.0,
        "human_review_total": metrics["human_review_total"],
        "human_review_rate": round(metrics["human_review_total"] / total, 4) if total else 0.0,
    }