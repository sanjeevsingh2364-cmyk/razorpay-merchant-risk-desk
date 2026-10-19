"""Part C (case notes): prompt, three validation gates, retry, fallback, guardrail."""
import json

import mock_llm
from retrieval import MIN_SCORE, retrieve

SCENARIO_FILE = "case_scenarios.json"
GUARDRAIL_PROBABILITY = 0.5
NOTE_KEYS = ["summary", "priority", "recommended_team", "recommended_action", "policy_id"]

# The case-note rules, written once in JSON Schema vocabulary.
NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_team": {"type": "string",
                             "enum": ["card_risk", "upi_risk", "merchant_ops", "customer_care"]},
        "recommended_action": {"type": "string",
                               "enum": ["approve", "hold_for_review", "block_and_notify"]},
        "policy_id": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5", "P6"]},
    },
    "required": NOTE_KEYS,
}


def build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text):
    """Role, task, constraints and output format, in that order."""
    properties = NOTE_SCHEMA["properties"]
    allowed = "\n".join(f"- {key}: " + (", ".join(properties[key]["enum"]) if "enum" in properties[key]
                                        else "free text, one or two sentences")
                        for key in NOTE_KEYS)
    return (
        "ROLE:\n"
        "You are a fraud analyst assistant on the Merchant Risk Desk.\n\n"
        "TASK:\n"
        "Write a short case note for the held payment below, using the policy text provided.\n"
        f"TXN_ID: {txn_id}\n"
        f"FRAUD_PROBABILITY: {fraud_probability}\n"
        f"AMOUNT_INR: {amount_inr}\n"
        f"POLICY_ID: {policy_id}\n"
        f"POLICY_TEXT: {policy_text}\n\n"
        "CONSTRAINTS:\n"
        "- Use only the policy text above; do not invent rules.\n"
        "- Reply with one JSON object and nothing else.\n\n"
        "OUTPUT FORMAT:\n"
        "A JSON object with exactly these five keys and allowed values:\n"
        f"{allowed}\n"
    )


def validate(text, retrieved_policy_id):
    """Return (note, gate): the parsed note and 0 if it passed, else (None, failed gate)."""
    # Gate 1: valid JSON object
    try:
        note = json.loads(text)
    except json.JSONDecodeError:
        return None, 1
    if not isinstance(note, dict):
        return None, 1

    # Gate 2: all required keys present and each a string
    for key in NOTE_SCHEMA["required"]:
        if key not in note or not isinstance(note[key], str):
            return None, 2

    # Gate 3: allowed values, and the policy must be the one we retrieved
    for key, rule in NOTE_SCHEMA["properties"].items():
        if "enum" in rule and note[key] not in rule["enum"]:
            return None, 3
    if note["policy_id"] != retrieved_policy_id:
        return None, 3
    return note, 0


def safe_default_note(policy_id):
    return {"summary": "Automated case note unavailable; manual review required.",
            "priority": "high", "recommended_team": "customer_care",
            "recommended_action": "hold_for_review", "policy_id": policy_id}


def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=3):
    """Retrieve a policy, ask the model for a note, validate, retry and apply the guardrail.

    A ConnectionError from the model is deliberately not caught here.
    """
    policy_id, policy_text, score = retrieve(question)
    if score < MIN_SCORE:
        return {"status": "no_policy_match", "attempts": 0, "failures": [], "case_note": None,
                "safe_default": False, "guardrail_override": False}

    prompt = build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text)
    failures = []
    note = None
    for _ in range(max_attempts):
        note, gate = validate(mock_llm.generate(prompt), policy_id)
        if gate == 0:
            break
        failures.append(gate)
        note = None

    safe_default = note is None
    if safe_default:
        note = safe_default_note(policy_id)

    override = False
    if note["recommended_action"] == "approve" and fraud_probability >= GUARDRAIL_PROBABILITY:
        note["recommended_action"] = "hold_for_review"
        override = True

    return {"status": "ok", "attempts": len(failures) + (0 if safe_default else 1),
            "failures": failures, "case_note": note,
            "safe_default": safe_default, "guardrail_override": override}


def main():
    mock_llm.reset()
    with open(SCENARIO_FILE) as f:
        scenarios = json.load(f)

    header = ["scenario", "policy", "attempts", "failures", "status", "safe default", "final action", "override"]
    print(f"{header[0]:<16}{header[1]:<8}{header[2]:<10}{header[3]:<12}{header[4]:<8}{header[5]:<14}{header[6]:<18}{header[7]}")
    total_calls = first_try = recovered = defaults = overrides = 0
    for number, s in enumerate(scenarios, start=1):
        policy_id = retrieve(s["question"])[0]
        result = run_case(s["txn_id"], s["fraud_probability"], s["amount_inr"], s["question"])
        total_calls += result["attempts"]
        first_try += result["attempts"] == 1 and not result["safe_default"]
        recovered += result["attempts"] > 1 and not result["safe_default"]
        defaults += result["safe_default"]
        overrides += result["guardrail_override"]
        print(f"{'S' + str(number) + ' (' + s['txn_id'] + ')':<16}{policy_id:<8}{result['attempts']:<10}"
            f"{str(result['failures']):<12}{result['status']:<8}"
            f"{'yes' if result['safe_default'] else 'no':<14}"
            f"{result['case_note']['recommended_action']:<18}"
            f"{'yes' if result['guardrail_override'] else 'no'}")
    print(f"\nTotals: {total_calls} model calls; {first_try} valid on first call; "
        f"{recovered} recovered by retry; {defaults} safe default; {overrides} guardrail override")


if __name__ == "__main__":
    main()