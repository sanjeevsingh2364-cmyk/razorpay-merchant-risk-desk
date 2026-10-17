"""Part C (retrieval): TF-IDF lookup of the fraud policy that fits a question."""
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_FILE = "fraud_policies.json"
QUERY_FILE = "retrieval_queries.json"

# Chosen from the measured scores: highest out-of-scope score is 0.4251 and
# the lowest in-scope score is 0.4862, so 0.45 sits between them.
MIN_SCORE = 0.45

with open(POLICY_FILE) as f:
    POLICIES = json.load(f)

_vectorizer = TfidfVectorizer()
_policy_matrix = _vectorizer.fit_transform([p["text"] for p in POLICIES])


def retrieve(query):
    """Return (policy_id, policy_text, score) for the most similar policy."""
    similarities = cosine_similarity(_vectorizer.transform([query]), _policy_matrix)[0]
    best = int(similarities.argmax())
    return POLICIES[best]["policy_id"], POLICIES[best]["text"], float(similarities[best])


def main():
    with open(QUERY_FILE) as f:
        queries = json.load(f)

    print(f"{'query':<68}{'expected':<10}{'retrieved':<10}{'score':<8}")
    hits, in_scope = 0, 0
    in_scope_scores, out_scope_scores = [], []
    for item in queries:
        policy_id, _, score = retrieve(item["query"])
        expected = item["expected_policy"]
        if expected is None:
            out_scope_scores.append(score)
        else:
            in_scope += 1
            hits += policy_id == expected
            in_scope_scores.append(score)
        print(f"{item['query']:<68}{str(expected):<10}{policy_id:<10}{score:.4f}")

    print(f"\nin-scope top-1 hits: {hits} of {in_scope}")
    print(f"highest out-of-scope score: {max(out_scope_scores):.4f}")
    print(f"lowest in-scope score:      {min(in_scope_scores):.4f}")
    print(f"MIN_SCORE chosen:           {MIN_SCORE}")


if __name__ == "__main__":
    main()