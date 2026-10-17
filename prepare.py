"""Part A: audit, clean and engineer the raw payment export."""
import pandas as pd

RAW_FILE = "transactions_raw.csv"
MIN_AMOUNT = 1
MAX_AMOUNT = 1_000_000
EXCLUDED_COLUMNS = ["txn_id", "customer_id", "chargeback_filed", "is_fraud"]


def normalise_text(series):
    """Strip surrounding spaces and convert to title case."""
    return series.str.strip().str.title()


def clean_transactions(raw_df):
    """Return a cleaned copy of the raw export; the input is left unchanged."""
    df = raw_df.drop_duplicates().copy()
    df["merchant_category"] = normalise_text(df["merchant_category"])
    df["city"] = normalise_text(df["city"])
    df = df[df["amount_inr"].notna()]
    df = df[(df["amount_inr"] >= MIN_AMOUNT) & (df["amount_inr"] <= MAX_AMOUNT)]
    return df.reset_index(drop=True)


def add_features(df):
    """Return a copy with amount_vs_avg_ratio, is_night and high_value added."""
    out = df.copy()
    out["amount_vs_avg_ratio"] = out["amount_inr"] / out["avg_amount_30d_inr"]
    out["is_night"] = (out["hour_of_day"] <= 5).astype(int)
    out["high_value"] = (out["amount_inr"] > 50_000).astype(int)
    return out


def print_audit(raw):
    deduped = raw.drop_duplicates()
    print("=== Audit (raw file) ===")
    print(f"rows, columns as read            : {raw.shape[0]}, {raw.shape[1]}")
    print(f"exact duplicate rows             : {raw.duplicated().sum()}")
    print(f"distinct merchant_category spell.: {raw['merchant_category'].nunique()}")
    print(f"distinct city spellings          : {raw['city'].nunique()}")
    print("=== After removing duplicates ===")
    print(f"rows                             : {len(deduped)}")
    print(f"missing amount_inr               : {deduped['amount_inr'].isna().sum()}")
    print(f"missing distance_from_home_km    : {deduped['distance_from_home_km'].isna().sum()}")
    out_of_range = (deduped["amount_inr"] < MIN_AMOUNT) | (deduped["amount_inr"] > MAX_AMOUNT)
    print(f"amount_inr < 1 or > 1,000,000    : {out_of_range.sum()}")


def print_cleaned_checkpoints(cleaned, featured):
    print("\n=== Cleaned table ===")
    print(f"rows                             : {len(cleaned)}")
    print(f"duplicate rows                   : {cleaned.duplicated().sum()}")
    print("merchant_category counts         :", cleaned["merchant_category"].value_counts().to_dict())
    print(f"distinct cities                  : {cleaned['city'].nunique()}")
    print("payment_mode counts              :", cleaned["payment_mode"].value_counts().to_dict())
    print(f"min / max amount_inr             : {cleaned['amount_inr'].min()} / {cleaned['amount_inr'].max()}")
    print(f"missing distance_from_home_km    : {cleaned['distance_from_home_km'].isna().sum()}")
    print(f"is_night sum                     : {featured['is_night'].sum()}")
    print(f"high_value sum                   : {featured['high_value'].sum()}")
    print(f"max amount_vs_avg_ratio          : {featured['amount_vs_avg_ratio'].max():.2f}")

    print("\n=== Leakage audit: chargeback_filed ===")
    leak = cleaned.groupby("chargeback_filed")["is_fraud"].agg(rows="size", fraud="sum")
    print(leak.to_string())

    fraud_rows = int(cleaned["is_fraud"].sum())
    fraud_rate = fraud_rows / len(cleaned)
    print("\n=== Never-fraud baseline ===")
    print(f"fraud rows                       : {fraud_rows} of {len(cleaned)} ({fraud_rate:.2%})")
    print(f"accuracy of 'never fraud'        : {1 - fraud_rate:.2%}")
    print("recall of 'never fraud'          : 0")

    model_inputs = [c for c in featured.columns if c not in EXCLUDED_COLUMNS]
    print("\nmodel-input columns (before one-hot encoding):")
    print(model_inputs)


if __name__ == "__main__":
    raw = pd.read_csv(RAW_FILE)
    print_audit(raw)
    cleaned = clean_transactions(raw)
    featured = add_features(cleaned)
    print_cleaned_checkpoints(cleaned, featured)