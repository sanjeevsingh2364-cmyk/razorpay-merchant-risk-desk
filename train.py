"""Part B: fraud model, imbalance handling and alert threshold."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare import EXCLUDED_COLUMNS, RAW_FILE, add_features, clean_transactions

SEED = 42
THRESHOLD_GRID = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]
MIN_RECALL = 0.65
PARAM_GRID = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}


def make_forest(**params):
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, **params)


def make_logistic(class_weight=None):
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000, class_weight=class_weight))])


def make_folds():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def load_model_table():
    """Cleaned, feature-engineered and one-hot encoded table plus the target."""
    featured = add_features(clean_transactions(pd.read_csv(RAW_FILE)))
    y = featured["is_fraud"]
    X = pd.get_dummies(featured.drop(columns=EXCLUDED_COLUMNS), dtype=int)
    leaky_column = featured["chargeback_filed"]
    return X, y, leaky_column


def split_data(X, y):
    """Stratified 60/20/20 split, then fill distance with the training median."""
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED)
    median = X_train["distance_from_home_km"].median()
    for part in (X_train, X_val, X_test):
        part["distance_from_home_km"] = part["distance_from_home_km"].fillna(median)
    return X_train, X_val, X_test, y_train, y_val, y_test, median


def tune_forest(X_train, y_train):
    search = GridSearchCV(make_forest(), PARAM_GRID, scoring="f1", cv=make_folds(), n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def pick_threshold(y_true, probabilities):
    """Highest grid threshold whose recall is at least MIN_RECALL (0.1 if none)."""
    qualifying = [t for t in THRESHOLD_GRID
                  if recall_score(y_true, probabilities >= t, zero_division=0) >= MIN_RECALL]
    return max(qualifying) if qualifying else 0.1


def build_final_model():
    """Repeat the whole Part B pipeline and return what the service needs."""
    X, y, _ = load_model_table()
    X_train, X_val, _, y_train, y_val, _, median = split_data(X, y)
    search = tune_forest(X_train, y_train)
    threshold = pick_threshold(y_val, search.best_estimator_.predict_proba(X_val)[:, 1])
    return {"model": search.best_estimator_, "columns": list(X.columns),
            "median": float(median), "threshold": threshold}


def scores(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def oversample_fraud(X_train, y_train):
    fraud_X = X_train[y_train == 1]
    legit_X = X_train[y_train == 0]
    extra_fraud = fraud_X.sample(n=len(legit_X), replace=True, random_state=SEED)
    X_over = pd.concat([legit_X, extra_fraud])
    y_over = pd.Series([0] * len(legit_X) + [1] * len(extra_fraud), index=X_over.index)
    return X_over, y_over


def show(title, table):
    print(f"\n=== {title} ===")
    print(table.round(3).to_string())


def main():
    X, y, leaky_column = load_model_table()
    print("Input columns:", len(X.columns))
    print(list(X.columns))
    X_train, X_val, X_test, y_train, y_val, y_test, median = split_data(X, y)

    print("\n=== Split ===")
    for name, part, target in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
        print(f"{name:<11}: {len(part)} rows, {int(target.sum())} fraud")
    print(f"training median of distance_from_home_km: {median}")

    # Three ways to treat imbalance, evaluated on validation at 0.5
    X_over, y_over = oversample_fraud(X_train, y_train)
    print(f"\nover-sampled training rows: {len(X_over)} ({int(y_over.sum())} fraud)")
    variants = {
        "plain": make_logistic().fit(X_train, y_train),
        "class_weight=balanced": make_logistic("balanced").fit(X_train, y_train),
        "over-sampled": make_logistic().fit(X_over, y_over),
    }
    rows = {name: scores(y_val, model.predict_proba(X_val)[:, 1] >= 0.5) for name, model in variants.items()}
    show("Imbalance treatments, validation at 0.5", pd.DataFrame(rows).T)

    # Cross-validation on the training split only
    folds = make_folds()
    cv_rows = {}
    for name, estimator in [("balanced logistic regression", make_logistic("balanced")),
                            ("untuned random forest", make_forest())]:
        f1_scores = cross_val_score(estimator, X_train, y_train, scoring="f1", cv=folds, n_jobs=-1)
        cv_rows[name] = {"mean_f1": f1_scores.mean(), "std_f1": f1_scores.std()}
    show("5-fold CV F1 (training split)", pd.DataFrame(cv_rows).T)

    search = tune_forest(X_train, y_train)
    best_index = search.best_index_
    print(f"\ngrid search: {len(search.cv_results_['params'])} combinations x 5 folds")
    print("best parameters:", search.best_params_)
    print(f"best mean CV F1: {search.best_score_:.3f} "
          f"(std across folds {search.cv_results_['std_test_score'][best_index]:.3f})")
    forest = search.best_estimator_

    # Threshold chosen on validation only
    val_proba = forest.predict_proba(X_val)[:, 1]
    threshold_rows = {}
    for t in THRESHOLD_GRID:
        flagged = val_proba >= t
        result = scores(y_val, flagged)
        threshold_rows[t] = {"precision": result["precision"], "recall": result["recall"],
                             "f1": result["f1"], "flagged": int(flagged.sum())}
    show("Validation threshold table", pd.DataFrame(threshold_rows).T)
    t_star = pick_threshold(y_val, val_proba)
    print(f"t_star = {t_star}")

    # One final test evaluation
    test_flagged = forest.predict_proba(X_test)[:, 1] >= t_star
    test_result = scores(y_test, test_flagged)
    print(f"\n=== Test split at t_star={t_star} (used once) ===")
    print({k: (round(float(v), 3) if k in ("precision", "recall", "f1") else int(v)) for k, v in test_result.items()})
    print("payments flagged:", int(test_flagged.sum()))
    print(f"legitimate payments held per fraud caught: {test_result['FP'] / test_result['TP']:.2f}")

    # Leakage experiment: chargeback_filed as the 27th column
    X_train_leaky = X_train.assign(chargeback_filed=leaky_column.loc[X_train.index])
    X_val_leaky = X_val.assign(chargeback_filed=leaky_column.loc[X_val.index])
    leaky_forest = make_forest(**search.best_params_).fit(X_train_leaky, y_train)
    clean_f1 = f1_score(y_val, forest.predict_proba(X_val)[:, 1] >= 0.5)
    leaky_f1 = f1_score(y_val, leaky_forest.predict_proba(X_val_leaky)[:, 1] >= 0.5)
    print("\n=== Leakage experiment (validation F1 at 0.5) ===")
    print(f"clean tuned forest: {clean_f1:.3f}   with chargeback_filed: {leaky_f1:.3f}")

    importance = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== Top 5 feature importances ===")
    print(importance.head(5).round(3).to_string())


if __name__ == "__main__":
    main()