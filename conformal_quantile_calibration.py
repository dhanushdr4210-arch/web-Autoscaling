import json
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_absolute_error


# ============================================================
# STARTSCALE — CONFORMAL QUANTILE CALIBRATION
# ============================================================

DATA = "v3_master_150_v3_spatial_features.csv"
TARGET = "request_to_ready_ms"

FEATURES = [
    "node_cpu_pct_v2",
    "node_memory_pct_v2",
    "worker_cpu_mean",
    "worker_memory_mean",
    "worker_memory_max",
    "worker_memory_std",
    "node_cpu_x_memory",
    "cluster_cpu_x_memory",
    "cpu_spatial_imbalance",
    "image_cache_state",
]

QUANTILES = {
    "p05": 0.05,
    "p10": 0.10,
    "p50": 0.50,
    "p90": 0.90,
    "p95": 0.95,
}


def make_model(alpha):
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )


def fit_quantiles(X_train, y_train):
    models = {}

    for name, q in QUANTILES.items():
        model = make_model(q)
        model.fit(X_train, y_train)
        models[name] = model

    return models


def predict_quantiles(models, X):
    return {
        name: model.predict(X)
        for name, model in models.items()
    }


def conformal_adjustment(y_cal, lower_cal, upper_cal, coverage):
    """
    CQR conformity score:

        s_i = max(
            lower_i - y_i,
            y_i - upper_i,
            0
        )

    Then choose the finite-sample conformal quantile.
    """

    scores = np.maximum(
        np.maximum(lower_cal - y_cal, y_cal - upper_cal),
        0.0,
    )

    n = len(scores)

    # Finite-sample conformal quantile index
    rank = int(np.ceil((n + 1) * coverage)) - 1
    rank = min(max(rank, 0), n - 1)

    adjustment = np.sort(scores)[rank]

    return float(adjustment)


def evaluate_fold(
    X_train,
    y_train,
    X_cal,
    y_cal,
    X_test,
    y_test,
):
    # --------------------------------------------------------
    # Train raw quantile models
    # --------------------------------------------------------

    models = fit_quantiles(X_train, y_train)

    cal_pred = predict_quantiles(models, X_cal)
    test_pred = predict_quantiles(models, X_test)

    # --------------------------------------------------------
    # P50 point prediction
    # --------------------------------------------------------

    p50_mae = mean_absolute_error(
        y_test,
        test_pred["p50"],
    )

    # --------------------------------------------------------
    # P90 raw interval
    # --------------------------------------------------------

    p90_lower_cal = cal_pred["p10"]
    p90_upper_cal = cal_pred["p90"]

    p90_adjustment = conformal_adjustment(
        y_cal,
        p90_lower_cal,
        p90_upper_cal,
        coverage=0.90,
    )

    raw_p90_lower = test_pred["p10"]
    raw_p90_upper = test_pred["p90"]

    cal_p90_lower = raw_p90_lower - p90_adjustment
    cal_p90_upper = raw_p90_upper + p90_adjustment

    p90_coverage = np.mean(
        (y_test >= cal_p90_lower)
        & (y_test <= cal_p90_upper)
    )

    p90_width = np.mean(
        cal_p90_upper - cal_p90_lower
    )

    # --------------------------------------------------------
    # P95 raw interval
    # --------------------------------------------------------

    p95_lower_cal = cal_pred["p05"]
    p95_upper_cal = cal_pred["p95"]

    p95_adjustment = conformal_adjustment(
        y_cal,
        p95_lower_cal,
        p95_upper_cal,
        coverage=0.95,
    )

    raw_p95_lower = test_pred["p05"]
    raw_p95_upper = test_pred["p95"]

    cal_p95_lower = raw_p95_lower - p95_adjustment
    cal_p95_upper = raw_p95_upper + p95_adjustment

    p95_coverage = np.mean(
        (y_test >= cal_p95_lower)
        & (y_test <= cal_p95_upper)
    )

    p95_width = np.mean(
        cal_p95_upper - cal_p95_lower
    )

    return {
        "p50_mae": p50_mae,

        "p90_coverage": p90_coverage,
        "p90_width": p90_width,
        "p90_adjustment_ms": p90_adjustment,

        "p95_coverage": p95_coverage,
        "p95_width": p95_width,
        "p95_adjustment_ms": p95_adjustment,
    }


def run_random(df):
    print("\n" + "=" * 80)
    print("Random 5-Fold — Conformal Calibration")
    print("=" * 80)

    X = df[FEATURES]
    y = df[TARGET].values

    outer = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for fold, (train_cal_idx, test_idx) in enumerate(
        outer.split(X, y), 1
    ):

        train_cal_idx = np.array(train_cal_idx)

        # 80% outer training is split into:
        # 60% proper training + 20% calibration
        inner = KFold(
            n_splits=4,
            shuffle=True,
            random_state=42 + fold,
        )

        proper_rel, cal_rel = next(
            inner.split(train_cal_idx)
        )

        proper_idx = train_cal_idx[proper_rel]
        cal_idx = train_cal_idx[cal_rel]

        result = evaluate_fold(
            X.iloc[proper_idx],
            y[proper_idx],
            X.iloc[cal_idx],
            y[cal_idx],
            X.iloc[test_idx],
            y[test_idx],
        )

        result["validation"] = "Random"
        result["fold"] = fold

        results.append(result)

        print(
            f"Fold {fold}: "
            f"P50 MAE={result['p50_mae']:.2f} ms | "
            f"P90 coverage={result['p90_coverage'] * 100:.1f}% | "
            f"P95 coverage={result['p95_coverage'] * 100:.1f}% | "
            f"P90 adj={result['p90_adjustment_ms']:.2f} ms | "
            f"P95 adj={result['p95_adjustment_ms']:.2f} ms"
        )

    return results


def run_stratified(df):
    print("\n" + "=" * 80)
    print("Stratified 5-Fold — Conformal Calibration")
    print("=" * 80)

    X = df[FEATURES]
    y = df[TARGET].values

    strata = df["source_condition"].values

    outer = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for fold, (train_cal_idx, test_idx) in enumerate(
        outer.split(X, strata), 1
    ):

        train_cal_idx = np.array(train_cal_idx)

        inner = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=42 + fold,
        )

        proper_rel, cal_rel = next(
            inner.split(
                train_cal_idx,
                strata[train_cal_idx],
            )
        )

        proper_idx = train_cal_idx[proper_rel]
        cal_idx = train_cal_idx[cal_rel]

        result = evaluate_fold(
            X.iloc[proper_idx],
            y[proper_idx],
            X.iloc[cal_idx],
            y[cal_idx],
            X.iloc[test_idx],
            y[test_idx],
        )

        result["validation"] = "Stratified"
        result["fold"] = fold

        results.append(result)

        print(
            f"Fold {fold}: "
            f"P50 MAE={result['p50_mae']:.2f} ms | "
            f"P90 coverage={result['p90_coverage'] * 100:.1f}% | "
            f"P95 coverage={result['p95_coverage'] * 100:.1f}% | "
            f"P90 adj={result['p90_adjustment_ms']:.2f} ms | "
            f"P95 adj={result['p95_adjustment_ms']:.2f} ms"
        )

    return results


def run_loco(df):
    print("\n" + "=" * 80)
    print("LOCO — Conformal Calibration")
    print("=" * 80)

    X = df[FEATURES]
    y = df[TARGET].values
    groups = df["source_condition"].values

    outer = GroupKFold(n_splits=5)

    results = []

    for fold, (train_cal_idx, test_idx) in enumerate(
        outer.split(X, y, groups), 1
    ):

        train_cal_idx = np.array(train_cal_idx)

        train_groups = groups[train_cal_idx]
        unique_groups = np.unique(train_groups)

        # Use one complete condition as calibration data.
        # This keeps calibration group-aware.
        calibration_group = unique_groups[-1]

        cal_idx = train_cal_idx[
            train_groups == calibration_group
        ]

        proper_idx = train_cal_idx[
            train_groups != calibration_group
        ]

        result = evaluate_fold(
            X.iloc[proper_idx],
            y[proper_idx],
            X.iloc[cal_idx],
            y[cal_idx],
            X.iloc[test_idx],
            y[test_idx],
        )

        result["validation"] = "LOCO"
        result["fold"] = fold
        result["calibration_group"] = calibration_group

        results.append(result)

        print(
            f"Fold {fold}: "
            f"P50 MAE={result['p50_mae']:.2f} ms | "
            f"P90 coverage={result['p90_coverage'] * 100:.1f}% | "
            f"P95 coverage={result['p95_coverage'] * 100:.1f}% | "
            f"P90 adj={result['p90_adjustment_ms']:.2f} ms | "
            f"P95 adj={result['p95_adjustment_ms']:.2f} ms"
        )

    return results


def print_summary(results, name):
    r = pd.DataFrame(results)

    print("\nSummary:")
    print(f"P50 MAE       : {r['p50_mae'].mean():.2f} ms")
    print(f"P90 coverage  : {r['p90_coverage'].mean() * 100:.2f}%")
    print(f"P90 width     : {r['p90_width'].mean():.2f} ms")
    print(f"P90 adjustment: {r['p90_adjustment_ms'].mean():.2f} ms")
    print(f"P95 coverage  : {r['p95_coverage'].mean() * 100:.2f}%")
    print(f"P95 width     : {r['p95_width'].mean():.2f} ms")
    print(f"P95 adjustment: {r['p95_adjustment_ms'].mean():.2f} ms")


def main():

    print("=" * 80)
    print("STARTSCALE — CONFORMAL QUANTILE CALIBRATION")
    print("=" * 80)

    df = pd.read_csv(DATA)

    print(f"\nDataset: {df.shape}")

    # Required conversion
    df["image_cache_state"] = (
        df["image_cache_state"]
        .astype(str)
        .str.lower()
        .map({
            "cold": 0,
            "warm": 1,
        })
    )

    if df[FEATURES].isnull().any().any():
        raise ValueError(
            "Missing values detected in model features."
        )

    random_results = run_random(df)
    print_summary(random_results, "Random")

    stratified_results = run_stratified(df)
    print_summary(stratified_results, "Stratified")

    loco_results = run_loco(df)
    print_summary(loco_results, "LOCO")

    all_results = (
        random_results
        + stratified_results
        + loco_results
    )

    output = pd.DataFrame(all_results)

    output.to_csv(
        "conformal_quantile_results.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("SAVED:")
    print("conformal_quantile_results.csv")
    print("=" * 80)

    print("\nCONFORMAL CALIBRATION COMPLETE")


if __name__ == "__main__":
    main()