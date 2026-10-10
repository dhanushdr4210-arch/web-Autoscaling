import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


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


def make_model(alpha):
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )


def conformal_score(y, lower, upper):
    return np.maximum(
        np.maximum(lower - y, y - upper),
        0.0,
    )


def conformal_quantile(scores, coverage):
    n = len(scores)

    rank = int(np.ceil((n + 1) * coverage)) - 1
    rank = min(max(rank, 0), n - 1)

    return np.sort(scores)[rank]


def fit_models(X, y):

    models = {}

    for name, q in [
        ("p10", 0.10),
        ("p50", 0.50),
        ("p90", 0.90),
    ]:
        model = make_model(q)
        model.fit(X, y)
        models[name] = model

    return models


def predict(models, X):

    return {
        name: model.predict(X)
        for name, model in models.items()
    }


def run_experiment(df):

    X = df[FEATURES]
    y = df[TARGET].values

    outer = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(X), 1
    ):

        train_idx = np.array(train_idx)

        # Split training data into proper training + calibration
        inner = KFold(
            n_splits=4,
            shuffle=True,
            random_state=100 + fold,
        )

        proper_rel, cal_rel = next(
            inner.split(train_idx)
        )

        proper_idx = train_idx[proper_rel]
        cal_idx = train_idx[cal_rel]

        models = fit_models(
            X.iloc[proper_idx],
            y[proper_idx],
        )

        cal_pred = predict(
            models,
            X.iloc[cal_idx],
        )

        test_pred = predict(
            models,
            X.iloc[test_idx],
        )

        # ----------------------------------------------------
        # Global conformal P90
        # ----------------------------------------------------

        global_scores = conformal_score(
            y[cal_idx],
            cal_pred["p10"],
            cal_pred["p90"],
        )

        global_adj = conformal_quantile(
            global_scores,
            0.90,
        )

        global_lower = (
            test_pred["p10"] - global_adj
        )

        global_upper = (
            test_pred["p90"] + global_adj
        )

        global_coverage = np.mean(
            (y[test_idx] >= global_lower)
            &
            (y[test_idx] <= global_upper)
        )

        global_width = np.mean(
            global_upper - global_lower
        )

        # ----------------------------------------------------
        # Cache-aware calibration
        # ----------------------------------------------------

        cache_adj = {}

        for state in [0, 1]:

            mask = (
                df.iloc[cal_idx]["image_cache_state"]
                .values == state
            )

            state_scores = global_scores[mask]

            if len(state_scores) == 0:
                cache_adj[state] = global_adj
            else:
                cache_adj[state] = conformal_quantile(
                    state_scores,
                    0.90,
                )

        test_cache = (
            df.iloc[test_idx]["image_cache_state"]
            .values
        )

        cache_lower = test_pred["p10"].copy()
        cache_upper = test_pred["p90"].copy()

        for state in [0, 1]:

            mask = test_cache == state

            cache_lower[mask] -= cache_adj[state]
            cache_upper[mask] += cache_adj[state]

        cache_coverage = np.mean(
            (y[test_idx] >= cache_lower)
            &
            (y[test_idx] <= cache_upper)
        )

        cache_width = np.mean(
            cache_upper - cache_lower
        )

        p50_mae = mean_absolute_error(
            y[test_idx],
            test_pred["p50"],
        )

        results.append({
            "fold": fold,
            "p50_mae": p50_mae,

            "global_p90_coverage":
                global_coverage,

            "global_p90_width":
                global_width,

            "global_adjustment_ms":
                global_adj,

            "cache_aware_p90_coverage":
                cache_coverage,

            "cache_aware_p90_width":
                cache_width,

            "warm_adjustment_ms":
                cache_adj[1],

            "cold_adjustment_ms":
                cache_adj[0],
        })

        print(
            f"\nFold {fold}"
        )

        print(
            f"P50 MAE              : "
            f"{p50_mae:.2f} ms"
        )

        print(
            f"Global P90 coverage  : "
            f"{global_coverage * 100:.1f}%"
        )

        print(
            f"Global P90 width     : "
            f"{global_width:.2f} ms"
        )

        print(
            f"Cache-aware coverage : "
            f"{cache_coverage * 100:.1f}%"
        )

        print(
            f"Cache-aware width    : "
            f"{cache_width:.2f} ms"
        )

        print(
            f"Warm adjustment      : "
            f"{cache_adj[1]:.2f} ms"
        )

        print(
            f"Cold adjustment      : "
            f"{cache_adj[0]:.2f} ms"
        )

    return pd.DataFrame(results)


def main():

    print("=" * 80)
    print("STARTSCALE — CACHE-AWARE P90 CONFORMAL CALIBRATION")
    print("=" * 80)

    df = pd.read_csv(DATA)

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
            "Missing feature values detected."
        )

    results = run_experiment(df)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"\nP50 MAE                 : "
        f"{results['p50_mae'].mean():.2f} ms"
    )

    print(
        f"\nGLOBAL P90"
    )

    print(
        f"Coverage                : "
        f"{results['global_p90_coverage'].mean() * 100:.2f}%"
    )

    print(
        f"Width                   : "
        f"{results['global_p90_width'].mean():.2f} ms"
    )

    print(
        f"Adjustment              : "
        f"{results['global_adjustment_ms'].mean():.2f} ms"
    )

    print(
        f"\nCACHE-AWARE P90"
    )

    print(
        f"Coverage                : "
        f"{results['cache_aware_p90_coverage'].mean() * 100:.2f}%"
    )

    print(
        f"Width                   : "
        f"{results['cache_aware_p90_width'].mean():.2f} ms"
    )

    print(
        f"Warm adjustment         : "
        f"{results['warm_adjustment_ms'].mean():.2f} ms"
    )

    print(
        f"Cold adjustment         : "
        f"{results['cold_adjustment_ms'].mean():.2f} ms"
    )

    results.to_csv(
        "cache_aware_conformal_results.csv",
        index=False,
    )

    print("\nSaved:")
    print("cache_aware_conformal_results.csv")

    print(
        "\nCACHE-AWARE CALIBRATION COMPLETE"
    )


if __name__ == "__main__":
    main()