import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_absolute_error


# ============================================================
# CONFIG
# ============================================================

FILE = "v3_master_150_v3_spatial_features.csv"
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


QUANTILES = [
    0.05,
    0.10,
    0.50,
    0.90,
    0.95,
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(FILE)

print("=" * 80)
print("STARTSCALE — QUANTILE READINESS-LATENCY PREDICTION")
print("=" * 80)

print("\nDataset:", df.shape)


# ============================================================
# FEATURES
# ============================================================

X = df[FEATURES].copy()

X["image_cache_state"] = (
    X["image_cache_state"]
    .astype(str)
    .str.lower()
    .map({
        "cold": 0,
        "warm": 1
    })
)

if X.isna().any().any():
    raise ValueError("Missing feature values detected.")

y = df[TARGET].values


# ============================================================
# QUANTILE MODEL
# ============================================================

def make_quantile_model(q):

    return GradientBoostingRegressor(
        loss="quantile",
        alpha=q,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )


# ============================================================
# METRICS
# ============================================================

def pinball_loss(y_true, prediction, q):

    error = y_true - prediction

    return np.mean(
        np.maximum(
            q * error,
            (q - 1) * error
        )
    )


def evaluate_predictions(
    y_true,
    predictions
):

    results = {}

    for q in QUANTILES:

        pred = predictions[q]

        results[f"pinball_{q}"] = pinball_loss(
            y_true,
            pred,
            q
        )

    # --------------------------------------------------------
    # Point prediction error
    # --------------------------------------------------------

    results["MAE_P50"] = mean_absolute_error(
        y_true,
        predictions[0.50]
    )

    # --------------------------------------------------------
    # P90 interval: P10 → P90
    # --------------------------------------------------------

    lower90 = predictions[0.10]
    upper90 = predictions[0.90]

    coverage90 = np.mean(
        (y_true >= lower90) &
        (y_true <= upper90)
    )

    width90 = np.mean(
        upper90 - lower90
    )

    results["coverage90"] = coverage90
    results["width90_ms"] = width90

    # --------------------------------------------------------
    # P95 interval: P5 → P95
    # --------------------------------------------------------

    lower95 = predictions[0.05]
    upper95 = predictions[0.95]

    coverage95 = np.mean(
        (y_true >= lower95) &
        (y_true <= upper95)
    )

    width95 = np.mean(
        upper95 - lower95
    )

    results["coverage95"] = coverage95
    results["width95_ms"] = width95

    return results


# ============================================================
# VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

random_splits = list(
    kf.split(df)
)


# ------------------------------------------------------------
# Stratified
# ------------------------------------------------------------

labels = df["source_condition"].astype(str)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

stratified_splits = list(
    skf.split(df, labels)
)


# ------------------------------------------------------------
# LOCO
# ------------------------------------------------------------

groups = df["source_condition"].astype(str)

gkf = GroupKFold(
    n_splits=5
)

loco_splits = list(
    gkf.split(
        df,
        groups=groups
    )
)


# ============================================================
# RUN VALIDATION
# ============================================================

all_results = []


for validation_name, splits in [

    ("Random 5-Fold", random_splits),

    ("Stratified 5-Fold", stratified_splits),

    ("LOCO", loco_splits),

]:

    print("\n" + "=" * 80)
    print(validation_name)
    print("=" * 80)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splits,
        start=1
    ):

        predictions = {}

        for q in QUANTILES:

            model = make_quantile_model(q)

            model.fit(
                X.iloc[train_idx],
                y[train_idx]
            )

            predictions[q] = model.predict(
                X.iloc[test_idx]
            )

        metrics = evaluate_predictions(
            y[test_idx],
            predictions
        )

        metrics["fold"] = fold

        fold_results.append(metrics)

        print(
            f"Fold {fold}: "
            f"P50 MAE={metrics['MAE_P50']:.2f} ms | "
            f"P90 coverage={metrics['coverage90']*100:.1f}% | "
            f"P95 coverage={metrics['coverage95']*100:.1f}%"
        )

    fold_df = pd.DataFrame(fold_results)

    summary = {
        "validation": validation_name,

        "P50_MAE_ms":
            fold_df["MAE_P50"].mean(),

        "P90_coverage":
            fold_df["coverage90"].mean(),

        "P90_width_ms":
            fold_df["width90_ms"].mean(),

        "P95_coverage":
            fold_df["coverage95"].mean(),

        "P95_width_ms":
            fold_df["width95_ms"].mean(),

        "P05_pinball":
            fold_df["pinball_0.05"].mean(),

        "P10_pinball":
            fold_df["pinball_0.1"].mean(),

        "P50_pinball":
            fold_df["pinball_0.5"].mean(),

        "P90_pinball":
            fold_df["pinball_0.9"].mean(),

        "P95_pinball":
            fold_df["pinball_0.95"].mean(),
    }

    all_results.append(summary)

    print("\nSummary:")

    print(
        f"P50 MAE       : "
        f"{summary['P50_MAE_ms']:.2f} ms"
    )

    print(
        f"P90 coverage  : "
        f"{summary['P90_coverage']*100:.2f}%"
    )

    print(
        f"P90 width     : "
        f"{summary['P90_width_ms']:.2f} ms"
    )

    print(
        f"P95 coverage  : "
        f"{summary['P95_coverage']*100:.2f}%"
    )

    print(
        f"P95 width     : "
        f"{summary['P95_width_ms']:.2f} ms"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    "quantile_readiness_results.csv",
    index=False
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("CALIBRATION TARGETS")
print("=" * 80)

print(
    """
Ideal reference points:

P90 interval:
    Target coverage ≈ 90%

P95 interval:
    Target coverage ≈ 95%

But:

    Higher coverage is not automatically better.

We also want narrow intervals.

Therefore we evaluate:

    Coverage
    +
    Sharpness / interval width
    +
    Pinball loss
"""
)


print("\nSaved:")
print("quantile_readiness_results.csv")

print("\n" + "=" * 80)
print("QUANTILE ANALYSIS COMPLETE")
print("=" * 80)