import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneGroupOut,
)
from sklearn.base import clone


# ============================================================
# 1. CONFIGURATION
# ============================================================

FILE = "v3_master_110_raw.csv"

RESULTS_OUTPUT = "v3_model_v2_comparison_110.csv"
IMPORTANCE_OUTPUT = "v3_model_v2_feature_importance_110.csv"
FEATURE_OUTPUT = "v3_model_v2_features_110.csv"

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 90)
print("STARTSCALE V3 MODEL V2 - 110 ROW DATASET")
print("=" * 90)

df = pd.read_csv(FILE)

print()
print(f"Input file : {FILE}")
print(f"Dataset shape : {df.shape}")


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def parse_percent(value):
    """
    Converts values such as:
        '6%'
        '18%'
        6
        18
    into float values:
        6.0
        18.0
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text.endswith("%"):
        text = text[:-1]

    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_cluster_state(value):
    """
    Parse pre_scale_all_nodes_json and calculate
    worker-only cluster statistics.

    The kind cluster contains:
        autoscaling-control-plane
        autoscaling-worker
        autoscaling-worker2
        autoscaling-worker3

    Only worker nodes are used for worker-level features.
    """

    if pd.isna(value):
        return {
            "worker_cpu_mean": np.nan,
            "worker_cpu_max": np.nan,
            "worker_cpu_std": np.nan,
            "worker_memory_mean": np.nan,
            "worker_memory_max": np.nan,
            "worker_memory_std": np.nan,
        }

    try:
        state = json.loads(value)
    except Exception:
        return {
            "worker_cpu_mean": np.nan,
            "worker_cpu_max": np.nan,
            "worker_cpu_std": np.nan,
            "worker_memory_mean": np.nan,
            "worker_memory_max": np.nan,
            "worker_memory_std": np.nan,
        }

    cpu_values = []
    memory_values = []

    for node_name, node_data in state.items():

        node_name_lower = str(node_name).lower()

        # Use only schedulable worker nodes.
        if "worker" not in node_name_lower:
            continue

        cpu_pct = parse_percent(
            node_data.get("cpu_pct")
        )

        memory_pct = parse_percent(
            node_data.get("memory_pct")
        )

        if not pd.isna(cpu_pct):
            cpu_values.append(cpu_pct)

        if not pd.isna(memory_pct):
            memory_values.append(memory_pct)

    if not cpu_values or not memory_values:
        return {
            "worker_cpu_mean": np.nan,
            "worker_cpu_max": np.nan,
            "worker_cpu_std": np.nan,
            "worker_memory_mean": np.nan,
            "worker_memory_max": np.nan,
            "worker_memory_std": np.nan,
        }

    return {
        "worker_cpu_mean": np.mean(cpu_values),
        "worker_cpu_max": np.max(cpu_values),
        "worker_cpu_std": np.std(cpu_values),

        "worker_memory_mean": np.mean(memory_values),
        "worker_memory_max": np.max(memory_values),
        "worker_memory_std": np.std(memory_values),
    }


def regression_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE and R2.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return mae, rmse, r2


def evaluate_splits(
    model,
    X,
    y,
    splits,
    validation_name,
):
    """
    Evaluate a model over predefined train/test splits.
    """

    predictions = np.full(
        shape=len(y),
        fill_value=np.nan,
        dtype=float,
    )

    for train_idx, test_idx in splits:

        estimator = clone(model)

        estimator.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        predictions[test_idx] = estimator.predict(
            X.iloc[test_idx]
        )

    assert not np.isnan(predictions).any()

    mae, rmse, r2 = regression_metrics(
        y,
        predictions,
    )

    return {
        "model": model.__class__.__name__,
        "validation": validation_name,
        "MAE_ms": mae,
        "RMSE_ms": rmse,
        "R2": r2,
    }


def evaluate_mean_baseline(
    X,
    y,
    splits,
    validation_name,
):
    """
    Proper fold-wise mean baseline.

    For every test fold, the prediction is the
    mean latency of the TRAINING portion only.
    """

    predictions = np.full(
        shape=len(y),
        fill_value=np.nan,
        dtype=float,
    )

    for train_idx, test_idx in splits:

        train_mean = y.iloc[train_idx].mean()

        predictions[test_idx] = train_mean

    mae, rmse, r2 = regression_metrics(
        y,
        predictions,
    )

    return {
        "model": "Mean Baseline",
        "validation": validation_name,
        "MAE_ms": mae,
        "RMSE_ms": rmse,
        "R2": r2,
    }


# ============================================================
# 4. PARSE CLUSTER STATE
# ============================================================

cluster_features = df[
    "pre_scale_all_nodes_json"
].apply(parse_cluster_state)

cluster_features = pd.DataFrame(
    cluster_features.tolist(),
    index=df.index,
)

print()
print("Cluster features generated.")


# ============================================================
# 5. BASE NODE FEATURES
# ============================================================

X = pd.DataFrame(
    {
        # ----------------------------------------------------
        # Selected node state
        # ----------------------------------------------------
        "node_cpu_pct": df[
            "pre_scale_node_cpu_pct"
        ].apply(parse_percent),

        "node_memory_pct": df[
            "pre_scale_node_memory_pct"
        ].apply(parse_percent),

        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------
        "cache_warm": (
            df["image_cache_state"]
            .astype(str)
            .str.lower()
            .eq("warm")
            .astype(int)
        ),
    },
    index=df.index,
)


# ============================================================
# 6. ADD CLUSTER FEATURES
# ============================================================

X = pd.concat(
    [
        X,
        cluster_features,
    ],
    axis=1,
)


# ============================================================
# 7. NODE VS CLUSTER FEATURES
# ============================================================

X["node_cpu_delta_mean"] = (
    X["node_cpu_pct"]
    - X["worker_cpu_mean"]
)

X["node_memory_delta_mean"] = (
    X["node_memory_pct"]
    - X["worker_memory_mean"]
)


# ============================================================
# 8. INTERACTION FEATURES
# ============================================================

X["node_cpu_x_memory"] = (
    X["node_cpu_pct"]
    * X["node_memory_pct"]
)

X["cluster_cpu_x_memory"] = (
    X["worker_cpu_mean"]
    * X["worker_memory_mean"]
)


# ============================================================
# 9. TARGET
# ============================================================

y = pd.to_numeric(
    df["request_to_ready_ms"],
    errors="coerce",
)


# ============================================================
# 10. FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [
    "node_cpu_pct",
    "node_memory_pct",
    "cache_warm",
    "worker_cpu_mean",
    "worker_cpu_max",
    "worker_cpu_std",
    "worker_memory_mean",
    "worker_memory_max",
    "worker_memory_std",
    "node_cpu_delta_mean",
    "node_memory_delta_mean",
    "node_cpu_x_memory",
    "cluster_cpu_x_memory",
]

X = X[FEATURE_COLUMNS]


# ============================================================
# 11. DATA QUALITY CHECK
# ============================================================

print()
print("Features:")
for feature in FEATURE_COLUMNS:
    print(f" - {feature}")

print()
print("Feature matrix shape:")
print(X.shape)

print()
print("Missing feature values:")
print(X.isna().sum())

print()

# IMPORTANT:
# Do NOT silently impute values.
# The raw 110-row data was already audited.
#
# If this assertion fails, it means there is a feature
# parsing problem that must be fixed rather than hidden.

if X.isna().any().any():
    print()
    print("ERROR: Feature matrix contains NaN values.")
    print()
    print(
        "Rows containing NaN:"
    )

    bad_rows = X[
        X.isna().any(axis=1)
    ]

    diagnostic_columns = [
        "source_condition",
        "trial",
        "pod_name",
        "node_name",
    ]

    print(
        df.loc[
            bad_rows.index,
            diagnostic_columns,
        ].to_string(index=True)
    )

    raise ValueError(
        "Stopping because feature extraction produced NaN."
    )

if y.isna().any():
    raise ValueError(
        "Target request_to_ready_ms contains NaN."
    )


# ============================================================
# 12. ANALYSIS GROUP
# ============================================================

groups = (
    df["source_condition"]
    .astype(str)
)


# ============================================================
# 13. MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
}


# ============================================================
# 14. RANDOM 5-FOLD
# ============================================================

print()
print("=" * 90)
print("RANDOM 5-FOLD")
print("=" * 90)

random_cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

random_splits = list(
    random_cv.split(X, y)
)

results = []

baseline_result = evaluate_mean_baseline(
    X,
    y,
    random_splits,
    "Random_5Fold",
)

results.append(
    baseline_result
)

print(
    f"Mean Baseline"
    f"          | MAE="
    f"{baseline_result['MAE_ms']:7.2f} ms"
    f" | RMSE="
    f"{baseline_result['RMSE_ms']:7.2f} ms"
    f" | R2="
    f"{baseline_result['R2']:7.4f}"
)

for name, model in models.items():

    result = evaluate_splits(
        model,
        X,
        y,
        random_splits,
        "Random_5Fold",
    )

    result["model"] = name

    results.append(
        result
    )

    print(
        f"{name:22s}"
        f"| MAE="
        f"{result['MAE_ms']:7.2f} ms"
        f" | RMSE="
        f"{result['RMSE_ms']:7.2f} ms"
        f" | R2="
        f"{result['R2']:7.4f}"
    )


# ============================================================
# 15. STRATIFIED 5-FOLD
# ============================================================

print()
print("=" * 90)
print("STRATIFIED 5-FOLD")
print("=" * 90)

# Create 5 latency bins.
target_bins = pd.qcut(
    y,
    q=N_SPLITS,
    labels=False,
    duplicates="drop",
)

stratified_cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

stratified_splits = list(
    stratified_cv.split(
        X,
        target_bins,
    )
)

baseline_result = evaluate_mean_baseline(
    X,
    y,
    stratified_splits,
    "Stratified_5Fold",
)

results.append(
    baseline_result
)

print(
    f"Mean Baseline"
    f"          | MAE="
    f"{baseline_result['MAE_ms']:7.2f} ms"
    f" | RMSE="
    f"{baseline_result['RMSE_ms']:7.2f} ms"
    f" | R2="
    f"{baseline_result['R2']:7.4f}"
)

for name, model in models.items():

    result = evaluate_splits(
        model,
        X,
        y,
        stratified_splits,
        "Stratified_5Fold",
    )

    result["model"] = name

    results.append(
        result
    )

    print(
        f"{name:22s}"
        f"| MAE="
        f"{result['MAE_ms']:7.2f} ms"
        f" | RMSE="
        f"{result['RMSE_ms']:7.2f} ms"
        f" | R2="
        f"{result['R2']:7.4f}"
    )


# ============================================================
# 16. CONDITION HOLDOUT / LOCO
# ============================================================

print()
print("=" * 90)
print("CONDITION HOLDOUT / LEAVE-ONE-CONDITION-OUT")
print("=" * 90)

logo = LeaveOneGroupOut()

holdout_splits = list(
    logo.split(
        X,
        y,
        groups=groups,
    )
)

baseline_result = evaluate_mean_baseline(
    X,
    y,
    holdout_splits,
    "Condition_Holdout",
)

results.append(
    baseline_result
)

print(
    f"Mean Baseline"
    f"          | MAE="
    f"{baseline_result['MAE_ms']:7.2f} ms"
    f" | RMSE="
    f"{baseline_result['RMSE_ms']:7.2f} ms"
    f" | R2="
    f"{baseline_result['R2']:7.4f}"
)

for name, model in models.items():

    result = evaluate_splits(
        model,
        X,
        y,
        holdout_splits,
        "Condition_Holdout",
    )

    result["model"] = name

    results.append(
        result
    )

    print(
        f"{name:22s}"
        f"| MAE="
        f"{result['MAE_ms']:7.2f} ms"
        f" | RMSE="
        f"{result['RMSE_ms']:7.2f} ms"
        f" | R2="
        f"{result['R2']:7.4f}"
    )


# ============================================================
# 17. SAVE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_OUTPUT,
    index=False,
)


# ============================================================
# 18. FULL-DATA GRADIENT BOOSTING FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 90)
print("MODEL V2 GRADIENT BOOSTING FEATURE IMPORTANCE")
print("=" * 90)

final_gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE,
)

final_gb.fit(
    X,
    y,
)

importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": final_gb.feature_importances_,
    }
)

importance = importance.sort_values(
    "importance",
    ascending=False,
)

print(
    importance.to_string(
        index=False
    )
)

importance.to_csv(
    IMPORTANCE_OUTPUT,
    index=False,
)


# ============================================================
# 19. SAVE FEATURE DATASET
# ============================================================

feature_output = X.copy()

feature_output["request_to_ready_ms"] = y.values

feature_output["analysis_group"] = (
    groups.values
)

feature_output.to_csv(
    FEATURE_OUTPUT,
    index=False,
)


# ============================================================
# 20. CONDITION SUMMARY
# ============================================================

print()
print("=" * 90)
print("CONDITION SUMMARY")
print("=" * 90)

summary = (
    df.groupby("source_condition")[
        "request_to_ready_ms"
    ]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )
    .sort_index()
)

print(
    summary.to_string()
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print()
print("=" * 90)
print("MODEL V2 110-ROW COMPLETE")
print("=" * 90)

print()
print("Saved:")
print(f" - {RESULTS_OUTPUT}")
print(f" - {IMPORTANCE_OUTPUT}")
print(f" - {FEATURE_OUTPUT}")

print()
print(
    "Input dataset remains unchanged:"
)
print(f" - {FILE}")

print()
