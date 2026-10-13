import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = "v3_model_v2_features_110.csv"
OUTPUT_FILE = "v3_holdout_diagnosis_110.csv"

RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("STARTSCALE V3 - 110 ROW CONDITION HOLDOUT DIAGNOSIS")
print("=" * 90)

df = pd.read_csv(FEATURE_FILE)

FEATURES = [
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

TARGET = "request_to_ready_ms"
GROUP = "analysis_group"


X = df[FEATURES].copy()
y = df[TARGET].astype(float)
groups = df[GROUP].astype(str)

print()
print(f"Rows    : {len(df)}")
print(f"Features: {len(FEATURES)}")
print(f"Groups  : {groups.nunique()}")


# ============================================================
# MODELS
# ============================================================

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE,
)


# ============================================================
# LEAVE-ONE-CONDITION-OUT
# ============================================================

logo = LeaveOneGroupOut()

results = []

for train_idx, test_idx in logo.split(
    X,
    y,
    groups=groups,
):

    train_X = X.iloc[train_idx]
    train_y = y.iloc[train_idx]

    test_X = X.iloc[test_idx]
    test_y = y.iloc[test_idx]

    held_out_group = groups.iloc[
        test_idx[0]
    ]

    # --------------------------------------------------------
    # Mean baseline
    # --------------------------------------------------------
    baseline_value = train_y.mean()

    baseline_pred = np.full(
        len(test_y),
        baseline_value,
    )

    baseline_mae = mean_absolute_error(
        test_y,
        baseline_pred,
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------
    rf.fit(
        train_X,
        train_y,
    )

    rf_pred = rf.predict(
        test_X
    )

    rf_mae = mean_absolute_error(
        test_y,
        rf_pred,
    )

    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------
    gb.fit(
        train_X,
        train_y,
    )

    gb_pred = gb.predict(
        test_X
    )

    gb_mae = mean_absolute_error(
        test_y,
        gb_pred,
    )

    # --------------------------------------------------------
    # Additional diagnostics
    # --------------------------------------------------------
    actual_mean = test_y.mean()

    rf_mean_pred = rf_pred.mean()
    gb_mean_pred = gb_pred.mean()

    rf_bias = rf_mean_pred - actual_mean
    gb_bias = gb_mean_pred - actual_mean

    results.append(
        {
            "condition": held_out_group,
            "n_test": len(test_y),

            "actual_mean_ms":
                actual_mean,

            "baseline_mae_ms":
                baseline_mae,

            "rf_mae_ms":
                rf_mae,

            "gb_mae_ms":
                gb_mae,

            "rf_minus_baseline_ms":
                rf_mae - baseline_mae,

            "gb_minus_baseline_ms":
                gb_mae - baseline_mae,

            "rf_mean_prediction_ms":
                rf_mean_pred,

            "gb_mean_prediction_ms":
                gb_mean_pred,

            "rf_bias_ms":
                rf_bias,

            "gb_bias_ms":
                gb_bias,
        }
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "gb_mae_ms",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 90)
print("CONDITION-BY-CONDITION HOLDOUT RESULTS")
print("=" * 90)

display_columns = [
    "condition",
    "n_test",
    "actual_mean_ms",
    "baseline_mae_ms",
    "rf_mae_ms",
    "gb_mae_ms",
    "rf_minus_baseline_ms",
    "gb_minus_baseline_ms",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ============================================================
# WORST CONDITIONS
# ============================================================

print()
print("=" * 90)
print("WORST GRADIENT BOOSTING HOLDOUT CONDITIONS")
print("=" * 90)

worst_gb = results_df.head(5)

print(
    worst_gb[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ============================================================
# WORST RF CONDITIONS
# ============================================================

print()
print("=" * 90)
print("WORST RANDOM FOREST HOLDOUT CONDITIONS")
print("=" * 90)

worst_rf = (
    results_df
    .sort_values(
        "rf_mae_ms",
        ascending=False,
    )
    .head(5)
)

print(
    worst_rf[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 90)
print("DIAGNOSIS COMPLETE")
print("=" * 90)

print(
    f"Saved: {OUTPUT_FILE}"
)