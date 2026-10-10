import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================================================================
# STARTSCALE V2 vs V3 SPATIAL FEATURE ABLATION
# ================================================================

V2_FILE = "v3_master_150_v2_features.csv"
V3_FILE = "v3_master_150_v3_spatial_features.csv"

TARGET = "request_to_ready_ms"

SPATIAL_FEATURES = [
    "worker_cpu_mean_spatial",
    "worker_cpu_max_spatial",
    "worker_cpu_std_spatial",
    "worker_memory_mean_spatial",
    "worker_memory_max_spatial",
    "worker_memory_std_spatial",
    "cluster_worker_cpu_total_spatial",
    "cpu_spatial_imbalance",
]

V2_FEATURES = [
    "node_cpu_pct_v2",
    "node_memory_pct_v2",
    "worker_cpu_mean",
    "worker_cpu_max",
    "worker_cpu_std",
    "worker_memory_mean",
    "worker_memory_max",
    "worker_memory_std",
    "node_cpu_x_memory",
    "cluster_cpu_x_memory",
    "image_cache_state",
]


def make_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt"
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

def evaluate_with_splits(df, features, splits, model_name):
    X = df[features].copy()
    y = df[TARGET].copy()

    # Encode cache state for machine-learning models
    if "image_cache_state" in X.columns:
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
        raise ValueError(
            "Feature encoding produced missing values. "
            "Check image_cache_state values."
        )
    results = []

    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):

        model = make_models()[model_name]

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, pred)

        results.append({
            "fold": fold_no,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"]
        })

    return pd.DataFrame(results)


def print_result(label, result_df):

    print(f"\n{label}")
    print("-" * 70)

    print(
        f"MAE  : {result_df['MAE'].mean():.2f} "
        f"+/- {result_df['MAE'].std():.2f}"
    )

    print(
        f"RMSE : {result_df['RMSE'].mean():.2f} "
        f"+/- {result_df['RMSE'].std():.2f}"
    )

    print(
        f"R2   : {result_df['R2'].mean():.4f} "
        f"+/- {result_df['R2'].std():.4f}"
    )


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 78)
print("STARTSCALE V2 vs V3 SPATIAL FEATURE ABLATION")
print("=" * 78)

v2 = pd.read_csv(V2_FILE)
v3 = pd.read_csv(V3_FILE)

print("\nV2 shape:", v2.shape)
print("V3 shape:", v3.shape)

if len(v2) != len(v3):
    raise ValueError("V2 and V3 row counts do not match.")

if not np.allclose(
    v2[TARGET].values,
    v3[TARGET].values,
    equal_nan=True
):
    raise ValueError(
        "Target values differ between V2 and V3. "
        "The datasets are not aligned."
    )

print("Target alignment: PASS")


# ================================================================
# CHECK FEATURES
# ================================================================

missing_v2 = [f for f in V2_FEATURES if f not in v2.columns]
missing_v3 = [f for f in V2_FEATURES + SPATIAL_FEATURES if f not in v3.columns]

if missing_v2:
    raise ValueError(f"Missing V2 features: {missing_v2}")

if missing_v3:
    raise ValueError(f"Missing V3 features: {missing_v3}")

print("Feature availability: PASS")


# ================================================================
# BUILD FEATURE SETS
# ================================================================

V3_FEATURES = V2_FEATURES + SPATIAL_FEATURES

print("\nFeature counts:")
print("V2 features:", len(V2_FEATURES))
print("V3 features:", len(V3_FEATURES))
print("Added spatial features:", len(SPATIAL_FEATURES))


# ================================================================
# REMOVE ROWS WITH MISSING VALUES
# ================================================================

required_columns = [TARGET] + V3_FEATURES

valid_mask = v3[required_columns].notna().all(axis=1)

if not valid_mask.all():
    removed = (~valid_mask).sum()
    print(f"\nWARNING: removing {removed} rows with missing values.")
    v2 = v2.loc[valid_mask].reset_index(drop=True)
    v3 = v3.loc[valid_mask].reset_index(drop=True)

else:
    print("Missing-value check: PASS")


# ================================================================
# SAME RANDOM 5-FOLD SPLITS FOR BOTH MODELS
# ================================================================

print("\n" + "=" * 78)
print("1. RANDOM 5-FOLD CROSS-VALIDATION")
print("=" * 78)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

random_splits = list(
    kf.split(v3)
)


# ================================================================
# RANDOM CV
# ================================================================

summary_rows = []

for model_name in ["Random Forest", "Gradient Boosting"]:

    v2_result = evaluate_with_splits(
        v2,
        V2_FEATURES,
        random_splits,
        model_name
    )

    v3_result = evaluate_with_splits(
        v3,
        V3_FEATURES,
        random_splits,
        model_name
    )

    print_result(
        f"V2 | {model_name}",
        v2_result
    )

    print_result(
        f"V3 Spatial | {model_name}",
        v3_result
    )

    v2_mae = v2_result["MAE"].mean()
    v3_mae = v3_result["MAE"].mean()

    v2_rmse = v2_result["RMSE"].mean()
    v3_rmse = v3_result["RMSE"].mean()

    v2_r2 = v2_result["R2"].mean()
    v3_r2 = v3_result["R2"].mean()

    summary_rows.append({
        "validation": "Random 5-Fold",
        "model": model_name,
        "V2_MAE": v2_mae,
        "V3_MAE": v3_mae,
        "MAE_change": v3_mae - v2_mae,
        "MAE_improvement_pct": ((v2_mae - v3_mae) / v2_mae) * 100,
        "V2_RMSE": v2_rmse,
        "V3_RMSE": v3_rmse,
        "RMSE_change": v3_rmse - v2_rmse,
        "RMSE_improvement_pct": ((v2_rmse - v3_rmse) / v2_rmse) * 100,
        "V2_R2": v2_r2,
        "V3_R2": v3_r2,
        "R2_change": v3_r2 - v2_r2
    })


# ================================================================
# STRATIFIED CV
# ================================================================

print("\n" + "=" * 78)
print("2. STRATIFIED 5-FOLD CROSS-VALIDATION")
print("=" * 78)

# Use experiment/source condition as the stratification label.
# This keeps the broad experiment composition similar across folds.

stratify_column = None

for candidate in [
    "source_condition",
    "experiment_original",
    "experiment"
]:
    if candidate in v3.columns:
        stratify_column = candidate
        break

if stratify_column is None:
    raise ValueError(
        "No suitable experiment/source-condition column found "
        "for stratified CV."
    )

strat_labels = v3[stratify_column].astype(str)

# StratifiedKFold requires every class to have at least 5 samples.
counts = strat_labels.value_counts()

if counts.min() < 5:

    print(
        f"\nWARNING: {stratify_column} contains a class with "
        f"fewer than 5 samples."
    )

    print(
        "Falling back to stratification by latency quartile."
    )

    strat_labels = pd.qcut(
        v3[TARGET],
        q=5,
        labels=False,
        duplicates="drop"
    )

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

stratified_splits = list(
    skf.split(v3, strat_labels)
)

for model_name in ["Random Forest", "Gradient Boosting"]:

    v2_result = evaluate_with_splits(
        v2,
        V2_FEATURES,
        stratified_splits,
        model_name
    )

    v3_result = evaluate_with_splits(
        v3,
        V3_FEATURES,
        stratified_splits,
        model_name
    )

    print_result(
        f"V2 | {model_name}",
        v2_result
    )

    print_result(
        f"V3 Spatial | {model_name}",
        v3_result
    )

    v2_mae = v2_result["MAE"].mean()
    v3_mae = v3_result["MAE"].mean()

    v2_rmse = v2_result["RMSE"].mean()
    v3_rmse = v3_result["RMSE"].mean()

    v2_r2 = v2_result["R2"].mean()
    v3_r2 = v3_result["R2"].mean()

    summary_rows.append({
        "validation": "Stratified 5-Fold",
        "model": model_name,
        "V2_MAE": v2_mae,
        "V3_MAE": v3_mae,
        "MAE_change": v3_mae - v2_mae,
        "MAE_improvement_pct": ((v2_mae - v3_mae) / v2_mae) * 100,
        "V2_RMSE": v2_rmse,
        "V3_RMSE": v3_rmse,
        "RMSE_change": v3_rmse - v2_rmse,
        "RMSE_improvement_pct": ((v2_rmse - v3_rmse) / v2_rmse) * 100,
        "V2_R2": v2_r2,
        "V3_R2": v3_r2,
        "R2_change": v3_r2 - v2_r2
    })


# ================================================================
# LOCO
# ================================================================

print("\n" + "=" * 78)
print("3. LEAVE-ONE-CONDITION-OUT (LOCO)")
print("=" * 78)

group_column = None

for candidate in [
    "source_condition",
    "experiment_original",
    "experiment"
]:
    if candidate in v3.columns:
        group_column = candidate
        break

if group_column is None:
    raise ValueError(
        "No experiment/source-condition column available for LOCO."
    )

groups = v3[group_column].astype(str)

unique_groups = groups.unique()

print("\nLOCO grouping column:", group_column)
print("Number of groups:", len(unique_groups))

group_counts = groups.value_counts()

print("\nGroup sizes:")
print(group_counts.to_string())


if len(unique_groups) < 2:
    raise ValueError("LOCO requires at least 2 groups.")

n_loco_splits = min(5, len(unique_groups))

gkf = GroupKFold(
    n_splits=n_loco_splits
)

loco_splits = list(
    gkf.split(v3, groups=groups)
)

for model_name in ["Random Forest", "Gradient Boosting"]:

    v2_result = evaluate_with_splits(
        v2,
        V2_FEATURES,
        loco_splits,
        model_name
    )

    v3_result = evaluate_with_splits(
        v3,
        V3_FEATURES,
        loco_splits,
        model_name
    )

    print_result(
        f"V2 | {model_name}",
        v2_result
    )

    print_result(
        f"V3 Spatial | {model_name}",
        v3_result
    )

    v2_mae = v2_result["MAE"].mean()
    v3_mae = v3_result["MAE"].mean()

    v2_rmse = v2_result["RMSE"].mean()
    v3_rmse = v3_result["RMSE"].mean()

    v2_r2 = v2_result["R2"].mean()
    v3_r2 = v3_result["R2"].mean()

    summary_rows.append({
        "validation": "LOCO",
        "model": model_name,
        "V2_MAE": v2_mae,
        "V3_MAE": v3_mae,
        "MAE_change": v3_mae - v2_mae,
        "MAE_improvement_pct": ((v2_mae - v3_mae) / v2_mae) * 100,
        "V2_RMSE": v2_rmse,
        "V3_RMSE": v3_rmse,
        "RMSE_change": v3_rmse - v2_rmse,
        "RMSE_improvement_pct": ((v2_rmse - v3_rmse) / v2_rmse) * 100,
        "V2_R2": v2_r2,
        "V3_R2": v3_r2,
        "R2_change": v3_r2 - v2_r2
    })


# ================================================================
# FINAL COMPARISON TABLE
# ================================================================

summary = pd.DataFrame(summary_rows)

print("\n" + "=" * 78)
print("FINAL V2 vs V3 SPATIAL ABLATION")
print("=" * 78)

display_columns = [
    "validation",
    "model",
    "V2_MAE",
    "V3_MAE",
    "MAE_improvement_pct",
    "V2_RMSE",
    "V3_RMSE",
    "RMSE_improvement_pct",
    "V2_R2",
    "V3_R2",
    "R2_change"
]

print(
    summary[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ================================================================
# SAVE RESULTS
# ================================================================

summary.to_csv(
    "v3_spatial_ablation_results.csv",
    index=False
)

print("\nSaved:")
print(" - v3_spatial_ablation_results.csv")


# ================================================================
# INTERPRETATION
# ================================================================

print("\n" + "=" * 78)
print("INTERPRETATION")
print("=" * 78)

for validation in summary["validation"].unique():

    subset = summary[
        summary["validation"] == validation
    ]

    for _, row in subset.iterrows():

        model = row["model"]
        improvement = row["MAE_improvement_pct"]
        r2_change = row["R2_change"]

        print(
            f"{validation:22s} | "
            f"{model:18s} | "
            f"MAE improvement = {improvement:7.2f}% | "
            f"R2 change = {r2_change:+.4f}"
        )

print("\n")
print("IMPORTANT:")
print("A positive MAE improvement means V3 Spatial is better.")
print("A negative MAE improvement means adding spatial features hurt MAE.")
print("R2 change > 0 means V3 explains more variance than V2.")
print("LOCO is the most important test for cross-condition generalization.")
print("=" * 78)