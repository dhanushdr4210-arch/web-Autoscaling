import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


V2_FILE = "v3_master_150_v2_features.csv"
V3_FILE = "v3_master_150_v3_spatial_features.csv"

TARGET = "request_to_ready_ms"

BASE_FEATURES = [
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

CPU_SPATIAL = [
    "worker_cpu_mean_spatial",
    "worker_cpu_max_spatial",
    "worker_cpu_std_spatial",
    "cluster_worker_cpu_total_spatial",
    "cpu_spatial_imbalance",
]

MEMORY_SPATIAL = [
    "worker_memory_mean_spatial",
    "worker_memory_max_spatial",
    "worker_memory_std_spatial",
]

IMBALANCE_ONLY = [
    "cpu_spatial_imbalance"
]

FEATURE_SETS = {
    "V2_Baseline": BASE_FEATURES,
    "V2_plus_CPU_Spatial": BASE_FEATURES + CPU_SPATIAL,
    "V2_plus_Memory_Spatial": BASE_FEATURES + MEMORY_SPATIAL,
    "V2_plus_Imbalance": BASE_FEATURES + IMBALANCE_ONLY,
}


def prepare_X(df, features):
    X = df[features].copy()

    if "image_cache_state" in X.columns:
        X["image_cache_state"] = (
            X["image_cache_state"]
            .astype(str)
            .str.lower()
            .map({"cold": 0, "warm": 1})
        )

    if X.isna().any().any():
        raise ValueError(
            "Missing values found after feature preparation."
        )

    return X


def make_model(name):
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt"
        )

    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )


def evaluate(df, features, splits, model_name):

    X = prepare_X(df, features)
    y = df[TARGET].values

    maes = []
    rmses = []
    r2s = []

    for train_idx, test_idx in splits:

        model = make_model(model_name)

        model.fit(
            X.iloc[train_idx],
            y[train_idx]
        )

        pred = model.predict(
            X.iloc[test_idx]
        )

        maes.append(
            mean_absolute_error(
                y[test_idx],
                pred
            )
        )

        rmses.append(
            np.sqrt(
                mean_squared_error(
                    y[test_idx],
                    pred
                )
            )
        )

        r2s.append(
            r2_score(
                y[test_idx],
                pred
            )
        )

    return {
        "MAE": np.mean(maes),
        "RMSE": np.mean(rmses),
        "R2": np.mean(r2s)
    }


# ================================================================
# LOAD
# ================================================================

print("=" * 78)
print("STARTSCALE SPATIAL FEATURE-LEVEL ABLATION")
print("=" * 78)

v2 = pd.read_csv(V2_FILE)
v3 = pd.read_csv(V3_FILE)

print("\nV2:", v2.shape)
print("V3:", v3.shape)

if not np.allclose(
    v2[TARGET].values,
    v3[TARGET].values
):
    raise ValueError("V2 and V3 target values are not aligned.")

print("Target alignment: PASS")


# ================================================================
# RANDOM 5-FOLD
# ================================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

random_splits = list(kf.split(v3))


# ================================================================
# STRATIFIED
# ================================================================

labels = v3["source_condition"].astype(str)

counts = labels.value_counts()

if counts.min() < 5:
    labels = pd.qcut(
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
    skf.split(v3, labels)
)


# ================================================================
# LOCO
# ================================================================

groups = v3["source_condition"].astype(str)

gkf = GroupKFold(
    n_splits=min(5, groups.nunique())
)

loco_splits = list(
    gkf.split(
        v3,
        groups=groups
    )
)


# ================================================================
# RUN ALL ABLATIONS
# ================================================================

results = []

for validation_name, splits in [
    ("Random 5-Fold", random_splits),
    ("Stratified 5-Fold", stratified_splits),
    ("LOCO", loco_splits),
]:

    print("\n" + "=" * 78)
    print(validation_name)
    print("=" * 78)

    for feature_set_name, features in FEATURE_SETS.items():

        df = v3

        for model_name in [
            "Random Forest",
            "Gradient Boosting"
        ]:

            metrics = evaluate(
                df,
                features,
                splits,
                model_name
            )

            results.append({
                "validation": validation_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            })

            print(
                f"{feature_set_name:25s} | "
                f"{model_name:18s} | "
                f"MAE={metrics['MAE']:8.2f} | "
                f"RMSE={metrics['RMSE']:8.2f} | "
                f"R2={metrics['R2']:+.4f}"
            )


# ================================================================
# RESULTS
# ================================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "spatial_feature_level_ablation_results.csv",
    index=False
)

print("\n" + "=" * 78)
print("SAVED")
print("=" * 78)

print(
    "spatial_feature_level_ablation_results.csv"
)


# ================================================================
# COMPARE AGAINST BASELINE
# ================================================================

print("\n" + "=" * 78)
print("IMPROVEMENT OVER V2 BASELINE")
print("=" * 78)

for validation in results_df["validation"].unique():

    for model in results_df["model"].unique():

        baseline = results_df[
            (results_df["validation"] == validation) &
            (results_df["model"] == model) &
            (results_df["feature_set"] == "V2_Baseline")
        ].iloc[0]

        print(
            f"\n{validation} | {model}"
        )

        for feature_set in [
            "V2_plus_CPU_Spatial",
            "V2_plus_Memory_Spatial",
            "V2_plus_Imbalance"
        ]:

            row = results_df[
                (results_df["validation"] == validation) &
                (results_df["model"] == model) &
                (results_df["feature_set"] == feature_set)
            ].iloc[0]

            mae_improvement = (
                (baseline["MAE"] - row["MAE"])
                / baseline["MAE"]
                * 100
            )

            rmse_improvement = (
                (baseline["RMSE"] - row["RMSE"])
                / baseline["RMSE"]
                * 100
            )

            r2_change = (
                row["R2"] - baseline["R2"]
            )

            print(
                f"  {feature_set:25s} | "
                f"MAE {mae_improvement:+7.2f}% | "
                f"RMSE {rmse_improvement:+7.2f}% | "
                f"R2 {r2_change:+.4f}"
            )


print("\n" + "=" * 78)
print("DONE")
print("=" * 78)