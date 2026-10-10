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


# B: explicitly include aggregate worker CPU
AGGREGATE_FEATURES = BASE_FEATURES + [
    "cluster_worker_cpu_total_spatial"
]


# C: aggregate CPU + spatial imbalance
SPATIAL_FEATURES = BASE_FEATURES + [
    "cluster_worker_cpu_total_spatial",
    "cpu_spatial_imbalance"
]


FEATURE_SETS = {
    "A_V2_Baseline": BASE_FEATURES,
    "B_Aggregate_CPU": AGGREGATE_FEATURES,
    "C_Aggregate_plus_Imbalance": SPATIAL_FEATURES,
}


def prepare_X(df, features):

    X = df[features].copy()

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
            "Missing values detected after encoding."
        )

    return X


def make_model(model_name):

    if model_name == "Random Forest":

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
# LOAD DATA
# ================================================================

print("=" * 78)
print("STARTSCALE: AGGREGATE CPU vs SPATIAL IMBALANCE")
print("=" * 78)

v2 = pd.read_csv(V2_FILE)
v3 = pd.read_csv(V3_FILE)

print("\nV2 shape:", v2.shape)
print("V3 shape:", v3.shape)


# ================================================================
# ALIGNMENT CHECK
# ================================================================

if not np.allclose(
    v2[TARGET].values,
    v3[TARGET].values
):
    raise ValueError(
        "Target values are not aligned between V2 and V3."
    )

print("Target alignment: PASS")


# ================================================================
# RANDOM 5-FOLD
# ================================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

random_splits = list(
    kf.split(v3)
)


# ================================================================
# STRATIFIED 5-FOLD
# ================================================================

labels = v3["source_condition"].astype(str)

if labels.value_counts().min() < 5:

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
# RUN
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

        for model_name in [
            "Random Forest",
            "Gradient Boosting"
        ]:

            metrics = evaluate(
                v3,
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
                "R2": metrics["R2"]
            })

            print(
                f"{feature_set_name:28s} | "
                f"{model_name:18s} | "
                f"MAE={metrics['MAE']:8.2f} | "
                f"RMSE={metrics['RMSE']:8.2f} | "
                f"R2={metrics['R2']:+.4f}"
            )


# ================================================================
# DATAFRAME
# ================================================================

results_df = pd.DataFrame(results)


# ================================================================
# SAVE
# ================================================================

results_df.to_csv(
    "aggregate_vs_spatial_results.csv",
    index=False
)


# ================================================================
# B vs C ANALYSIS
# ================================================================

print("\n" + "=" * 78)
print("KEY TEST: DOES SPATIAL IMBALANCE ADD INFORMATION?")
print("=" * 78)

for validation in results_df["validation"].unique():

    for model in results_df["model"].unique():

        baseline = results_df[
            (results_df["validation"] == validation) &
            (results_df["model"] == model) &
            (results_df["feature_set"] == "B_Aggregate_CPU")
        ].iloc[0]

        spatial = results_df[
            (results_df["validation"] == validation) &
            (results_df["model"] == model) &
            (results_df["feature_set"] == "C_Aggregate_plus_Imbalance")
        ].iloc[0]

        mae_improvement = (
            (baseline["MAE"] - spatial["MAE"])
            / baseline["MAE"]
            * 100
        )

        rmse_improvement = (
            (baseline["RMSE"] - spatial["RMSE"])
            / baseline["RMSE"]
            * 100
        )

        r2_change = (
            spatial["R2"] - baseline["R2"]
        )

        print(
            f"\n{validation} | {model}"
        )

        print(
            f"  MAE improvement  : "
            f"{mae_improvement:+.2f}%"
        )

        print(
            f"  RMSE improvement : "
            f"{rmse_improvement:+.2f}%"
        )

        print(
            f"  R2 change        : "
            f"{r2_change:+.4f}"
        )


print("\n" + "=" * 78)
print("Saved: aggregate_vs_spatial_results.csv")
print("=" * 78)