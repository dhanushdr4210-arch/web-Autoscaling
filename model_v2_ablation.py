import pandas as pd
import numpy as np
import json

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    cross_val_predict
)

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv("v3_master_100_raw.csv")


# ============================================================
# ANALYSIS GROUP
# ============================================================

df["analysis_group"] = df["source_condition"]

cpu_mask = df["source_condition"] == "CPU_SWEEP"

df.loc[cpu_mask, "analysis_group"] = (
    df.loc[cpu_mask, "experiment_original"]
)


# ============================================================
# TARGET
# ============================================================

y = df["request_to_ready_ms"].astype(float)


# ============================================================
# PARSE CLUSTER JSON
# ============================================================

def parse_cluster_state(raw_json):

    data = json.loads(raw_json)

    workers = {
        name: values
        for name, values in data.items()
        if name.startswith("autoscaling-worker")
    }

    cpu = []
    memory = []

    for values in workers.values():

        cpu.append(
            float(
                str(values["cpu_pct"])
                .replace("%", "")
            )
        )

        memory.append(
            float(
                str(values["memory_pct"])
                .replace("%", "")
            )
        )

    return {
        "worker_cpu_mean": np.mean(cpu),
        "worker_cpu_max": np.max(cpu),
        "worker_cpu_std": np.std(cpu),

        "worker_memory_mean": np.mean(memory),
        "worker_memory_max": np.max(memory),
        "worker_memory_std": np.std(memory),
    }


cluster = pd.DataFrame(
    df["pre_scale_all_nodes_json"]
    .apply(parse_cluster_state)
    .tolist(),
    index=df.index
)


# ============================================================
# BASE FEATURES
# ============================================================

base = pd.DataFrame({
    "node_cpu_pct":
        pd.to_numeric(
            df["pre_scale_node_cpu_pct"],
            errors="coerce"
        ),

    "node_memory_pct":
        pd.to_numeric(
            df["pre_scale_node_memory_pct"],
            errors="coerce"
        ),

    "cache_warm":
        df["image_cache_state"]
        .astype(str)
        .str.lower()
        .eq("warm")
        .astype(int),
})


# ============================================================
# V1 FEATURES
# ============================================================

v1 = base.copy()

v1["node_cpu_x_memory"] = (
    v1["node_cpu_pct"] *
    v1["node_memory_pct"]
)


# ============================================================
# V2 CLUSTER FEATURES
# ============================================================

cluster_features = cluster.copy()

cluster_features["node_cpu_delta_mean"] = (
    base["node_cpu_pct"] -
    cluster["worker_cpu_mean"]
)

cluster_features["node_memory_delta_mean"] = (
    base["node_memory_pct"] -
    cluster["worker_memory_mean"]
)

cluster_features["node_cpu_x_memory"] = (
    base["node_cpu_pct"] *
    base["node_memory_pct"]
)

cluster_features["cluster_cpu_x_memory"] = (
    cluster["worker_cpu_mean"] *
    cluster["worker_memory_mean"]
)


v2 = pd.concat(
    [base, cluster_features],
    axis=1
)


# ============================================================
# ABLATION 1
# V2 WITHOUT worker_cpu_std
# ============================================================

v2_no_cpu_std = v2.drop(
    columns=["worker_cpu_std"]
)


# ============================================================
# ABLATION 2
# V2 WITHOUT CLUSTER FEATURES
# Keep only node + cache + local interaction
# ============================================================

v2_no_cluster = base.copy()

v2_no_cluster["node_cpu_x_memory"] = (
    v2_no_cluster["node_cpu_pct"] *
    v2_no_cluster["node_memory_pct"]
)


# ============================================================
# DATASET COLLECTION
# ============================================================

feature_sets = {
    "V1_Core": v1,
    "V2_Full": v2,
    "V2_NoWorkerCPUStd": v2_no_cpu_std,
    "V2_NoCluster": v2_no_cluster,
}


# ============================================================
# MODELS
# ============================================================

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=3,
        random_state=42
    ),
}


# ============================================================
# VALIDATION
# ============================================================

random_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

stratified_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

group_cv = GroupKFold(
    n_splits=df["analysis_group"].nunique()
)

groups = df["analysis_group"]


# ============================================================
# RUN
# ============================================================

results = []

print("=" * 90)
print("STARTSCALE V2 FEATURE ABLATION")
print("=" * 90)


for feature_set_name, X in feature_sets.items():

    print("\n" + "=" * 90)
    print("FEATURE SET:", feature_set_name)
    print("FEATURES:", list(X.columns))
    print("=" * 90)

    for model_name, model in models.items():

        # ----------------------------------------------------
        # Random
        # ----------------------------------------------------

        pred = cross_val_predict(
            model,
            X,
            y,
            cv=random_cv
        )

        mae_random = mean_absolute_error(
            y,
            pred
        )

        # ----------------------------------------------------
        # Stratified
        # ----------------------------------------------------

        pred = cross_val_predict(
            model,
            X,
            y,
            cv=stratified_cv.split(
                X,
                groups
            )
        )

        mae_stratified = mean_absolute_error(
            y,
            pred
        )

        # ----------------------------------------------------
        # Condition Holdout
        # ----------------------------------------------------

        pred = cross_val_predict(
            model,
            X,
            y,
            cv=group_cv,
            groups=groups
        )

        mae_holdout = mean_absolute_error(
            y,
            pred
        )

        print(
            f"{model_name:18s} | "
            f"Random={mae_random:8.2f} | "
            f"Stratified={mae_stratified:8.2f} | "
            f"Holdout={mae_holdout:8.2f}"
        )

        results.append({
            "feature_set": feature_set_name,
            "model": model_name,
            "random_MAE_ms": mae_random,
            "stratified_MAE_ms": mae_stratified,
            "condition_holdout_MAE_ms": mae_holdout,
        })


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "v3_model_v2_ablation_results.csv",
    index=False
)


# ============================================================
# FEATURE DISTRIBUTION BY CONDITION
# ============================================================

print("\n" + "=" * 90)
print("KEY CLUSTER FEATURE DISTRIBUTION")
print("=" * 90)

distribution = (
    pd.concat(
        [
            df["analysis_group"],
            cluster
        ],
        axis=1
    )
    .groupby("analysis_group")
    .agg(
        cpu_mean=("worker_cpu_mean", "mean"),
        cpu_max=("worker_cpu_max", "mean"),
        cpu_std=("worker_cpu_std", "mean"),
        mem_mean=("worker_memory_mean", "mean"),
        mem_max=("worker_memory_max", "mean"),
        mem_std=("worker_memory_std", "mean"),
    )
    .round(3)
)

print(
    distribution.to_string()
)

distribution.to_csv(
    "v3_cluster_feature_distribution.csv"
)


print("\n" + "=" * 90)
print("ABLATION COMPLETE")
print("=" * 90)

print("Saved:")
print(" - v3_model_v2_ablation_results.csv")
print(" - v3_cluster_feature_distribution.csv")

print("=" * 90)