import pandas as pd
import numpy as np
import json

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

FILE = "v3_master_100_raw.csv"

df = pd.read_csv(FILE)


# ============================================================
# 2. ANALYSIS GROUP
# ============================================================

df["analysis_group"] = df["source_condition"]

cpu_mask = df["source_condition"] == "CPU_SWEEP"

df.loc[
    cpu_mask,
    "analysis_group"
] = df.loc[
    cpu_mask,
    "experiment_original"
]


# ============================================================
# 3. PARSE ALL-NODE CLUSTER STATE
# ============================================================

def parse_cluster_state(raw_json):

    data = json.loads(raw_json)

    # Only worker nodes represent schedulable application capacity.
    worker_nodes = {
        name: values
        for name, values in data.items()
        if name.startswith("autoscaling-worker")
    }

    cpu_values = []
    memory_values = []

    for values in worker_nodes.values():

        cpu_text = str(values["cpu_pct"]).replace("%", "")
        memory_text = str(values["memory_pct"]).replace("%", "")

        cpu_values.append(float(cpu_text))
        memory_values.append(float(memory_text))

    return {
        "worker_cpu_mean": np.mean(cpu_values),
        "worker_cpu_max": np.max(cpu_values),
        "worker_cpu_std": np.std(cpu_values),

        "worker_memory_mean": np.mean(memory_values),
        "worker_memory_max": np.max(memory_values),
        "worker_memory_std": np.std(memory_values),
    }


cluster_features = df[
    "pre_scale_all_nodes_json"
].apply(parse_cluster_state)

cluster_features = pd.DataFrame(
    cluster_features.tolist(),
    index=df.index
)


# ============================================================
# 4. BASE NODE FEATURES
# ============================================================

X = pd.DataFrame({

    # Selected node state
    "node_cpu_pct": pd.to_numeric(
        df["pre_scale_node_cpu_pct"],
        errors="coerce"
    ),

    "node_memory_pct": pd.to_numeric(
        df["pre_scale_node_memory_pct"],
        errors="coerce"
    ),

    # Cache
    "cache_warm": (
        df["image_cache_state"]
        .astype(str)
        .str.lower()
        .eq("warm")
        .astype(int)
    ),
})


# ============================================================
# 5. ADD CLUSTER FEATURES
# ============================================================

X = pd.concat(
    [
        X,
        cluster_features
    ],
    axis=1
)


# ============================================================
# 6. NODE VS CLUSTER FEATURES
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
# 7. INTERACTION FEATURES
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
# 8. TARGET
# ============================================================

y = df[
    "request_to_ready_ms"
].astype(float)


# ============================================================
# 9. GROUPS
# ============================================================

groups = df[
    "analysis_group"
]


# ============================================================
# 10. SHOW FEATURES
# ============================================================

print("=" * 90)
print("STARTSCALE V3 MODEL V2")
print("=" * 90)

print("\nFeatures:")

for feature in X.columns:
    print(" -", feature)

print("\nFeature matrix shape:")
print(X.shape)

print("\nMissing feature values:")
print(X.isna().sum())


# ============================================================
# 11. MODELS
# ============================================================

models = {

    "Mean Baseline": DummyRegressor(
        strategy="mean"
    ),

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=3,
        random_state=42
    ),
}


# ============================================================
# 12. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
    cv,
    validation_name,
    groups_arg=None
):

    if groups_arg is None:

        predictions = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            n_jobs=None
        )

    else:

        predictions = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            groups=groups_arg,
            n_jobs=None
        )

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    print(
        f"{name:22s} | "
        f"MAE={mae:8.2f} ms | "
        f"RMSE={rmse:8.2f} ms | "
        f"R2={r2:8.4f}"
    )

    return {
        "model": name,
        "validation": validation_name,
        "MAE_ms": mae,
        "RMSE_ms": rmse,
        "R2": r2,
    }


# ============================================================
# 13. VALIDATION DEFINITIONS
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
    n_splits=groups.nunique()
)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

results = []


# ------------------------------------------------------------
# RANDOM 5-FOLD
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("RANDOM 5-FOLD")
print("=" * 90)

for name, model in models.items():

    result = evaluate_model(
        name,
        model,
        random_cv,
        "Random_5Fold"
    )

    results.append(result)


# ------------------------------------------------------------
# STRATIFIED 5-FOLD
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("STRATIFIED 5-FOLD")
print("=" * 90)

for name, model in models.items():

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=stratified_cv.split(
            X,
            groups
        ),
        n_jobs=None
    )

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    print(
        f"{name:22s} | "
        f"MAE={mae:8.2f} ms | "
        f"RMSE={rmse:8.2f} ms | "
        f"R2={r2:8.4f}"
    )

    results.append({
        "model": name,
        "validation": "Stratified_5Fold",
        "MAE_ms": mae,
        "RMSE_ms": rmse,
        "R2": r2,
    })


# ------------------------------------------------------------
# CONDITION HOLDOUT
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("CONDITION-HOLDOUT")
print("=" * 90)

for name, model in models.items():

    result = evaluate_model(
        name,
        model,
        group_cv,
        "Condition_Holdout",
        groups_arg=groups
    )

    results.append(result)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "v3_model_v2_comparison.csv",
    index=False
)


# ============================================================
# 16. FULL-DATA GRADIENT BOOSTING FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 90)
print("MODEL V2 GRADIENT BOOSTING FEATURE IMPORTANCE")
print("=" * 90)

final_gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=2,
    min_samples_leaf=3,
    random_state=42
)

final_gb.fit(
    X,
    y
)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": final_gb.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(
        index=False
    )
)

importance.to_csv(
    "v3_model_v2_feature_importance.csv",
    index=False
)


# ============================================================
# 17. SAVE FEATURE MATRIX
# ============================================================

feature_output = X.copy()

feature_output["request_to_ready_ms"] = y.values
feature_output["analysis_group"] = groups.values

feature_output.to_csv(
    "v3_model_v2_features.csv",
    index=False
)


# ============================================================
# 18. FINAL
# ============================================================

print("\n" + "=" * 90)
print("MODEL V2 COMPLETE")
print("=" * 90)

print("Saved:")
print(" - v3_model_v2_comparison.csv")
print(" - v3_model_v2_feature_importance.csv")
print(" - v3_model_v2_features.csv")

print("=" * 90)