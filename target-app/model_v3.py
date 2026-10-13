import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# LOAD DATA
# ============================================================

FILE = "v3_master_100_raw.csv"

df = pd.read_csv(FILE)

# ============================================================
# CREATE ANALYSIS GROUP
# ============================================================

# CPU_SWEEP contains four separate controlled conditions.
# Other experiments already have their own condition names.

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
# TARGET
# ============================================================

y = df["request_to_ready_ms"].astype(float)

# ============================================================
# FEATURES
# ============================================================

X = pd.DataFrame({
    "cpu_pct": pd.to_numeric(
        df["pre_scale_node_cpu_pct"],
        errors="coerce"
    ),

    "memory_pct": pd.to_numeric(
        df["pre_scale_node_memory_pct"],
        errors="coerce"
    ),

    "cache_warm": (
        df["image_cache_state"]
        .astype(str)
        .str.lower()
        .eq("warm")
        .astype(int)
    ),
})

# Interaction feature
X["cpu_x_memory"] = (
    X["cpu_pct"] * X["memory_pct"]
)

# ============================================================
# VALIDATION
# ============================================================

random_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

group_cv = GroupKFold(
    n_splits=df["analysis_group"].nunique()
)

groups = df["analysis_group"]

# ============================================================
# MODELS
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
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
    cv,
    cv_name,
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
        "validation": cv_name,
        "MAE_ms": mae,
        "RMSE_ms": rmse,
        "R2": r2,
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

print("=" * 80)
print("STARTSCALE V3 MODEL COMPARISON")
print("=" * 80)

print("\nFeatures:")
print(X.columns.tolist())

print("\nAnalysis groups:")
print(
    df["analysis_group"]
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# RANDOM 5-FOLD
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("RANDOM 5-FOLD CROSS-VALIDATION")
print("=" * 80)

for name, model in models.items():

    result = evaluate_model(
        name,
        model,
        random_cv,
        "Random_5Fold"
    )

    results.append(result)

# ------------------------------------------------------------
# CONDITION-HOLDOUT
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CONDITION-HOLDOUT CROSS-VALIDATION")
print("=" * 80)

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
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["validation", "MAE_ms"]
)

results_df.to_csv(
    "v3_model_comparison.csv",
    index=False
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("GRADIENT BOOSTING FEATURE IMPORTANCE")
print("=" * 80)

final_gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=2,
    min_samples_leaf=3,
    random_state=42
)

final_gb.fit(X, y)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": final_gb.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(
        index=False
    )
)

importance.to_csv(
    "v3_gradient_boosting_feature_importance.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON SAVED")
print("=" * 80)

print(
    "Output: v3_model_comparison.csv"
)

print(
    "Output: v3_gradient_boosting_feature_importance.csv"
)

print("=" * 80)