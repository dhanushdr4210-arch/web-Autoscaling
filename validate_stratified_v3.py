import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("v3_master_100_raw.csv")

# CPU sweep contains four separate conditions.
df["analysis_group"] = df["source_condition"]

cpu_mask = df["source_condition"] == "CPU_SWEEP"

df.loc[cpu_mask, "analysis_group"] = (
    df.loc[cpu_mask, "experiment_original"]
)

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

# CPU-memory interaction
X["cpu_x_memory"] = (
    X["cpu_pct"] * X["memory_pct"]
)

y = df["request_to_ready_ms"].astype(float)

# ============================================================
# STRATIFIED 5-FOLD
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

strata = df["analysis_group"]

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

print("=" * 80)
print("STARTSCALE V3 STRATIFIED 5-FOLD VALIDATION")
print("=" * 80)

print("\nCondition distribution:")

print(
    df["analysis_group"]
    .value_counts()
    .sort_index()
)

results = []

for name, model in models.items():

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv.split(X, strata),
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
        "MAE_ms": mae,
        "RMSE_ms": rmse,
        "R2": r2,
    })

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "v3_stratified_5fold_results.csv",
    index=False
)

print("\nSaved:")
print("v3_stratified_5fold_results.csv")

print("=" * 80)