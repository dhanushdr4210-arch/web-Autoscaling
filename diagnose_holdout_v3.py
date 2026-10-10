import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv("v3_master_100_raw.csv")

# CPU sweep has 4 distinct experimental conditions.
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

X["cpu_x_memory"] = (
    X["cpu_pct"] * X["memory_pct"]
)

y = df["request_to_ready_ms"].astype(float)

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

logo = LeaveOneGroupOut()

results = []

print("=" * 90)
print("STARTSCALE V3 CONDITION-HOLDOUT DIAGNOSTIC")
print("=" * 90)

for train_idx, test_idx in logo.split(
    X,
    y,
    groups=groups
):

    train_X = X.iloc[train_idx]
    test_X = X.iloc[test_idx]

    train_y = y.iloc[train_idx]
    test_y = y.iloc[test_idx]

    heldout_group = groups.iloc[test_idx].iloc[0]

    actual_mean = test_y.mean()

    print("\n" + "-" * 90)
    print("HELD-OUT CONDITION:", heldout_group)
    print("Actual mean latency:", round(actual_mean, 2), "ms")
    print("Trials:", len(test_idx))

    for name, model in models.items():

        model.fit(
            train_X,
            train_y
        )

        pred = model.predict(test_X)

        mae = mean_absolute_error(
            test_y,
            pred
        )

        predicted_mean = pred.mean()

        bias = (
            predicted_mean -
            actual_mean
        )

        results.append({
            "heldout_condition": heldout_group,
            "model": name,
            "actual_mean_ms": actual_mean,
            "predicted_mean_ms": predicted_mean,
            "bias_ms": bias,
            "MAE_ms": mae,
        })

        print(
            f"{name:22s} "
            f"MAE={mae:8.2f} ms  "
            f"PredMean={predicted_mean:8.2f} ms  "
            f"Bias={bias:8.2f} ms"
        )

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "v3_condition_holdout_diagnostic.csv",
    index=False
)

# ============================================================
# AGGREGATE
# ============================================================

print("\n" + "=" * 90)
print("AVERAGE HOLDOUT PERFORMANCE")
print("=" * 90)

summary = (
    results_df
    .groupby("model")
    .agg(
        mean_MAE_ms=("MAE_ms", "mean"),
        median_MAE_ms=("MAE_ms", "median"),
        mean_abs_bias_ms=(
            "bias_ms",
            lambda x: np.mean(np.abs(x))
        ),
    )
    .sort_values("mean_MAE_ms")
)

print(
    summary.round(2).to_string()
)

print("\nSaved:")
print("v3_condition_holdout_diagnostic.csv")

print("=" * 90)