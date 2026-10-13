import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATA
# ============================================================

FILE = "v3_master_140_raw.csv"

df = pd.read_csv(FILE)

print("=" * 90)
print("STARTSCALE V3 — CONDITION HOLDOUT DIAGNOSTIC")
print("=" * 90)

print(f"Dataset shape: {df.shape}")


# ============================================================
# FEATURES
# ============================================================

def parse_percent(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        value = value.strip()

        if value.endswith("%"):
            return float(value[:-1])

    return float(value)


def parse_cluster_features(row):

    state = row["pre_scale_all_nodes_json"]

    if isinstance(state, str):
        import json
        state = json.loads(state)

    worker_cpu = []
    worker_memory = []

    for node_name, node_data in state.items():

        name = str(node_name).lower()

        if "worker" not in name:
            continue

        # New schema
        if "cpu_m" in node_data:
            cpu_m = pd.to_numeric(
                node_data.get("cpu_m"),
                errors="coerce"
            )

            cpu_pct = (
                cpu_m / 16000.0 * 100.0
                if not pd.isna(cpu_m)
                else np.nan
            )

        # Old schema
        else:
            cpu_pct = parse_percent(
                node_data.get("cpu_pct")
            )

        # New schema
        if "memory_mi" in node_data:

            memory_mi = pd.to_numeric(
                node_data.get("memory_mi"),
                errors="coerce"
            )

            memory_pct = (
                memory_mi / (7970460 / 1024.0) * 100.0
                if not pd.isna(memory_mi)
                else np.nan
            )

        # Old schema
        else:
            memory_pct = parse_percent(
                node_data.get("memory_pct")
            )

        worker_cpu.append(cpu_pct)
        worker_memory.append(memory_pct)

    worker_cpu = np.array(worker_cpu, dtype=float)
    worker_memory = np.array(worker_memory, dtype=float)

    return pd.Series({
        "worker_cpu_mean": np.nanmean(worker_cpu),
        "worker_cpu_max": np.nanmax(worker_cpu),
        "worker_cpu_std": np.nanstd(worker_cpu),

        "worker_memory_mean": np.nanmean(worker_memory),
        "worker_memory_max": np.nanmax(worker_memory),
        "worker_memory_std": np.nanstd(worker_memory),
    })


# Generate worker features
cluster_features = df.apply(
    parse_cluster_features,
    axis=1
)

df = pd.concat(
    [df, cluster_features],
    axis=1
)


# ============================================================
# NODE FEATURES
# ============================================================

df["node_cpu_pct"] = df["pre_scale_node_cpu_pct"].apply(
    parse_percent
)

df["node_memory_pct"] = df["pre_scale_node_memory_pct"].apply(
    parse_percent
)

df["cache_warm"] = (
    df["image_cache_state"]
    .astype(str)
    .str.lower()
    .eq("warm")
    .astype(int)
)


df["node_cpu_x_memory"] = (
    df["node_cpu_pct"] *
    df["node_memory_pct"]
)

df["cluster_cpu_x_memory"] = (
    df["worker_cpu_mean"] *
    df["worker_memory_mean"]
)

df["node_cpu_delta_mean"] = (
    df["node_cpu_pct"] -
    df["worker_cpu_mean"]
)

df["node_memory_delta_mean"] = (
    df["node_memory_pct"] -
    df["worker_memory_mean"]
)


# ============================================================
# FEATURE MATRIX
# ============================================================

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

X = df[FEATURES]
y = df["request_to_ready_ms"]

conditions = df["source_condition"]


# ============================================================
# CHECK
# ============================================================

print("\nMissing values:")

print(X.isna().sum())

if X.isna().sum().sum() > 0:
    raise ValueError("Feature matrix contains NaN values.")


# ============================================================
# LOCO
# ============================================================

print("\n")
print("=" * 90)
print("CONDITION-BY-CONDITION LOCO DIAGNOSTIC")
print("=" * 90)

results = []


for condition in sorted(conditions.unique()):

    train_mask = conditions != condition
    test_mask = conditions == condition

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    # -------------------------------
    # Baseline
    # -------------------------------

    baseline_prediction = np.repeat(
        y_train.mean(),
        len(y_test)
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_prediction
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_prediction
        )
    )

    # -------------------------------
    # Random Forest
    # -------------------------------

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    rf_prediction = rf.predict(X_test)

    rf_mae = mean_absolute_error(
        y_test,
        rf_prediction
    )

    rf_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            rf_prediction
        )
    )

    # -------------------------------
    # Gradient Boosting
    # -------------------------------

    gb = GradientBoostingRegressor(
        random_state=42
    )

    gb.fit(X_train, y_train)

    gb_prediction = gb.predict(X_test)

    gb_mae = mean_absolute_error(
        y_test,
        gb_prediction
    )

    gb_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            gb_prediction
        )
    )

    # -------------------------------
    # Improvement
    # -------------------------------

    improvement = (
        (baseline_mae - gb_mae)
        / baseline_mae
        * 100
    )

    results.append({
        "condition": condition,
        "n": len(y_test),

        "baseline_mae": baseline_mae,
        "rf_mae": rf_mae,
        "gb_mae": gb_mae,

        "baseline_rmse": baseline_rmse,
        "rf_rmse": rf_rmse,
        "gb_rmse": gb_rmse,

        "gb_improvement_pct": improvement
    })


# ============================================================
# DISPLAY
# ============================================================

result_df = pd.DataFrame(results)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

print("\n")
print(result_df.to_string(index=False))


# ============================================================
# SORT BY GB ERROR
# ============================================================

print("\n")
print("=" * 90)
print("HARDEST CONDITIONS — GRADIENT BOOSTING")
print("=" * 90)

hardest = result_df.sort_values(
    "gb_mae",
    ascending=False
)

print(
    hardest[
        [
            "condition",
            "n",
            "baseline_mae",
            "gb_mae",
            "gb_improvement_pct"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT = "v3_loco_diagnostic_140.csv"

result_df.to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(f" - {OUTPUT}")

print("\nDONE.")