import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


V3_FILE = "v3_master_150_v3_spatial_features.csv"
TARGET = "request_to_ready_ms"


# ============================================================
# CLEAN FEATURE SETS
# ============================================================

# A:
# Everything except worker-level CPU aggregate/spatial features.
A_FEATURES = [
    "node_cpu_pct_v2",
    "node_memory_pct_v2",

    "worker_memory_mean",
    "worker_memory_max",
    "worker_memory_std",

    "node_cpu_x_memory",
    "cluster_cpu_x_memory",

    "image_cache_state",
]


# B:
# A + aggregate worker CPU.
B_FEATURES = A_FEATURES + [
    "worker_cpu_mean",
]


# C:
# B + spatial distribution information.
C_FEATURES = B_FEATURES + [
    "cpu_spatial_imbalance",
]


FEATURE_SETS = {
    "A_NonWorkerCPU": A_FEATURES,
    "B_AggregateCPU": B_FEATURES,
    "C_AggregatePlusSpatial": C_FEATURES,
}


# ============================================================
# DATA
# ============================================================

df = pd.read_csv(V3_FILE)

print("=" * 80)
print("STARTSCALE: CLEAN AGGREGATE CPU vs SPATIAL ABLATION")
print("=" * 80)

print("\nDataset shape:", df.shape)


# ============================================================
# ENCODE FEATURES
# ============================================================

def prepare_X(features):

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

        print("\nMissing values:")
        print(X.isna().sum())

        raise ValueError(
            "Missing values detected."
        )

    return X


# ============================================================
# MODELS
# ============================================================

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


# ============================================================
# EVALUATION
# ============================================================

def evaluate(features, splits, model_name):

    X = prepare_X(features)

    y = df[TARGET].values

    mae_values = []
    rmse_values = []
    r2_values = []

    for train_idx, test_idx in splits:

        model = make_model(model_name)

        model.fit(
            X.iloc[train_idx],
            y[train_idx]
        )

        prediction = model.predict(
            X.iloc[test_idx]
        )

        mae = mean_absolute_error(
            y[test_idx],
            prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                y[test_idx],
                prediction
            )
        )

        r2 = r2_score(
            y[test_idx],
            prediction
        )

        mae_values.append(mae)
        rmse_values.append(rmse)
        r2_values.append(r2)

    return {
        "MAE": np.mean(mae_values),
        "RMSE": np.mean(rmse_values),
        "R2": np.mean(r2_values),
    }


# ============================================================
# RANDOM 5-FOLD
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

random_splits = list(
    kf.split(df)
)


# ============================================================
# STRATIFIED 5-FOLD
# ============================================================

# Stratify using source condition because every condition
# contains approximately 10 observations.

strat_labels = df["source_condition"].astype(str)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

stratified_splits = list(
    skf.split(df, strat_labels)
)


# ============================================================
# LOCO
# ============================================================

groups = df["source_condition"].astype(str)

gkf = GroupKFold(
    n_splits=5
)

loco_splits = list(
    gkf.split(
        df,
        groups=groups
    )
)


# ============================================================
# RUN EXPERIMENT
# ============================================================

all_results = []


VALIDATION_TYPES = [
    ("Random 5-Fold", random_splits),
    ("Stratified 5-Fold", stratified_splits),
    ("LOCO", loco_splits),
]


for validation_name, splits in VALIDATION_TYPES:

    print("\n" + "=" * 80)
    print(validation_name)
    print("=" * 80)

    for feature_name, features in FEATURE_SETS.items():

        for model_name in [
            "Random Forest",
            "Gradient Boosting"
        ]:

            metrics = evaluate(
                features,
                splits,
                model_name
            )

            result = {
                "validation": validation_name,
                "feature_set": feature_name,
                "model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }

            all_results.append(result)

            print(
                f"{feature_name:28s} | "
                f"{model_name:18s} | "
                f"MAE={metrics['MAE']:8.2f} | "
                f"RMSE={metrics['RMSE']:8.2f} | "
                f"R2={metrics['R2']:+.4f}"
            )


results_df = pd.DataFrame(all_results)


# ============================================================
# SAVE
# ============================================================

output_file = "clean_aggregate_vs_spatial_results.csv"

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# A -> B
# AGGREGATE CPU CONTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("A -> B : VALUE OF AGGREGATE WORKER CPU")
print("=" * 80)

for validation in results_df["validation"].unique():

    for model in results_df["model"].unique():

        A = results_df[
            (results_df["validation"] == validation) &
            (results_df["model"] == model) &
            (results_df["feature_set"] == "A_NonWorkerCPU")
        ].iloc[0]

        B = results_df[
            (results_df["validation"] == validation) &
            (results_df["model"] == model) &
            (results_df["feature_set"] == "B_AggregateCPU")
        ].iloc[0]

        mae_change = (
            (A["MAE"] - B["MAE"])
            / A["MAE"]
            * 100
        )

        rmse_change = (
            (A["RMSE"] - B["RMSE"])
            / A["RMSE"]
            * 100
        )

        r2_change = B["R2"] - A["R2"]

        print(f"\n{validation} | {model}")

        print(
            f"  MAE improvement  : {mae_change:+.2f}%"
        )

        print(
            f"  RMSE improvement : {rmse_change:+.2f}%"
        )

        print(
            f"  R2 change        : {r2_change:+.4f}"
        )


# ============================================================
# B -> C
# SPATIAL CONTRIBUTION BEYOND AGGREGATE CPU
# ============================================================

print("\n" + "=" * 80)
print("B -> C : SPATIAL VALUE BEYOND AGGREGATE CPU")
print("=" * 80)

for validation in results_df["validation"].unique():

    for model in results_df["model"].unique():

        B = results_df[
            (results_df["validation"] == validation) &
            (results_df["model"] == model) &
            (results_df["feature_set"] == "B_AggregateCPU")
        ].iloc[0]

        C = results_df[
            (results_df["validation"] == validation) &
            (results_df["model"] == model) &
            (results_df["feature_set"] == "C_AggregatePlusSpatial")
        ].iloc[0]

        mae_change = (
            (B["MAE"] - C["MAE"])
            / B["MAE"]
            * 100
        )

        rmse_change = (
            (B["RMSE"] - C["RMSE"])
            / B["RMSE"]
            * 100
        )

        r2_change = C["R2"] - B["R2"]

        print(f"\n{validation} | {model}")

        print(
            f"  MAE improvement  : {mae_change:+.2f}%"
        )

        print(
            f"  RMSE improvement : {rmse_change:+.2f}%"
        )

        print(
            f"  R2 change        : {r2_change:+.4f}"
        )


print("\n" + "=" * 80)
print(
    f"Saved: {output_file}"
)
print("=" * 80)