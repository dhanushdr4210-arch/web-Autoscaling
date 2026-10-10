import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

DATASET = "startscale_startup_model_ready_v2.csv"

FEATURES = [
    "cpu_pct",
    "memory_pct",
    "cache_warm",
    "scale_from_num",
]

TARGET = "target_latency_ms"
GROUP = "condition"

df = pd.read_csv(DATASET)

print("=" * 65)
print(" StartScale Leave-One-Condition-Out Validation V2")
print("=" * 65)

print(f"Dataset: {DATASET}")
print(f"Rows: {len(df)}")
print()

print("Conditions:")
print(df[GROUP].value_counts().to_string())

X = df[FEATURES]
y = df[TARGET]
groups = df[GROUP]

model = GradientBoostingRegressor(
    n_estimators=120,
    learning_rate=0.05,
    max_depth=2,
    min_samples_leaf=2,
    random_state=42,
)

logo = LeaveOneGroupOut()

all_predictions = []
all_actual = []
all_conditions = []

fold = 0

for train_idx, test_idx in logo.split(
    X,
    y,
    groups,
):
    fold += 1

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    held_out_condition = groups.iloc[test_idx].iloc[0]

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print()
    print(f"Fold {fold}")
    print(
        f"Held-out condition: "
        f"{held_out_condition}"
    )
    print(f"Train samples: {len(train_idx)}")
    print(f"Test samples : {len(test_idx)}")
    print(f"MAE          : {mae:.2f} ms")
    print(f"RMSE         : {rmse:.2f} ms")
    print(f"R²           : {r2:.4f}")

    all_predictions.extend(predictions)
    all_actual.extend(y_test)
    all_conditions.extend(
        [held_out_condition] * len(test_idx)
    )

overall_mae = mean_absolute_error(
    all_actual,
    all_predictions,
)

overall_rmse = np.sqrt(
    mean_squared_error(
        all_actual,
        all_predictions,
    )
)

overall_r2 = r2_score(
    all_actual,
    all_predictions,
)

print()
print("=" * 65)
print(" OVERALL V2 UNSEEN-CONDITION PERFORMANCE")
print("=" * 65)

print(f"MAE  : {overall_mae:.2f} ms")
print(f"RMSE : {overall_rmse:.2f} ms")
print(f"R²   : {overall_r2:.4f}")

results = pd.DataFrame(
    {
        "condition": all_conditions,
        "actual_ms": all_actual,
        "predicted_ms": all_predictions,
    }
)

results["absolute_error_ms"] = (
    results["actual_ms"]
    - results["predicted_ms"]
).abs()

results.to_csv(
    "startscale_grouped_validation_results_v2.csv",
    index=False,
)

print()
print(
    "Saved: "
    "startscale_grouped_validation_results_v2.csv"
)