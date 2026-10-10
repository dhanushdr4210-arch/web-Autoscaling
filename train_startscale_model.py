import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

DATASET = "startscale_model_features_v1.csv"

df = pd.read_csv(DATASET)

FEATURES = [
    "cpu_pct",
    "memory_pct",
    "cache_warm",
    "scale_from_num",
]

TARGET = "target_latency_ms"

X = df[FEATURES]
y = df[TARGET]

model = GradientBoostingRegressor(
    n_estimators=120,
    learning_rate=0.05,
    max_depth=2,
    min_samples_leaf=2,
    random_state=42,
)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
)

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

baseline_prediction = np.full(
    len(y),
    y.mean(),
)

baseline_mae = mean_absolute_error(
    y,
    baseline_prediction,
)

print("=" * 60)
print(" StartScale Gradient Boosting Pilot Model")
print("=" * 60)

print(f"Samples: {len(df)}")
print(f"Features: {FEATURES}")
print()

print("5-Fold Cross-Validation")
print(f"MAE  : {mae:.2f} ms")
print(f"RMSE : {rmse:.2f} ms")
print(f"R²   : {r2:.4f}")
print()

print("Mean-prediction baseline")
print(f"MAE  : {baseline_mae:.2f} ms")
print()

model.fit(X, y)

print("Feature importance")
print("-" * 40)

for feature, importance in sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
):
    print(
        f"{feature:20s} "
        f"{importance:.4f}"
    )

results = df.copy()
results["predicted_latency_ms"] = predictions
results["absolute_error_ms"] = (
    results[TARGET]
    - results["predicted_latency_ms"]
).abs()

results.to_csv(
    "startscale_model_predictions_v1.csv",
    index=False,
)

print()
print(
    "Saved: "
    "startscale_model_predictions_v1.csv"
)