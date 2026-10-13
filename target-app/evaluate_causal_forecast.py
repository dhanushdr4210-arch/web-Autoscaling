import pandas as pd
import numpy as np

TRACE_FILE = "azure_replay_20min.csv"

TRACE_MIN_RPS = 25.0
TRACE_MAX_RPS = 40.0

SMOOTHING_SEC = 10
FORECAST_HORIZON_SEC = 10
HIGH_LOAD_RPS = 35.0
REQUIRED_LEAD_SEC = 8.98

d = pd.read_csv(TRACE_FILE)

rps = (
    TRACE_MIN_RPS
    + (d["requests_per_second"] - d["requests_per_second"].min())
    / (d["requests_per_second"].max() - d["requests_per_second"].min())
    * (TRACE_MAX_RPS - TRACE_MIN_RPS)
)

d["rps"] = rps

d["smoothed_rps"] = (
    d["rps"]
    .rolling(SMOOTHING_SEC, min_periods=SMOOTHING_SEC)
    .mean()
)

# Causal forecast:
# At time t, only the previous 5-second trend is used.
d["slope_5s"] = (
    d["smoothed_rps"]
    - d["smoothed_rps"].shift(5)
) / 5.0

d["forecast_10s"] = (
    d["smoothed_rps"]
    + d["slope_5s"] * FORECAST_HORIZON_SEC
)

# Actual future value is used ONLY for evaluation.
d["actual_10s"] = d["smoothed_rps"].shift(
    -FORECAST_HORIZON_SEC
)

valid = d[
    d["smoothed_rps"].notna()
    & d["forecast_10s"].notna()
    & d["actual_10s"].notna()
].copy()

error = (
    valid["forecast_10s"]
    - valid["actual_10s"]
)

print("=" * 80)
print("STARTSCALE CAUSAL FORECAST EVALUATION")
print("=" * 80)

print()
print(f"Samples              : {len(valid)}")
print(f"Smoothing             : {SMOOTHING_SEC} sec")
print(f"Forecast horizon      : {FORECAST_HORIZON_SEC} sec")
print(f"High-load reference   : {HIGH_LOAD_RPS} RPS")
print(f"Required lead         : {REQUIRED_LEAD_SEC:.2f} sec")

print()
print(f"MAE                   : {error.abs().mean():.3f} RPS")
print(
    f"RMSE                  : "
    f"{np.sqrt((error ** 2).mean()):.3f} RPS"
)
print(
    f"P90 absolute error    : "
    f"{error.abs().quantile(0.90):.3f} RPS"
)

# Detect actual high-load episode starts.
high = d["smoothed_rps"] >= HIGH_LOAD_RPS

starts = np.where(
    high
    & ~np.r_[False, high.to_numpy()[:-1]]
)[0]

print()
print(f"Actual high-load starts: {len(starts)}")

# For every timestamp, determine whether the causal forecast
# predicts high load 10 seconds ahead.
valid["predicted_high_load"] = (
    valid["forecast_10s"] >= HIGH_LOAD_RPS
)

print(
    f"Predicted high-load points: "
    f"{int(valid['predicted_high_load'].sum())}"
)

# Save complete offline evaluation.
valid[
    [
        "second",
        "smoothed_rps",
        "slope_5s",
        "forecast_10s",
        "actual_10s",
        "predicted_high_load",
    ]
].to_csv(
    "startscale_causal_forecast_evaluation.csv",
    index=False,
)

print()
print(
    "Saved: "
    "startscale_causal_forecast_evaluation.csv"
)

print()
print("=" * 80)
