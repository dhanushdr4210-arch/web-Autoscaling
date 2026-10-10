import numpy as np
import pandas as pd


# ============================================================
# STARTSCALE — TAIL & CONDITION ANALYSIS
# ============================================================

DATA = "v3_master_150_v3_spatial_features.csv"
TARGET = "request_to_ready_ms"

df = pd.read_csv(DATA)

print("=" * 80)
print("STARTSCALE — TAIL & CONDITION ANALYSIS")
print("=" * 80)

print(f"\nDataset shape: {df.shape}")

# ------------------------------------------------------------
# Overall distribution
# ------------------------------------------------------------

y = df[TARGET]

p90 = y.quantile(0.90)
p95 = y.quantile(0.95)
p99 = y.quantile(0.99)

print("\n" + "=" * 80)
print("OVERALL READINESS-LATENCY DISTRIBUTION")
print("=" * 80)

print(f"Mean : {y.mean():.2f} ms")
print(f"Std  : {y.std():.2f} ms")
print(f"Min  : {y.min():.2f} ms")
print(f"P50  : {y.quantile(0.50):.2f} ms")
print(f"P75  : {y.quantile(0.75):.2f} ms")
print(f"P90  : {p90:.2f} ms")
print(f"P95  : {p95:.2f} ms")
print(f"P99  : {p99:.2f} ms")
print(f"Max  : {y.max():.2f} ms")


# ------------------------------------------------------------
# Tail buckets
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TAIL EVENT COUNTS")
print("=" * 80)

thresholds = [
    (">= P90", p90),
    (">= P95", p95),
    (">= P99", p99),
    (">= 8000 ms", 8000),
    (">= 9000 ms", 9000),
    (">= 10000 ms", 10000),
]

for name, threshold in thresholds:
    count = int((y >= threshold).sum())
    pct = count / len(df) * 100

    print(
        f"{name:<15}: "
        f"{count:3d} rows "
        f"({pct:5.2f}%)"
    )


# ------------------------------------------------------------
# Identify extreme events
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("EXTREME READINESS EVENTS")
print("=" * 80)

extreme = df[df[TARGET] >= p95].copy()

cols = [
    "experiment",
    "trial",
    "source_condition",
    "request_to_ready_ms",
    "image_cache_state",
    "node_cpu_pct_v2",
    "node_memory_pct_v2",
    "worker_cpu_mean",
    "cpu_spatial_imbalance",
]

available_cols = [
    c for c in cols if c in extreme.columns
]

print(
    extreme[
        available_cols
    ]
    .sort_values(
        TARGET,
        ascending=False
    )
    .to_string(index=False)
)


# ------------------------------------------------------------
# Condition-level statistics
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CONDITION-LEVEL LATENCY STATISTICS")
print("=" * 80)

condition_stats = (
    df.groupby("source_condition")[TARGET]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        p75=lambda x: x.quantile(0.75),
        p90=lambda x: x.quantile(0.90),
        p95=lambda x: x.quantile(0.95),
        maximum="max",
    )
    .reset_index()
)

condition_stats["cv_pct"] = (
    condition_stats["std"]
    / condition_stats["mean"]
    * 100
)

print(
    condition_stats
    .sort_values(
        "p95",
        ascending=False
    )
    .to_string(index=False)
)


# ------------------------------------------------------------
# High-delay event concentration by condition
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("HIGH-DELAY EVENT CONCENTRATION")
print("=" * 80)

high = df[TARGET] >= p95

tail_by_condition = (
    df.assign(high_delay=high)
    .groupby("source_condition")
    .agg(
        total_trials=(TARGET, "count"),
        high_delay_events=("high_delay", "sum"),
        mean_latency=(TARGET, "mean"),
        p95_latency=(
            TARGET,
            lambda x: x.quantile(0.95)
        ),
        max_latency=(TARGET, "max"),
    )
    .reset_index()
)

tail_by_condition["high_delay_pct"] = (
    tail_by_condition["high_delay_events"]
    / tail_by_condition["total_trials"]
    * 100
)

print(
    tail_by_condition
    .sort_values(
        "high_delay_events",
        ascending=False
    )
    .to_string(index=False)
)


# ------------------------------------------------------------
# Cache-state comparison
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CACHE-STATE TAIL ANALYSIS")
print("=" * 80)

cache_stats = (
    df.groupby("image_cache_state")[TARGET]
    .agg(
        count="count",
        mean="mean",
        median="median",
        p90=lambda x: x.quantile(0.90),
        p95=lambda x: x.quantile(0.95),
        maximum="max",
    )
    .reset_index()
)

print(
    cache_stats.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Spatial imbalance comparison
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SPATIAL IMBALANCE TAIL ANALYSIS")
print("=" * 80)

imbalance_median = df[
    "cpu_spatial_imbalance"
].median()

df["spatial_group"] = np.where(
    df["cpu_spatial_imbalance"]
    <= imbalance_median,
    "lower_imbalance",
    "higher_imbalance",
)

spatial_stats = (
    df.groupby("spatial_group")[TARGET]
    .agg(
        count="count",
        mean="mean",
        median="median",
        p90=lambda x: x.quantile(0.90),
        p95=lambda x: x.quantile(0.95),
        maximum="max",
    )
    .reset_index()
)

print(
    spatial_stats.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

condition_stats.to_csv(
    "tail_condition_statistics.csv",
    index=False,
)

tail_by_condition.to_csv(
    "tail_condition_concentration.csv",
    index=False,
)

cache_stats.to_csv(
    "tail_cache_statistics.csv",
    index=False,
)

spatial_stats.to_csv(
    "tail_spatial_statistics.csv",
    index=False,
)


print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print("tail_condition_statistics.csv")
print("tail_condition_concentration.csv")
print("tail_cache_statistics.csv")
print("tail_spatial_statistics.csv")

print("\nTAIL & CONDITION ANALYSIS COMPLETE")