import pandas as pd
import numpy as np


FILE = "v3_master_150_v3_spatial_features.csv"
TARGET = "request_to_ready_ms"


print("=" * 80)
print("STARTSCALE — STARTUP READINESS DELAY ANALYSIS")
print("=" * 80)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(FILE)

print("\nDataset shape:", df.shape)

latency = df[TARGET].dropna()


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("1. OVERALL REQUEST-TO-READY DISTRIBUTION")
print("=" * 80)

stats = latency.describe(
    percentiles=[
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99
    ]
)

print(stats)


print("\nExplicit statistics:")

print(f"Mean       : {latency.mean():.2f} ms")
print(f"Median     : {latency.median():.2f} ms")
print(f"Std        : {latency.std():.2f} ms")
print(f"Minimum    : {latency.min():.2f} ms")
print(f"Maximum    : {latency.max():.2f} ms")

for q in [0.50, 0.75, 0.90, 0.95, 0.99]:

    print(
        f"P{int(q*100):02d}        : "
        f"{latency.quantile(q):.2f} ms"
    )


# ============================================================
# RANGE
# ============================================================

print("\n" + "=" * 80)
print("2. DELAY RANGE")
print("=" * 80)

print(
    f"Range: "
    f"{latency.min():.2f} → "
    f"{latency.max():.2f} ms"
)

print(
    f"Spread: "
    f"{latency.max() - latency.min():.2f} ms"
)


# ============================================================
# COLD VS WARM
# ============================================================

print("\n" + "=" * 80)
print("3. CACHE STATE")
print("=" * 80)

if "image_cache_state" in df.columns:

    cache_stats = (
        df.groupby("image_cache_state")[TARGET]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ])
    )

    print(cache_stats)


# ============================================================
# SOURCE CONDITIONS
# ============================================================

print("\n" + "=" * 80)
print("4. STARTUP LATENCY BY EXPERIMENTAL CONDITION")
print("=" * 80)

condition_stats = (
    df.groupby("source_condition")[TARGET]
    .agg([
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ])
    .sort_values("mean")
)

print(condition_stats.to_string())


# ============================================================
# CONDITION PERCENTILES
# ============================================================

print("\n" + "=" * 80)
print("5. P90 / P95 BY CONDITION")
print("=" * 80)

condition_percentiles = (
    df.groupby("source_condition")[TARGET]
    .quantile([0.50, 0.90, 0.95])
    .unstack()
)

condition_percentiles.columns = [
    "P50",
    "P90",
    "P95"
]

print(
    condition_percentiles
    .sort_values("P90")
    .to_string()
)


# ============================================================
# LATENCY BUCKETS
# ============================================================

print("\n" + "=" * 80)
print("6. READINESS DELAY BUCKETS")
print("=" * 80)

bins = [
    0,
    4000,
    5000,
    6000,
    7000,
    8000,
    10000,
    np.inf
]

labels = [
    "<4s",
    "4–5s",
    "5–6s",
    "6–7s",
    "7–8s",
    "8–10s",
    ">10s"
]

buckets = pd.cut(
    latency,
    bins=bins,
    labels=labels,
    right=False
)

bucket_counts = buckets.value_counts(
    sort=False
)

print(bucket_counts)

print("\nPercentages:")

print(
    (bucket_counts / len(latency) * 100)
    .round(2)
)


# ============================================================
# HIGH-DELAY EVENTS
# ============================================================

print("\n" + "=" * 80)
print("7. HIGH-DELAY EVENTS")
print("=" * 80)

p95 = latency.quantile(0.95)

high_delay = df[
    df[TARGET] >= p95
].copy()

print(
    f"P95 threshold: {p95:.2f} ms"
)

print(
    f"Events >= P95: {len(high_delay)}"
)

columns_to_show = [
    "experiment",
    "trial",
    "source_condition",
    "node_name",
    "image_cache_state",
    TARGET
]

available = [
    c for c in columns_to_show
    if c in high_delay.columns
]

print(
    high_delay[available]
    .sort_values(TARGET, ascending=False)
    .to_string(index=False)
)


# ============================================================
# SAME CONDITION VARIABILITY
# ============================================================

print("\n" + "=" * 80)
print("8. WITHIN-CONDITION VARIABILITY")
print("=" * 80)

within_condition = (
    df.groupby("source_condition")[TARGET]
    .agg([
        "count",
        "mean",
        "std"
    ])
)

within_condition["cv_percent"] = (
    within_condition["std"]
    / within_condition["mean"]
    * 100
)

print(
    within_condition
    .sort_values("cv_percent", ascending=False)
    .to_string()
)


# ============================================================
# SUMMARY FOR CONTROL
# ============================================================

print("\n" + "=" * 80)
print("9. CONTROL-RELEVANT SUMMARY")
print("=" * 80)

print(
    f"Mean readiness delay : "
    f"{latency.mean()/1000:.3f} seconds"
)

print(
    f"P50 readiness delay  : "
    f"{latency.quantile(.50)/1000:.3f} seconds"
)

print(
    f"P90 readiness delay  : "
    f"{latency.quantile(.90)/1000:.3f} seconds"
)

print(
    f"P95 readiness delay  : "
    f"{latency.quantile(.95)/1000:.3f} seconds"
)

print(
    f"P99 readiness delay  : "
    f"{latency.quantile(.99)/1000:.3f} seconds"
)

print(
    "\nInterpretation:"
)

print(
    "A reactive autoscaler that waits for demand "
    "before requesting capacity cannot make the "
    "new pod Ready instantaneously."
)

print(
    "The measured request-to-ready delay therefore "
    "represents a potentially important control-loop "
    "dead time."
)


# ============================================================
# SAVE
# ============================================================

condition_stats.to_csv(
    "startup_delay_condition_stats.csv"
)

condition_percentiles.to_csv(
    "startup_delay_condition_percentiles.csv"
)

print("\nSaved:")
print("  startup_delay_condition_stats.csv")
print("  startup_delay_condition_percentiles.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)