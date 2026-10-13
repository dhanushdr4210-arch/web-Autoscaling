import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

FILE = "v3_master_100_raw.csv"
OUTDIR = Path("v3_eda_outputs")
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(FILE)

# ============================================================
# BASIC CLEANING FOR ANALYSIS
# ============================================================

numeric_columns = [
    "pre_scale_node_cpu_pct",
    "pre_scale_node_memory_pct",
    "request_to_ready_ms",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# CPU sweep has multiple CPU levels under one source_condition.
# Preserve those levels using the original experiment label.
df["analysis_condition"] = df["source_condition"]

cpu_mask = df["source_condition"] == "CPU_SWEEP"

df.loc[
    cpu_mask,
    "analysis_condition"
] = df.loc[
    cpu_mask,
    "experiment_original"
]

# ============================================================
# 1. OVERALL DATASET SUMMARY
# ============================================================

print("=" * 70)
print("STARTSCALE V3 EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nOverall latency:")
print(
    df["request_to_ready_ms"]
    .describe()
    .round(2)
)

print("\nCondition counts:")
print(
    df["analysis_condition"]
    .value_counts()
    .sort_index()
)

# ============================================================
# 2. CONDITION SUMMARY
# ============================================================

condition_summary = (
    df.groupby("analysis_condition")
      .agg(
          trials=("request_to_ready_ms", "count"),
          cpu_mean=("pre_scale_node_cpu_pct", "mean"),
          cpu_min=("pre_scale_node_cpu_pct", "min"),
          cpu_max=("pre_scale_node_cpu_pct", "max"),
          memory_mean=("pre_scale_node_memory_pct", "mean"),
          memory_min=("pre_scale_node_memory_pct", "min"),
          memory_max=("pre_scale_node_memory_pct", "max"),
          latency_mean=("request_to_ready_ms", "mean"),
          latency_median=("request_to_ready_ms", "median"),
          latency_std=("request_to_ready_ms", "std"),
          latency_min=("request_to_ready_ms", "min"),
          latency_max=("request_to_ready_ms", "max"),
      )
      .round(2)
)

print("\n" + "=" * 70)
print("CONDITION SUMMARY")
print("=" * 70)
print(condition_summary.to_string())

condition_summary.to_csv(
    OUTDIR / "v3_eda_condition_summary.csv"
)

# ============================================================
# 3. LATENCY DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    df["request_to_ready_ms"],
    bins=15,
    edgecolor="black"
)

plt.xlabel("Request-to-Ready Latency (ms)")
plt.ylabel("Number of Trials")
plt.title("V3 Startup Latency Distribution")
plt.tight_layout()

plt.savefig(
    OUTDIR / "01_latency_distribution.png",
    dpi=200
)

plt.close()

# ============================================================
# 4. LATENCY BY CONDITION - BOXPLOT
# ============================================================

conditions = sorted(
    df["analysis_condition"].unique()
)

box_data = [
    df.loc[
        df["analysis_condition"] == condition,
        "request_to_ready_ms"
    ]
    for condition in conditions
]

plt.figure(figsize=(12, 7))

plt.boxplot(
    box_data,
    tick_labels=conditions,
    showmeans=True
)

plt.xlabel("Experimental Condition")
plt.ylabel("Request-to-Ready Latency (ms)")
plt.title("Startup Latency by Experimental Condition")
plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTDIR / "02_latency_by_condition.png",
    dpi=200
)

plt.close()

# ============================================================
# 5. CPU VS LATENCY
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["pre_scale_node_cpu_pct"],
    df["request_to_ready_ms"],
    alpha=0.75
)

plt.xlabel("Pre-scale Node CPU (%)")
plt.ylabel("Request-to-Ready Latency (ms)")
plt.title("CPU Utilization vs Startup Latency")
plt.grid(True, alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTDIR / "03_cpu_vs_latency.png",
    dpi=200
)

plt.close()

# ============================================================
# 6. MEMORY VS LATENCY
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["pre_scale_node_memory_pct"],
    df["request_to_ready_ms"],
    alpha=0.75
)

plt.xlabel("Pre-scale Node Memory (%)")
plt.ylabel("Request-to-Ready Latency (ms)")
plt.title("Memory Utilization vs Startup Latency")
plt.grid(True, alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTDIR / "04_memory_vs_latency.png",
    dpi=200
)

plt.close()

# ============================================================
# 7. CPU SWEEP ONLY
# ============================================================

cpu_df = df[
    df["source_condition"] == "CPU_SWEEP"
].copy()

cpu_conditions = [
    "V3_LowPressure_Warm",
    "V3_CPU7_Warm",
    "V3_CPU13_Warm",
    "V3_CPU19_Warm",
]

cpu_box_data = []

cpu_labels = []

for condition in cpu_conditions:

    subset = cpu_df[
        cpu_df["experiment_original"] == condition
    ]

    if len(subset) > 0:

        cpu_box_data.append(
            subset["request_to_ready_ms"]
        )

        cpu_labels.append(condition)

plt.figure(figsize=(11, 7))

plt.boxplot(
    cpu_box_data,
    tick_labels=cpu_labels,
    showmeans=True
)

plt.xlabel("CPU Sweep Condition")
plt.ylabel("Request-to-Ready Latency (ms)")
plt.title("CPU Sweep: Startup Latency Distribution")
plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTDIR / "05_cpu_sweep_latency.png",
    dpi=200
)

plt.close()

# ============================================================
# 8. CACHE WARM VS COLD
# ============================================================

cache_df = df[
    df["source_condition"].isin(
        ["CACHE_WARM", "CACHE_COLD"]
    )
].copy()

cache_summary = (
    cache_df.groupby("source_condition")
            .agg(
                trials=("request_to_ready_ms", "count"),
                mean_latency=("request_to_ready_ms", "mean"),
                median_latency=("request_to_ready_ms", "median"),
                std_latency=("request_to_ready_ms", "std"),
            )
            .round(2)
)

print("\n" + "=" * 70)
print("CACHE SUMMARY")
print("=" * 70)

print(cache_summary.to_string())

cache_summary.to_csv(
    OUTDIR / "v3_eda_cache_summary.csv"
)

cache_conditions = [
    "CACHE_WARM",
    "CACHE_COLD",
]

cache_box_data = [
    cache_df.loc[
        cache_df["source_condition"] == condition,
        "request_to_ready_ms"
    ]
    for condition in cache_conditions
]

plt.figure(figsize=(8, 6))

plt.boxplot(
    cache_box_data,
    tick_labels=cache_conditions,
    showmeans=True
)

plt.xlabel("Cache State")
plt.ylabel("Request-to-Ready Latency (ms)")
plt.title("Warm vs Cold Cache Startup Latency")

plt.tight_layout()

plt.savefig(
    OUTDIR / "06_cache_warm_vs_cold.png",
    dpi=200
)

plt.close()

# ============================================================
# 9. COMBINED CPU + MEMORY CONDITIONS
# ============================================================

combined_conditions = [
    "COMBINED_CPU11_MEM16",
    "COMBINED_HIGHCPU_MEM16",
]

combined_df = df[
    df["source_condition"].isin(
        combined_conditions
    )
].copy()

combined_summary = (
    combined_df.groupby("source_condition")
               .agg(
                   trials=("request_to_ready_ms", "count"),
                   cpu_mean=("pre_scale_node_cpu_pct", "mean"),
                   memory_mean=("pre_scale_node_memory_pct", "mean"),
                   latency_mean=("request_to_ready_ms", "mean"),
                   latency_median=("request_to_ready_ms", "median"),
                   latency_std=("request_to_ready_ms", "std"),
               )
               .round(2)
)

print("\n" + "=" * 70)
print("COMBINED CPU + MEMORY SUMMARY")
print("=" * 70)

print(combined_summary.to_string())

combined_summary.to_csv(
    OUTDIR / "v3_eda_combined_summary.csv"
)

combined_box_data = [
    combined_df.loc[
        combined_df["source_condition"] == condition,
        "request_to_ready_ms"
    ]
    for condition in combined_conditions
]

plt.figure(figsize=(9, 6))

plt.boxplot(
    combined_box_data,
    tick_labels=combined_conditions,
    showmeans=True
)

plt.xlabel("Combined Condition")
plt.ylabel("Request-to-Ready Latency (ms)")
plt.title("Combined CPU + Memory Startup Latency")
plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTDIR / "07_combined_cpu_memory.png",
    dpi=200
)

plt.close()

# ============================================================
# 10. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

correlation_columns = [
    "pre_scale_node_cpu_pct",
    "pre_scale_node_memory_pct",
    "request_to_ready_ms",
]

pearson = df[
    correlation_columns
].corr(method="pearson")

spearman = df[
    correlation_columns
].corr(method="spearman")

print("\nPearson correlation:")
print(pearson.round(4))

print("\nSpearman correlation:")
print(spearman.round(4))

pearson.to_csv(
    OUTDIR / "v3_eda_pearson_correlation.csv"
)

spearman.to_csv(
    OUTDIR / "v3_eda_spearman_correlation.csv"
)

# ============================================================
# 11. CPU SWEEP CORRELATION
# ============================================================

cpu_pearson = cpu_df[
    [
        "pre_scale_node_cpu_pct",
        "request_to_ready_ms"
    ]
].corr(
    method="pearson"
).iloc[0, 1]

cpu_spearman = cpu_df[
    [
        "pre_scale_node_cpu_pct",
        "request_to_ready_ms"
    ]
].corr(
    method="spearman"
).iloc[0, 1]

print("\nCPU SWEEP correlations:")
print(
    "Pearson :",
    round(cpu_pearson, 4)
)

print(
    "Spearman:",
    round(cpu_spearman, 4)
)

# ============================================================
# 12. OUTLIER SCREENING - DO NOT DELETE
# ============================================================

q1 = df["request_to_ready_ms"].quantile(0.25)
q3 = df["request_to_ready_ms"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[
    (df["request_to_ready_ms"] < lower) |
    (df["request_to_ready_ms"] > upper)
].copy()

print("\n" + "=" * 70)
print("OUTLIER SCREENING")
print("=" * 70)

print(
    "Q1   :",
    round(q1, 2)
)

print(
    "Q3   :",
    round(q3, 2)
)

print(
    "IQR  :",
    round(iqr, 2)
)

print(
    "Lower:",
    round(lower, 2)
)

print(
    "Upper:",
    round(upper, 2)
)

print(
    "\nPotential IQR outlier rows:",
    len(outliers)
)

if len(outliers) > 0:

    print(
        outliers[
            [
                "source_condition",
                "trial",
                "pre_scale_node_cpu_pct",
                "pre_scale_node_memory_pct",
                "request_to_ready_ms",
            ]
        ].sort_values(
            "request_to_ready_ms",
            ascending=False
        ).to_string(index=False)
    )

outliers.to_csv(
    OUTDIR / "v3_eda_potential_outliers.csv",
    index=False
)

# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(
    f"EDA files saved in: {OUTDIR.resolve()}"
)

print("\nGenerated plots:")

plot_files = sorted(
    OUTDIR.glob("*.png")
)

for file in plot_files:
    print(" -", file.name)

print("\nGenerated tables:")

csv_files = sorted(
    OUTDIR.glob("*.csv")
)

for file in csv_files:
    print(" -", file.name)

print("=" * 70)
