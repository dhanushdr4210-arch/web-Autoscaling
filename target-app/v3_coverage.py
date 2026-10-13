import pandas as pd

# ============================================================
# STARTSCALE V3 EXPERIMENTAL COVERAGE ANALYSIS
# ============================================================

FILE = "v3_master_100_raw.csv"

df = pd.read_csv(FILE)

print("=" * 90)
print("STARTSCALE V3 EXPERIMENTAL COVERAGE")
print("=" * 90)


# ============================================================
# 1. BASIC DATA CHECK
# ============================================================

print("\n[1] DATASET")

print("Rows   :", len(df))
print("Columns:", len(df.columns))

required = [
    "pre_scale_node_cpu_pct",
    "pre_scale_node_memory_pct",
    "image_cache_state",
    "request_to_ready_ms",
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    print("\nERROR - Missing required columns:")

    for col in missing:
        print(" -", col)

    raise SystemExit(
        "\nCoverage analysis stopped."
    )

print("PASS - Required columns are present")


# ============================================================
# 2. NUMERIC CONVERSION
# ============================================================

df["cpu"] = pd.to_numeric(
    df["pre_scale_node_cpu_pct"],
    errors="coerce"
)

df["memory"] = pd.to_numeric(
    df["pre_scale_node_memory_pct"],
    errors="coerce"
)

df["latency"] = pd.to_numeric(
    df["request_to_ready_ms"],
    errors="coerce"
)


# ============================================================
# 3. CACHE FEATURE
# ============================================================

df["cache_warm"] = (
    df["image_cache_state"]
    .astype(str)
    .str.lower()
    .eq("warm")
    .astype(int)
)


# ============================================================
# 4. CPU BINS
# ============================================================

def cpu_bin(x):

    if x <= 2:
        return "CPU_0_2"

    elif x <= 8:
        return "CPU_6_8"

    elif x <= 14:
        return "CPU_11_14"

    else:
        return "CPU_17_20"


df["cpu_bin"] = df["cpu"].apply(cpu_bin)


# ============================================================
# 5. MEMORY BINS
# ============================================================

def memory_bin(x):

    if x <= 4:
        return "MEM_2_4"

    elif x <= 18:
        return "MEM_16_18"

    else:
        return "MEM_22_23"


df["memory_bin"] = df["memory"].apply(memory_bin)


# ============================================================
# 6. CPU BIN COUNTS
# ============================================================

print("\n" + "=" * 90)
print("[2] CPU BIN COVERAGE")
print("=" * 90)

cpu_counts = (
    df["cpu_bin"]
    .value_counts()
    .sort_index()
)

print(cpu_counts.to_string())


# ============================================================
# 7. MEMORY BIN COUNTS
# ============================================================

print("\n" + "=" * 90)
print("[3] MEMORY BIN COVERAGE")
print("=" * 90)

memory_counts = (
    df["memory_bin"]
    .value_counts()
    .sort_index()
)

print(memory_counts.to_string())


# ============================================================
# 8. CPU × MEMORY COVERAGE
# ============================================================

print("\n" + "=" * 90)
print("[4] CPU × MEMORY COVERAGE")
print("=" * 90)

coverage = pd.crosstab(
    df["cpu_bin"],
    df["memory_bin"]
)

print(
    coverage.to_string()
)


# ============================================================
# 9. EXPECTED GRID
# ============================================================

cpu_levels = [
    "CPU_0_2",
    "CPU_6_8",
    "CPU_11_14",
    "CPU_17_20",
]

memory_levels = [
    "MEM_2_4",
    "MEM_16_18",
    "MEM_22_23",
]

coverage_complete = (
    coverage
    .reindex(
        index=cpu_levels,
        columns=memory_levels,
        fill_value=0
    )
)

print("\nExpected experimental grid:")
print(
    coverage_complete.to_string()
)


# ============================================================
# 10. FIND MISSING CPU × MEMORY CELLS
# ============================================================

print("\n" + "=" * 90)
print("[5] MISSING CPU × MEMORY CELLS")
print("=" * 90)

missing_cells = []

for cpu_level in cpu_levels:

    for memory_level in memory_levels:

        count = coverage_complete.loc[
            cpu_level,
            memory_level
        ]

        if count == 0:

            missing_cells.append(
                (
                    cpu_level,
                    memory_level
                )
            )

if missing_cells:

    print(
        "Missing combinations:"
    )

    for cpu_level, memory_level in missing_cells:

        print(
            f" - {cpu_level} + {memory_level}"
        )

else:

    print(
        "PASS - Every CPU × memory cell has data"
    )


# ============================================================
# 11. CACHE COVERAGE
# ============================================================

print("\n" + "=" * 90)
print("[6] CACHE STATE COVERAGE")
print("=" * 90)

cache_coverage = pd.crosstab(
    [
        df["cpu_bin"],
        df["memory_bin"],
    ],
    df["image_cache_state"]
)

cache_coverage = cache_coverage.reindex(
    columns=["cold", "warm"],
    fill_value=0
)

print(
    cache_coverage.to_string()
)


# ============================================================
# 12. CACHE MISSING COMBINATIONS
# ============================================================

print("\n" + "=" * 90)
print("[7] CACHE COVERAGE GAPS")
print("=" * 90)

cache_gaps = []

for cpu_level in cpu_levels:

    for memory_level in memory_levels:

        try:

            cold_count = cache_coverage.loc[
                (cpu_level, memory_level),
                "cold"
            ]

        except KeyError:

            cold_count = 0

        try:

            warm_count = cache_coverage.loc[
                (cpu_level, memory_level),
                "warm"
            ]

        except KeyError:

            warm_count = 0

        if cold_count == 0:

            cache_gaps.append(
                (
                    cpu_level,
                    memory_level,
                    "cold"
                )
            )

        if warm_count == 0:

            cache_gaps.append(
                (
                    cpu_level,
                    memory_level,
                    "warm"
                )
            )


if cache_gaps:

    for cpu_level, memory_level, cache_state in cache_gaps:

        print(
            f" - {cpu_level} + {memory_level} + {cache_state}"
        )

else:

    print(
        "PASS - Every CPU × memory cell has both cache states"
    )


# ============================================================
# 13. EXACT CPU / MEMORY / CACHE COMBINATIONS
# ============================================================

print("\n" + "=" * 90)
print("[8] EXACT EXPERIMENTAL COMBINATIONS")
print("=" * 90)

combo = (
    df.groupby(
        [
            "cpu_bin",
            "memory_bin",
            "image_cache_state",
        ]
    )
    .agg(
        trials=("latency", "count"),
        cpu_mean=("cpu", "mean"),
        memory_mean=("memory", "mean"),
        latency_mean=("latency", "mean"),
        latency_median=("latency", "median"),
        latency_std=("latency", "std"),
    )
    .reset_index()
)

combo = combo.sort_values(
    [
        "cpu_bin",
        "memory_bin",
        "image_cache_state",
    ]
)

print(
    combo.round(2).to_string(
        index=False
    )
)


# ============================================================
# 14. CONDITION-LEVEL COVERAGE
# ============================================================

print("\n" + "=" * 90)
print("[9] SOURCE CONDITION COVERAGE")
print("=" * 90)

condition_summary = (
    df.groupby("source_condition")
      .agg(
          trials=("latency", "count"),
          cpu_mean=("cpu", "mean"),
          memory_mean=("memory", "mean"),
          latency_mean=("latency", "mean"),
      )
      .round(2)
)

print(
    condition_summary.to_string()
)


# ============================================================
# 15. ACTUAL CPU / MEMORY RANGES
# ============================================================

print("\n" + "=" * 90)
print("[10] ACTUAL MEASURED RANGES")
print("=" * 90)

print(
    "CPU:",
    df["cpu"].min(),
    "to",
    df["cpu"].max(),
    "%"
)

print(
    "Memory:",
    df["memory"].min(),
    "to",
    df["memory"].max(),
    "%"
)

print(
    "Latency:",
    round(df["latency"].min(), 2),
    "to",
    round(df["latency"].max(), 2),
    "ms"
)


# ============================================================
# 16. SAVE COVERAGE TABLES
# ============================================================

coverage_complete.to_csv(
    "v3_cpu_memory_coverage.csv"
)

cache_coverage.to_csv(
    "v3_cache_coverage.csv"
)

combo.to_csv(
    "v3_exact_experimental_combinations.csv",
    index=False
)

condition_summary.to_csv(
    "v3_condition_coverage_summary.csv"
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("FINAL COVERAGE SUMMARY")
print("=" * 90)

print(
    f"Total trials          : {len(df)}"
)

print(
    f"CPU × Memory cells    : "
    f"{len(cpu_levels) * len(memory_levels)} possible"
)

print(
    f"Observed cells        : "
    f"{sum(
        coverage_complete.values.flatten() > 0
    )}"
)

print(
    f"Missing cells         : "
    f"{len(missing_cells)}"
)

print(
    f"Cache gaps            : "
    f"{len(cache_gaps)}"
)

if missing_cells:

    print(
        "\nSTATUS: EXPERIMENTAL GRID IS INCOMPLETE"
    )

else:

    print(
        "\nSTATUS: CPU × MEMORY GRID IS COMPLETE"
    )

print("\nSaved files:")

print(" - v3_cpu_memory_coverage.csv")
print(" - v3_cache_coverage.csv")
print(" - v3_exact_experimental_combinations.csv")
print(" - v3_condition_coverage_summary.csv")

print("=" * 90)