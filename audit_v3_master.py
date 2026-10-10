import pandas as pd

FILE = "v3_master_100_raw.csv"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FILE)

print("=" * 70)
print("STARTSCALE V3 MASTER DATA QUALITY AUDIT")
print("=" * 70)

# ============================================================
# 0. REQUIRED COLUMNS CHECK
# ============================================================

required_columns = [
    "source_condition",
    "trial",
    "scale_from",
    "scale_to",
    "scale_request_time",
    "pod_name",
    "request_to_ready_ms",
    "pre_scale_node_cpu_pct",
    "pre_scale_node_memory_pct",
    "image_cache_state",
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

print("\n[0] REQUIRED COLUMN CHECK")

if missing_required:
    print("FAIL - Missing required columns:")
    for col in missing_required:
        print("  -", col)

    raise SystemExit(
        "\nAudit stopped because required columns are missing."
    )
else:
    print("PASS - All required columns are present")

# ============================================================
# 1. BASIC SHAPE
# ============================================================

print("\n[1] SHAPE")
print("Rows   :", len(df))
print("Columns:", len(df.columns))

# ============================================================
# 2. SOURCE CONDITION COUNTS
# ============================================================

print("\n[2] SOURCE CONDITION COUNTS")

print(
    df["source_condition"]
    .value_counts()
    .sort_index()
)

# ============================================================
# 3. CRITICAL COLUMN MISSING VALUES
# ============================================================

critical = [
    "source_condition",
    "trial",
    "scale_from",
    "scale_to",
    "scale_request_time",
    "pod_name",
    "request_to_ready_ms",
    "pre_scale_node_cpu_pct",
    "pre_scale_node_memory_pct",
    "image_cache_state",
]

print("\n[3] CRITICAL COLUMN MISSING VALUES")

missing_counts = df[critical].isna().sum()

print(missing_counts)

critical_missing_total = missing_counts.sum()

# ============================================================
# 4. DUPLICATE EXPERIMENTAL ROW CHECK
# ============================================================

print("\n[4] DUPLICATE EXPERIMENTAL ROW CHECK")

# CPU_SWEEP contains repeated trial numbers because it contains
# four separate CPU conditions. Therefore trial alone cannot
# identify a unique experiment.
#
# scale_request_time + pod_name identify the actual scale event.

dupe_key = [
    "source_condition",
    "trial",
    "scale_request_time",
    "pod_name",
]

dupes = df[
    df.duplicated(
        subset=dupe_key,
        keep=False
    )
]

if len(dupes) == 0:
    print("PASS - No duplicate experimental rows")
else:
    print("WARNING - Duplicate experimental rows found:")

    print(
        dupes[
            [
                "source_condition",
                "trial",
                "scale_request_time",
                "pod_name",
            ]
        ].to_string(index=False)
    )

# ============================================================
# 5. TRIAL COUNT CONSISTENCY
# ============================================================

print("\n[5] TRIAL COUNT CONSISTENCY")

trial_counts = (
    df.groupby("source_condition")["trial"]
      .nunique()
      .sort_index()
)

print(trial_counts)

print("\nExpected source-level trial counts:")

expected_trials = {
    "CACHE_COLD": 10,
    "CACHE_WARM": 10,
    "COMBINED_CPU11_MEM16": 10,
    "COMBINED_HIGHCPU_MEM16": 10,
    "CPU_SWEEP": 10,
    "MEM16": 10,
    "MEM23": 10,
}

trial_problems = []

for condition, expected in expected_trials.items():

    actual = trial_counts.get(condition, 0)

    if actual != expected:
        trial_problems.append(
            f"{condition}: expected {expected}, found {actual}"
        )

if trial_problems:

    print("WARNING - Trial-count differences:")

    for problem in trial_problems:
        print("  -", problem)

else:

    print("PASS - Trial counts are consistent")

# ============================================================
# 6. LATENCY SANITY
# ============================================================

print("\n[6] LATENCY SANITY")

latency = df["request_to_ready_ms"]

print("Minimum:", latency.min())
print("Maximum:", latency.max())
print("Mean   :", round(latency.mean(), 2))
print("Median :", round(latency.median(), 2))
print("Std    :", round(latency.std(), 2))

non_positive = (latency <= 0).sum()

missing_latency = latency.isna().sum()

print("Non-positive:", non_positive)
print("Missing     :", missing_latency)

# ============================================================
# 7. CPU / MEMORY SANITY
# ============================================================

print("\n[7] RESOURCE RANGE")

cpu = df["pre_scale_node_cpu_pct"]
memory = df["pre_scale_node_memory_pct"]

print(
    "CPU min/max:",
    cpu.min(),
    "/",
    cpu.max()
)

print(
    "MEM min/max:",
    memory.min(),
    "/",
    memory.max()
)

bad_cpu = df[
    (cpu < 0) |
    (cpu > 100)
]

bad_memory = df[
    (memory < 0) |
    (memory > 100)
]

print("Invalid CPU rows:", len(bad_cpu))
print("Invalid MEM rows:", len(bad_memory))

# ============================================================
# 8. SCALE TRANSITION
# ============================================================

print("\n[8] SCALE TRANSITION")

transition_counts = (
    df.groupby(
        ["scale_from", "scale_to"]
    )
    .size()
    .sort_index()
)

print(transition_counts)

invalid_transition = df[
    ~(
        (df["scale_from"] == 0) &
        (df["scale_to"] == 1)
    )
]

print(
    "\nInvalid 0 -> 1 transition rows:",
    len(invalid_transition)
)

# ============================================================
# 9. CACHE STATES
# ============================================================

print("\n[9] CACHE STATES")

print(
    df["image_cache_state"]
    .value_counts(dropna=False)
)

# ============================================================
# 10. ORIGINAL EXPERIMENT LABELS
# ============================================================

print("\n[10] ORIGINAL EXPERIMENT LABELS")

if "experiment_original" in df.columns:

    print(
        df["experiment_original"]
        .value_counts(dropna=False)
        .sort_index()
    )

else:

    print("experiment_original column not found")

# ============================================================
# 11. IMAGE CONSISTENCY
# ============================================================

print("\n[11] IMAGE / CACHE CONSISTENCY")

if "image" in df.columns:

    image_by_cache = (
        df.groupby("image_cache_state")["image"]
          .nunique()
    )

    print("Unique images by cache state:")
    print(image_by_cache)

else:

    print("Image column not found")

# ============================================================
# 12. CONDITION-LEVEL SUMMARY
# ============================================================

print("\n[12] CONDITION SUMMARY")

summary = (
    df.groupby("source_condition")
      .agg(
          trials=("request_to_ready_ms", "count"),

          cpu_mean=(
              "pre_scale_node_cpu_pct",
              "mean"
          ),

          cpu_min=(
              "pre_scale_node_cpu_pct",
              "min"
          ),

          cpu_max=(
              "pre_scale_node_cpu_pct",
              "max"
          ),

          mem_mean=(
              "pre_scale_node_memory_pct",
              "mean"
          ),

          mem_min=(
              "pre_scale_node_memory_pct",
              "min"
          ),

          mem_max=(
              "pre_scale_node_memory_pct",
              "max"
          ),

          latency_mean=(
              "request_to_ready_ms",
              "mean"
          ),

          latency_median=(
              "request_to_ready_ms",
              "median"
          ),

          latency_std=(
              "request_to_ready_ms",
              "std"
          ),

          latency_min=(
              "request_to_ready_ms",
              "min"
          ),

          latency_max=(
              "request_to_ready_ms",
              "max"
          ),
      )
      .round(2)
)

print(summary.to_string())

# ============================================================
# 13. POD UNIQUENESS
# ============================================================

print("\n[13] POD NAME UNIQUENESS")

pod_duplicates = df[
    df.duplicated(
        subset=["pod_name"],
        keep=False
    )
]

if len(pod_duplicates) == 0:

    print("PASS - Every pod name is unique")

else:

    print(
        "WARNING - Repeated pod names found:"
    )

    print(
        pod_duplicates[
            [
                "source_condition",
                "trial",
                "pod_name",
            ]
        ].to_string(index=False)
    )

# ============================================================
# 14. SCALE REQUEST TIMESTAMP UNIQUENESS
# ============================================================

print("\n[14] SCALE REQUEST TIMESTAMP UNIQUENESS")

timestamp_duplicates = df[
    df.duplicated(
        subset=["scale_request_time"],
        keep=False
    )
]

if len(timestamp_duplicates) == 0:

    print(
        "PASS - Every scale request timestamp is unique"
    )

else:

    print(
        "WARNING - Repeated scale request timestamps:"
    )

    print(
        timestamp_duplicates[
            [
                "source_condition",
                "trial",
                "scale_request_time",
                "pod_name",
            ]
        ].to_string(index=False)
    )

# ============================================================
# 15. OVERALL DECISION
# ============================================================

print("\n" + "=" * 70)
print("FINAL AUDIT DECISION")
print("=" * 70)

problems = []

# Row count
if len(df) != 100:
    problems.append(
        f"Expected 100 rows, found {len(df)}"
    )

# Critical missing
if critical_missing_total > 0:
    problems.append(
        "Critical-column missing values exist"
    )

# Experimental duplicates
if len(dupes) > 0:
    problems.append(
        "Duplicate experimental rows exist"
    )

# Trial counts
if trial_problems:
    problems.append(
        "Trial counts are inconsistent"
    )

# Latency
if non_positive > 0:
    problems.append(
        "Non-positive latency exists"
    )

# CPU
if len(bad_cpu) > 0:
    problems.append(
        "Invalid CPU percentage exists"
    )

# Memory
if len(bad_memory) > 0:
    problems.append(
        "Invalid memory percentage exists"
    )

# Transition
if len(invalid_transition) > 0:
    problems.append(
        "Invalid scale transition exists"
    )

# Pod names
if len(pod_duplicates) > 0:
    problems.append(
        "Duplicate pod names exist"
    )

# Timestamp
if len(timestamp_duplicates) > 0:
    problems.append(
        "Duplicate scale request timestamps exist"
    )

# ============================================================
# PRINT FINAL STATUS
# ============================================================

if problems:

    print("AUDIT STATUS: REVIEW REQUIRED")

    print("\nProblems found:")

    for problem in problems:
        print(" -", problem)

else:

    print("AUDIT STATUS: PASS")

    print("\nThe 100-row V3 master dataset passed")
    print("the current structural and measurement checks.")

print("=" * 70)