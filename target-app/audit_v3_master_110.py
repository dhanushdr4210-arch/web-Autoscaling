import pandas as pd
from pathlib import Path

FILE = Path("v3_master_110_raw.csv")

EXPECTED_COUNTS = {
    "CPU_SWEEP": 40,
    "CACHE_COLD": 10,
    "CACHE_WARM": 10,
    "COMBINED_CPU06_08_MEM16_18": 10,
    "COMBINED_CPU11_MEM16": 10,
    "COMBINED_HIGHCPU_MEM16": 10,
    "MEM16": 10,
    "MEM23": 10,
}

NEW_CONDITION = "COMBINED_CPU06_08_MEM16_18"

REQUIRED_COLUMNS = [
    "experiment",
    "trial",
    "scale_from",
    "scale_to",
    "scale_request_time",
    "pod_name",
    "node_name",
    "image",
    "image_id",
    "cpu_request",
    "memory_request",
    "configured_startup_delay_ms",
    "pre_scale_node_cpu_cores",
    "pre_scale_node_cpu_pct",
    "pre_scale_node_memory",
    "pre_scale_node_memory_pct",
    "pre_scale_all_nodes_json",
    "request_to_ready_ms",
    "image_cache_state",
    "container_started",
    "source_condition",
]


def pct_to_float(series):
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )


def main():
    print("=" * 70)
    print(" StartScale V3 MASTER 110-ROW AUDIT")
    print("=" * 70)

    if not FILE.exists():
        raise FileNotFoundError(
            f"File not found: {FILE}"
        )

    df = pd.read_csv(FILE)

    print(f"File : {FILE}")
    print(f"Rows : {len(df)}")
    print(f"Cols : {len(df.columns)}")
    print()

    # ---------------------------------------------------------
    # 1. Row count
    # ---------------------------------------------------------
    assert len(df) == 110, (
        f"Expected 110 rows, found {len(df)}"
    )

    print("[PASS] Row count = 110")

    # ---------------------------------------------------------
    # 2. Required columns
    # ---------------------------------------------------------
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )

    print("[PASS] Required columns")

    # ---------------------------------------------------------
    # 3. Critical missing values
    # ---------------------------------------------------------
    critical_columns = [
        "experiment",
        "trial",
        "scale_from",
        "scale_to",
        "scale_request_time",
        "pod_name",
        "node_name",
        "request_to_ready_ms",
        "image_cache_state",
        "source_condition",
    ]

    missing = df[critical_columns].isna().sum()

    assert missing.sum() == 0, (
        f"Missing values detected:\n{missing}"
    )

    print("[PASS] Critical columns have no missing values")

    # ---------------------------------------------------------
    # 4. Positive latency
    # ---------------------------------------------------------
    latency = pd.to_numeric(
        df["request_to_ready_ms"],
        errors="coerce",
    )

    assert latency.notna().all(), (
        "Non-numeric latency values found."
    )

    assert (latency > 0).all(), (
        "Non-positive latency values found."
    )

    print("[PASS] All request_to_ready_ms values are positive")

    # ---------------------------------------------------------
    # 5. Scale transition
    # ---------------------------------------------------------
    assert (df["scale_from"] == 0).all(), (
        "Some rows have scale_from != 0."
    )

    assert (df["scale_to"] == 1).all(), (
        "Some rows have scale_to != 1."
    )

    print("[PASS] All rows are 0 -> 1")

    # ---------------------------------------------------------
    # 6. Unique pod names
    # ---------------------------------------------------------
    duplicate_pods = df["pod_name"].duplicated().sum()

    assert duplicate_pods == 0, (
        f"Duplicate pod names: {duplicate_pods}"
    )

    print("[PASS] Pod names are unique")

    # ---------------------------------------------------------
    # 7. Unique timestamps
    # ---------------------------------------------------------
    duplicate_times = (
        df["scale_request_time"].duplicated().sum()
    )

    assert duplicate_times == 0, (
        f"Duplicate scale_request_time values: "
        f"{duplicate_times}"
    )

    print("[PASS] Scale request timestamps are unique")

    # ---------------------------------------------------------
    # 8. Source-condition counts
    # ---------------------------------------------------------
    counts = (
        df.groupby("source_condition")
        .size()
        .sort_index()
    )

    print()
    print("Rows by source_condition:")
    print(counts.to_string())
    print()

    assert set(counts.index) == set(EXPECTED_COUNTS.keys()), (
        "Unexpected source_condition values found."
    )

    for condition, expected in EXPECTED_COUNTS.items():
        actual = int(counts.loc[condition])

        assert actual == expected, (
            f"{condition}: expected {expected}, "
            f"found {actual}"
        )

    print(
        "[PASS] All source conditions have expected row counts"
    )

    # ---------------------------------------------------------
    # 9. New condition validation
    # ---------------------------------------------------------
    new = df[
        df["source_condition"] == NEW_CONDITION
    ].copy()

    assert len(new) == 10, (
        f"Expected 10 new-condition rows, "
        f"found {len(new)}"
    )

    trials = sorted(
        pd.to_numeric(
            new["trial"],
            errors="coerce",
        ).astype(int).tolist()
    )

    assert trials == list(range(1, 11)), (
        f"Unexpected trials: {trials}"
    )

    cpu = pct_to_float(
        new["pre_scale_node_cpu_pct"]
    )

    mem = pct_to_float(
        new["pre_scale_node_memory_pct"]
    )

    print()
    print("New condition summary:")
    print(f"  Rows           : {len(new)}")
    print(
        f"  CPU min/max    : "
        f"{cpu.min():.1f}% / {cpu.max():.1f}%"
    )
    print(
        f"  Memory min/max : "
        f"{mem.min():.1f}% / {mem.max():.1f}%"
    )
    print(
        f"  Latency mean   : "
        f"{new['request_to_ready_ms'].mean():.2f} ms"
    )
    print(
        f"  Latency median : "
        f"{new['request_to_ready_ms'].median():.2f} ms"
    )

    assert (
        new["image_cache_state"]
        .astype(str)
        .str.lower()
        .eq("warm")
        .all()
    ), "New condition is not entirely warm-cache."

    assert (new["scale_from"] == 0).all()
    assert (new["scale_to"] == 1).all()

    print("[PASS] New condition is valid")

    # ---------------------------------------------------------
    # 10. Cache-state counts
    # ---------------------------------------------------------
    cache_counts = (
        df["image_cache_state"]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    print()
    print("Cache state counts:")
    print(cache_counts.to_string())

    # ---------------------------------------------------------
    # 11. Duplicate experimental keys
    # ---------------------------------------------------------
    key_columns = [
        "source_condition",
        "trial",
        "scale_request_time",
        "pod_name",
    ]

    duplicate_experiments = df.duplicated(
        subset=key_columns
    ).sum()

    assert duplicate_experiments == 0, (
        f"Duplicate experimental rows: "
        f"{duplicate_experiments}"
    )

    print()
    print("[PASS] No duplicate experimental rows")

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("AUDIT STATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()