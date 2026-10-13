import pandas as pd
from pathlib import Path


FILE = Path("v3_master_120_raw.csv")

EXPECTED_COUNTS = {
    "CPU_SWEEP": 40,
    "CACHE_COLD": 10,
    "CACHE_WARM": 10,
    "COMBINED_CPU06_08_MEM16_18": 10,
    "COMBINED_CPU06_08_MEM16_18_COLD": 10,
    "COMBINED_CPU11_MEM16": 10,
    "COMBINED_HIGHCPU_MEM16": 10,
    "MEM16": 10,
    "MEM23": 10,
}

NEW_WARM = "COMBINED_CPU06_08_MEM16_18"
NEW_COLD = "COMBINED_CPU06_08_MEM16_18_COLD"

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
    print(" StartScale V3 MASTER 120-ROW AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    if not FILE.exists():
        raise FileNotFoundError(
            f"File not found: {FILE}"
        )

    df = pd.read_csv(FILE)

    print(
        f"File : {FILE}"
    )

    print(
        f"Rows : {len(df)}"
    )

    print(
        f"Cols : {len(df.columns)}"
    )

    print()

    # ---------------------------------------------------------
    # 1. Row count
    # ---------------------------------------------------------

    assert len(df) == 120, (
        f"Expected 120 rows, found {len(df)}"
    )

    print(
        "[PASS] Row count = 120"
    )

    # ---------------------------------------------------------
    # 2. Required columns
    # ---------------------------------------------------------

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )

    print(
        "[PASS] Required columns"
    )

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

    missing = (
        df[critical_columns]
        .isna()
        .sum()
    )

    assert missing.sum() == 0, (
        f"Missing critical values:\n{missing}"
    )

    print(
        "[PASS] Critical columns have no missing values"
    )

    # ---------------------------------------------------------
    # 4. Positive latency
    # ---------------------------------------------------------

    latency = pd.to_numeric(
        df["request_to_ready_ms"],
        errors="coerce",
    )

    assert latency.notna().all(), (
        "Non-numeric latency values found"
    )

    assert (latency > 0).all(), (
        "Non-positive latency values found"
    )

    print(
        "[PASS] All request_to_ready_ms values are positive"
    )

    # ---------------------------------------------------------
    # 5. Scale transition
    # ---------------------------------------------------------

    assert (
        df["scale_from"] == 0
    ).all(), (
        "Some rows have scale_from != 0"
    )

    assert (
        df["scale_to"] == 1
    ).all(), (
        "Some rows have scale_to != 1"
    )

    print(
        "[PASS] All rows are 0 -> 1"
    )

    # ---------------------------------------------------------
    # 6. Unique pod names
    # ---------------------------------------------------------

    duplicate_pods = (
        df["pod_name"]
        .duplicated()
        .sum()
    )

    assert duplicate_pods == 0, (
        f"Duplicate pod names: {duplicate_pods}"
    )

    print(
        "[PASS] Pod names are unique"
    )

    # ---------------------------------------------------------
    # 7. Unique scale timestamps
    # ---------------------------------------------------------

    duplicate_times = (
        df["scale_request_time"]
        .duplicated()
        .sum()
    )

    assert duplicate_times == 0, (
        f"Duplicate scale_request_time values: "
        f"{duplicate_times}"
    )

    print(
        "[PASS] Scale request timestamps are unique"
    )

    # ---------------------------------------------------------
    # 8. Source-condition counts
    # ---------------------------------------------------------

    counts = (
        df.groupby("source_condition")
        .size()
        .sort_index()
    )

    print()
    print(
        "Rows by source_condition:"
    )

    print(
        counts.to_string()
    )

    print()

    assert (
        set(counts.index)
        == set(EXPECTED_COUNTS.keys())
    ), (
        "Unexpected source_condition values found"
    )

    for condition, expected in EXPECTED_COUNTS.items():

        actual = int(
            counts.loc[condition]
        )

        assert actual == expected, (
            f"{condition}: expected {expected}, "
            f"found {actual}"
        )

    print(
        "[PASS] All source conditions have expected row counts"
    )

    # ---------------------------------------------------------
    # 9. Warm matched condition
    # ---------------------------------------------------------

    warm = df[
        df["source_condition"]
        == NEW_WARM
    ].copy()

    assert len(warm) == 10

    assert (
        warm["image_cache_state"]
        .astype(str)
        .str.lower()
        .eq("warm")
        .all()
    )

    warm_cpu = pct_to_float(
        warm["pre_scale_node_cpu_pct"]
    )

    warm_mem = pct_to_float(
        warm["pre_scale_node_memory_pct"]
    )

    print()
    print(
        "Warm matched condition:"
    )

    print(
        f"  Rows          : {len(warm)}"
    )

    print(
        f"  CPU min/max   : "
        f"{warm_cpu.min():.1f}% / "
        f"{warm_cpu.max():.1f}%"
    )

    print(
        f"  Memory min/max: "
        f"{warm_mem.min():.1f}% / "
        f"{warm_mem.max():.1f}%"
    )

    print(
        f"  Mean latency  : "
        f"{warm['request_to_ready_ms'].mean():.2f} ms"
    )

    # ---------------------------------------------------------
    # 10. Cold matched condition
    # ---------------------------------------------------------

    cold = df[
        df["source_condition"]
        == NEW_COLD
    ].copy()

    assert len(cold) == 10

    assert (
        cold["image_cache_state"]
        .astype(str)
        .str.lower()
        .eq("cold")
        .all()
    )

    cold_cpu = pct_to_float(
        cold["pre_scale_node_cpu_pct"]
    )

    cold_mem = pct_to_float(
        cold["pre_scale_node_memory_pct"]
    )

    print()
    print(
        "Cold matched condition:"
    )

    print(
        f"  Rows          : {len(cold)}"
    )

    print(
        f"  CPU min/max   : "
        f"{cold_cpu.min():.1f}% / "
        f"{cold_cpu.max():.1f}%"
    )

    print(
        f"  Memory min/max: "
        f"{cold_mem.min():.1f}% / "
        f"{cold_mem.max():.1f}%"
    )

    print(
        f"  Mean latency  : "
        f"{cold['request_to_ready_ms'].mean():.2f} ms"
    )

    # ---------------------------------------------------------
    # 11. Warm vs cold summary
    # ---------------------------------------------------------

    warm_mean = (
        warm["request_to_ready_ms"]
        .mean()
    )

    cold_mean = (
        cold["request_to_ready_ms"]
        .mean()
    )

    difference = (
        cold_mean - warm_mean
    )

    print()
    print(
        "Matched warm vs cold:"
    )

    print(
        f"  Warm mean : {warm_mean:.2f} ms"
    )

    print(
        f"  Cold mean : {cold_mean:.2f} ms"
    )

    print(
        f"  Difference: {difference:.2f} ms"
    )

    # ---------------------------------------------------------
    # 12. Cache counts
    # ---------------------------------------------------------

    cache_counts = (
        df["image_cache_state"]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Cache state counts:"
    )

    print(
        cache_counts.to_string()
    )

    # ---------------------------------------------------------
    # 13. Duplicate experimental keys
    # ---------------------------------------------------------

    key_columns = [
        "source_condition",
        "trial",
        "scale_request_time",
        "pod_name",
    ]

    duplicate_experiments = (
        df.duplicated(
            subset=key_columns
        )
        .sum()
    )

    assert duplicate_experiments == 0, (
        f"Duplicate experimental rows: "
        f"{duplicate_experiments}"
    )

    print()
    print(
        "[PASS] No duplicate experimental rows"
    )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "AUDIT STATUS: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()