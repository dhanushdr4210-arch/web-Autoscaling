import pandas as pd
from pathlib import Path


OLD_MASTER = Path("v3_master_110_raw.csv")
NEW_CELL = Path("v3_combined_cpu06_08_mem16_18_cold.csv")
OUTPUT = Path("v3_master_120_raw.csv")

SOURCE_CONDITION = "COMBINED_CPU06_08_MEM16_18_COLD"


def main():

    print("=" * 70)
    print("StartScale V3 - Append Matched Cold-Cache Cell")
    print("=" * 70)

    # ---------------------------------------------------------
    # File checks
    # ---------------------------------------------------------

    if not OLD_MASTER.exists():
        raise FileNotFoundError(
            f"Missing old master: {OLD_MASTER}"
        )

    if not NEW_CELL.exists():
        raise FileNotFoundError(
            f"Missing new cold dataset: {NEW_CELL}"
        )

    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------

    master = pd.read_csv(
        OLD_MASTER
    )

    new = pd.read_csv(
        NEW_CELL
    )

    print(
        f"Old master rows : {len(master)}"
    )

    print(
        f"New cold rows   : {len(new)}"
    )

    print()

    # ---------------------------------------------------------
    # Expected sizes
    # ---------------------------------------------------------

    if len(master) != 110:
        raise ValueError(
            f"Expected 110 rows in old master, "
            f"found {len(master)}"
        )

    if len(new) != 10:
        raise ValueError(
            f"Expected 10 rows in new cold dataset, "
            f"found {len(new)}"
        )

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    required_columns = [
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
    ]

    missing = [
        col
        for col in required_columns
        if col not in new.columns
    ]

    if missing:
        raise ValueError(
            f"New dataset missing columns: {missing}"
        )

    # ---------------------------------------------------------
    # Check new cold dataset
    # ---------------------------------------------------------

    if not (
        new["scale_from"] == 0
    ).all():
        raise ValueError(
            "New cold dataset contains scale_from != 0"
        )

    if not (
        new["scale_to"] == 1
    ).all():
        raise ValueError(
            "New cold dataset contains scale_to != 1"
        )

    if not (
        new["image_cache_state"]
        .astype(str)
        .str.lower()
        .eq("cold")
        .all()
    ):
        raise ValueError(
            "New dataset is not entirely cold-cache."
        )

    if new[
        "request_to_ready_ms"
    ].isna().any():
        raise ValueError(
            "New cold dataset contains missing latency."
        )

    # ---------------------------------------------------------
    # Add master metadata
    # ---------------------------------------------------------

    new["source_condition"] = (
        SOURCE_CONDITION
    )

    if "experiment_original" in master.columns:
        new["experiment_original"] = (
            new["experiment"]
        )

    if "cpu_num" in master.columns:
        new["cpu_num"] = pd.NA

    # ---------------------------------------------------------
    # Ensure same columns/order
    # ---------------------------------------------------------

    for col in master.columns:

        if col not in new.columns:
            new[col] = pd.NA

    new = new[
        master.columns
    ]

    # ---------------------------------------------------------
    # Duplicate timestamp check
    # ---------------------------------------------------------

    old_times = set(
        master[
            "scale_request_time"
        ]
        .astype(str)
    )

    new_times = set(
        new[
            "scale_request_time"
        ]
        .astype(str)
    )

    overlap = (
        old_times
        .intersection(
            new_times
        )
    )

    if overlap:
        raise ValueError(
            "Duplicate scale_request_time values found:\n"
            + "\n".join(
                sorted(overlap)
            )
        )

    # ---------------------------------------------------------
    # Duplicate pod check
    # ---------------------------------------------------------

    old_pods = set(
        master["pod_name"]
        .astype(str)
    )

    new_pods = set(
        new["pod_name"]
        .astype(str)
    )

    pod_overlap = (
        old_pods
        .intersection(
            new_pods
        )
    )

    if pod_overlap:
        raise ValueError(
            "Duplicate pod names found:\n"
            + "\n".join(
                sorted(pod_overlap)
            )
        )

    # ---------------------------------------------------------
    # Append
    # ---------------------------------------------------------

    combined = pd.concat(
        [
            master,
            new,
        ],
        ignore_index=True,
    )

    if len(combined) != 120:
        raise ValueError(
            f"Expected 120 rows, "
            f"found {len(combined)}"
        )

    combined.to_csv(
        OUTPUT,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print(
        "APPEND SUCCESS"
    )

    print("-" * 70)

    print(
        f"Output dataset : {OUTPUT}"
    )

    print(
        f"Total rows     : {len(combined)}"
    )

    print()

    print(
        "New cold condition:"
    )

    print(
        new[
            [
                "source_condition",
                "trial",
                "request_to_ready_ms",
                "pre_scale_node_cpu_pct",
                "pre_scale_node_memory_pct",
                "image_cache_state",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Cold mean latency: "
        f"{new['request_to_ready_ms'].mean():.2f} ms"
    )

    print(
        "Cold median latency: "
        f"{new['request_to_ready_ms'].median():.2f} ms"
    )

    print()

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()