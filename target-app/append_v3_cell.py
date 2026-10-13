import pandas as pd
from pathlib import Path

OLD_MASTER = Path("v3_master_100_raw.csv")
NEW_CELL = Path("v3_combined_cpu06_08_mem16_18.csv")
OUTPUT = Path("v3_master_110_raw.csv")

SOURCE_CONDITION = "COMBINED_CPU06_08_MEM16_18"


def main():
    print("=" * 70)
    print(" StartScale V3 - Append New Experimental Cell")
    print("=" * 70)

    if not OLD_MASTER.exists():
        raise FileNotFoundError(
            f"Missing old master: {OLD_MASTER}"
        )

    if not NEW_CELL.exists():
        raise FileNotFoundError(
            f"Missing new cell dataset: {NEW_CELL}"
        )

    master = pd.read_csv(OLD_MASTER)
    new = pd.read_csv(NEW_CELL)

    print(f"Old master rows : {len(master)}")
    print(f"New cell rows   : {len(new)}")
    print()

    if len(master) != 100:
        raise ValueError(
            f"Expected old master to contain 100 rows, "
            f"but found {len(master)}."
        )

    if len(new) != 10:
        raise ValueError(
            f"Expected new cell to contain 10 rows, "
            f"but found {len(new)}."
        )

    required_new_columns = [
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
        col for col in required_new_columns
        if col not in new.columns
    ]

    if missing:
        raise ValueError(
            f"New dataset is missing columns: {missing}"
        )

    # ---------------------------------------------------------
    # Add master-only metadata columns.
    # ---------------------------------------------------------
    new["source_condition"] = SOURCE_CONDITION

    if "experiment_original" in master.columns:
        new["experiment_original"] = new["experiment"]

    if "cpu_num" in master.columns:
        new["cpu_num"] = pd.NA

    # ---------------------------------------------------------
    # Make sure every master column exists.
    # ---------------------------------------------------------
    for col in master.columns:
        if col not in new.columns:
            new[col] = pd.NA

    # Keep exactly the same column order as the old master.
    new = new[master.columns]

    # ---------------------------------------------------------
    # Duplicate protection.
    # ---------------------------------------------------------
    existing_timestamps = set(
        master["scale_request_time"].astype(str)
    )

    new_timestamps = set(
        new["scale_request_time"].astype(str)
    )

    overlap = existing_timestamps.intersection(
        new_timestamps
    )

    if overlap:
        raise ValueError(
            "Duplicate scale_request_time values detected:\n"
            + "\n".join(sorted(overlap))
        )

    # ---------------------------------------------------------
    # Basic sanity checks for the new condition.
    # ---------------------------------------------------------
    if not all(new["scale_from"] == 0):
        raise ValueError(
            "New cell contains scale_from values other than 0."
        )

    if not all(new["scale_to"] == 1):
        raise ValueError(
            "New cell contains scale_to values other than 1."
        )

    if not all(new["image_cache_state"].astype(str) == "warm"):
        raise ValueError(
            "New cell is not entirely warm-cache."
        )

    if new["request_to_ready_ms"].isna().any():
        raise ValueError(
            "New cell contains missing latency values."
        )

    # ---------------------------------------------------------
    # Append.
    # ---------------------------------------------------------
    combined = pd.concat(
        [master, new],
        ignore_index=True,
    )

    if len(combined) != 110:
        raise ValueError(
            f"Expected 110 total rows, got {len(combined)}."
        )

    combined.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("APPEND SUCCESS")
    print("-" * 70)
    print(f"Output dataset : {OUTPUT}")
    print(f"Total rows     : {len(combined)}")
    print()

    print("New condition:")
    print(
        combined[
            combined["source_condition"]
            == SOURCE_CONDITION
        ][
            [
                "source_condition",
                "trial",
                "request_to_ready_ms",
                "pre_scale_node_cpu_pct",
                "pre_scale_node_memory_pct",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "New condition mean latency: "
        f"{new['request_to_ready_ms'].mean():.2f} ms"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()