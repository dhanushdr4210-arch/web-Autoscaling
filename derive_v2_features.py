import pandas as pd
import json
import numpy as np


INPUT = "v3_master_150_raw.csv"
OUTPUT = "v3_master_150_v2_features.csv"


def to_float(value):
    """Convert values such as 6%, 0.5, or '6%' to float."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        value = value.strip()

        if value.endswith("%"):
            value = value[:-1]

        try:
            return float(value)
        except ValueError:
            return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def node_metrics(node):
    """Return CPU% and memory% from either metrics format."""

    # Format using already-calculated percentages
    if "cpu_pct" in node and "memory_pct" in node:
        return (
            to_float(node["cpu_pct"]),
            to_float(node["memory_pct"])
        )

    # Format using raw CPU millicores and memory MiB
    if "cpu_m" in node and "memory_mi" in node:
        cpu_pct = float(node["cpu_m"]) / 160.0
        memory_pct = float(node["memory_mi"]) / 79.7046

        return cpu_pct, memory_pct

    return np.nan, np.nan
df = pd.read_csv(INPUT)

rows = []
errors = []

for idx, row in df.iterrows():

    try:
        nodes = json.loads(row["pre_scale_all_nodes_json"])

        workers = {
            name: data
            for name, data in nodes.items()
            if "worker" in name
        }

        worker_cpu = []
        worker_memory = []

        for name, data in workers.items():

            cpu_pct, memory_pct = node_metrics(data)

            if not np.isnan(cpu_pct):
                worker_cpu.append(cpu_pct)

            if not np.isnan(memory_pct):
                worker_memory.append(memory_pct)

        worker_cpu = np.array(worker_cpu, dtype=float)
        worker_memory = np.array(worker_memory, dtype=float)

        # Node-level values for the node receiving the target pod.
        node_cpu = to_float(row["pre_scale_node_cpu_pct"])
        node_memory = to_float(row["pre_scale_node_memory_pct"])

        result = row.to_dict()

        # V2 derived features
        result["node_cpu_pct_v2"] = node_cpu
        result["node_memory_pct_v2"] = node_memory

        result["worker_cpu_mean"] = (
            worker_cpu.mean() if len(worker_cpu) else np.nan
        )

        result["worker_cpu_max"] = (
            worker_cpu.max() if len(worker_cpu) else np.nan
        )

        result["worker_cpu_std"] = (
            worker_cpu.std() if len(worker_cpu) else np.nan
        )

        result["worker_memory_mean"] = (
            worker_memory.mean() if len(worker_memory) else np.nan
        )

        result["worker_memory_max"] = (
            worker_memory.max() if len(worker_memory) else np.nan
        )

        result["worker_memory_std"] = (
            worker_memory.std() if len(worker_memory) else np.nan
        )

        result["node_cpu_x_memory"] = (
            node_cpu * node_memory
            if not np.isnan(node_cpu) and not np.isnan(node_memory)
            else np.nan
        )

        result["cluster_cpu_x_memory"] = (
            result["worker_cpu_mean"] * result["worker_memory_mean"]
            if not np.isnan(result["worker_cpu_mean"])
            and not np.isnan(result["worker_memory_mean"])
            else np.nan
        )

        rows.append(result)

    except Exception as exc:

        errors.append({
            "row": idx,
            "condition": row.get("source_condition", "UNKNOWN"),
            "error": str(exc)
        })

        result = row.to_dict()

        # Keep the raw row even if feature extraction fails.
        for col in [
            "node_cpu_pct_v2",
            "node_memory_pct_v2",
            "worker_cpu_mean",
            "worker_cpu_max",
            "worker_cpu_std",
            "worker_memory_mean",
            "worker_memory_max",
            "worker_memory_std",
            "node_cpu_x_memory",
            "cluster_cpu_x_memory",
        ]:
            result[col] = np.nan

        rows.append(result)


out = pd.DataFrame(rows)

out.to_csv(OUTPUT, index=False)

print("\n=== V2 FEATURE EXTRACTION COMPLETE ===")
print(f"Input rows : {len(df)}")
print(f"Output rows: {len(out)}")
print(f"Errors     : {len(errors)}")

print("\n=== MISSINGNESS ===")

feature_cols = [
    "node_cpu_pct_v2",
    "node_memory_pct_v2",
    "worker_cpu_mean",
    "worker_cpu_max",
    "worker_cpu_std",
    "worker_memory_mean",
    "worker_memory_max",
    "worker_memory_std",
    "node_cpu_x_memory",
    "cluster_cpu_x_memory",
]

print(
    out[feature_cols]
    .isna()
    .sum()
    .to_string()
)

print("\n=== CONDITION COUNTS ===")

print(
    out["source_condition"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n=== OUTPUT ===")
print(OUTPUT)

if errors:
    print("\n=== EXTRACTION ERRORS ===")

    for error in errors:
        print(error)