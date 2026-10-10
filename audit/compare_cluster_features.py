import json
import numpy as np
import pandas as pd

raw = pd.read_csv("v3_master_150_raw.csv")
saved = pd.read_csv("v3_model_v2_features_150.csv")

def parse_percent(value):
    if pd.isna(value):
        return np.nan
    return float(str(value).strip().removesuffix("%"))

def parse_cluster_state(value):
    state = json.loads(value)
    cpu_values, memory_values = [], []

    for name, node in state.items():
        if "worker" not in str(name).lower():
            continue

        if "cpu_m" in node:
            v = pd.to_numeric(node.get("cpu_m"), errors="coerce")
            cpu = v / 16000 * 100 if pd.notna(v) else np.nan
        else:
            cpu = parse_percent(node.get("cpu_pct"))

        if "memory_mi" in node:
            v = pd.to_numeric(node.get("memory_mi"), errors="coerce")
            mem = v / (7970460 / 1024) * 100 if pd.notna(v) else np.nan
        else:
            mem = parse_percent(node.get("memory_pct"))

        if pd.notna(cpu):
            cpu_values.append(cpu)
        if pd.notna(mem):
            memory_values.append(mem)

    return {
        "worker_cpu_mean": np.mean(cpu_values),
        "worker_cpu_max": np.max(cpu_values),
        "worker_cpu_std": np.std(cpu_values),
        "worker_memory_mean": np.mean(memory_values),
        "worker_memory_max": np.max(memory_values),
        "worker_memory_std": np.std(memory_values),
    }

recomputed = pd.DataFrame(
    raw["pre_scale_all_nodes_json"].apply(parse_cluster_state).tolist()
)

print("Rows checked:", len(raw))
print("Cluster feature comparison (max absolute difference):")
for col in recomputed.columns:
    diff = np.abs(recomputed[col] - saved[col])
    print(f"{col}: {diff.max():.12g}")
