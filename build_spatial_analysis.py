import pandas as pd
import json
import numpy as np

DIST = "v3_spatial_distributed.csv"
CONC = "v3_spatial_concentrated.csv"
OUT = "v3_spatial_20_analysis.csv"

def process(path, condition):
    df = pd.read_csv(path)

    rows = []

    for _, r in df.iterrows():
        nodes = json.loads(r["pre_scale_all_nodes_json"])

        cpus = [
            nodes["autoscaling-worker"]["cpu_m"],
            nodes["autoscaling-worker2"]["cpu_m"],
            nodes["autoscaling-worker3"]["cpu_m"]
        ]

        mems = [
            nodes["autoscaling-worker"]["memory_mi"],
            nodes["autoscaling-worker2"]["memory_mi"],
            nodes["autoscaling-worker3"]["memory_mi"]
        ]

        rows.append({
            "condition": condition,
            "trial": int(r["trial"]),
            "worker_cpu_mean": np.mean(cpus),
            "worker_cpu_max": np.max(cpus),
            "worker_cpu_std": np.std(cpus, ddof=0),
            "worker_memory_mean": np.mean(mems),
            "worker_memory_max": np.max(mems),
            "worker_memory_std": np.std(mems, ddof=0),
            "cluster_worker_cpu_total": np.sum(cpus),
            "cpu_spatial_imbalance": np.max(cpus) - np.min(cpus),
            "request_to_ready_ms": float(r["request_to_ready_ms"]),
            "image_cache_state": r["image_cache_state"]
        })

    return pd.DataFrame(rows)

distributed = process(DIST, "distributed")
concentrated = process(CONC, "concentrated")

result = pd.concat([distributed, concentrated], ignore_index=True)

result.to_csv(OUT, index=False)

print("=" * 70)
print("SPATIAL 20-TRIAL ANALYSIS DATASET")
print("=" * 70)
print(f"Rows: {len(result)}")
print(f"Output: {OUT}")
print()
print(result.groupby("condition")["request_to_ready_ms"].agg(
    ["count", "mean", "median", "std", "min", "max"]
))
print()
print("Spatial CPU features:")
print(result.groupby("condition")[
    ["worker_cpu_mean", "worker_cpu_max",
     "worker_cpu_std", "cpu_spatial_imbalance"]
].mean())
