import numpy as np
import pandas as pd

raw = pd.read_csv("v3_master_150_raw.csv")
saved = pd.read_csv("v3_model_v2_features_150.csv")

node_cpu = pd.to_numeric(raw["pre_scale_node_cpu_pct"].astype(str).str.replace("%", "", regex=False))
node_mem = pd.to_numeric(raw["pre_scale_node_memory_pct"].astype(str).str.replace("%", "", regex=False))

cluster = pd.DataFrame(
    raw["pre_scale_all_nodes_json"].apply(
        lambda x: __import__("json").loads(x)
    ).apply(
        lambda state: {
            "cpu": np.mean([
                float(n["cpu_pct"].replace("%", "")) if isinstance(n.get("cpu_pct"), str) else float(n["cpu_pct"])
                if "cpu_pct" in n else float(n["cpu_m"]) / 16000 * 100
                for name, n in state.items() if "worker" in name.lower()
            ]),
            "memory": np.mean([
                float(n["memory_pct"].replace("%", "")) if isinstance(n.get("memory_pct"), str) else float(n["memory_pct"])
                if "memory_pct" in n else float(n["memory_mi"]) / (7970460 / 1024) * 100
                for name, n in state.items() if "worker" in name.lower()
            ])
        }
    ).tolist()
)

checks = {
    "node_cpu_delta_mean": node_cpu - cluster["cpu"] - saved["node_cpu_delta_mean"],
    "node_memory_delta_mean": node_mem - cluster["memory"] - saved["node_memory_delta_mean"],
    "node_cpu_x_memory": node_cpu * node_mem - saved["node_cpu_x_memory"],
    "cluster_cpu_x_memory": cluster["cpu"] * cluster["memory"] - saved["cluster_cpu_x_memory"],
}

for name, diff in checks.items():
    print(f"{name}: max absolute difference = {np.abs(diff).max():.12g}")
