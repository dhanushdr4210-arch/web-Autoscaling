import pandas as pd
import json
import numpy as np

INPUT = "v3_master_150_v2_features.csv"
OUTPUT = "v3_master_150_v3_spatial_features.csv"

df = pd.read_csv(INPUT)


def cpu_to_millicores(value):
    value = str(value).strip()

    if value.endswith("m"):
        return float(value[:-1])

    return float(value) * 1000.0


def memory_to_mi(value):
    value = str(value).strip()

    if value.endswith("Mi"):
        return float(value[:-2])

    if value.endswith("Gi"):
        return float(value[:-2]) * 1024.0

    return float(value)


def extract_cpu(info):

    if "cpu_m" in info:
        return float(info["cpu_m"])

    if "cpu_cores" in info:
        return cpu_to_millicores(info["cpu_cores"])

    raise KeyError(
        f"CPU field not found. Available fields: {list(info.keys())}"
    )


def extract_memory(info):

    if "memory_mi" in info:
        return float(info["memory_mi"])

    if "memory_mib" in info:
        return float(info["memory_mib"])

    if "memory" in info:
        return memory_to_mi(info["memory"])

    raise KeyError(
        f"Memory field not found. Available fields: {list(info.keys())}"
    )


def get_worker_metrics(node_json):

    data = json.loads(node_json)

    workers = []

    for node_name, info in data.items():

        if "control-plane" in node_name.lower():
            continue

        if not isinstance(info, dict):
            continue

        try:
            cpu = extract_cpu(info)
            memory = extract_memory(info)

            workers.append({
                "cpu": cpu,
                "memory": memory
            })

        except KeyError:
            continue

    if len(workers) != 3:
        raise ValueError(
            f"Expected 3 worker nodes, found {len(workers)}"
        )

    return workers


features = []

for _, row in df.iterrows():

    workers = get_worker_metrics(
        row["pre_scale_all_nodes_json"]
    )

    cpus = [x["cpu"] for x in workers]
    mems = [x["memory"] for x in workers]

    features.append({

        "worker_cpu_mean_spatial":
            np.mean(cpus),

        "worker_cpu_max_spatial":
            np.max(cpus),

        "worker_cpu_std_spatial":
            np.std(cpus, ddof=0),

        "worker_memory_mean_spatial":
            np.mean(mems),

        "worker_memory_max_spatial":
            np.max(mems),

        "worker_memory_std_spatial":
            np.std(mems, ddof=0),

        "cluster_worker_cpu_total_spatial":
            np.sum(cpus),

        "cpu_spatial_imbalance":
            np.max(cpus) - np.min(cpus)
    })


spatial = pd.DataFrame(features)

result = pd.concat(
    [
        df.reset_index(drop=True),
        spatial
    ],
    axis=1
)

result.to_csv(
    OUTPUT,
    index=False
)


print("=" * 70)
print("150-ROW SPATIAL FEATURE DATASET")
print("=" * 70)

print("Original shape :", df.shape)
print("New shape      :", result.shape)
print("Output         :", OUTPUT)

print()
print("New spatial features:")

for column in spatial.columns:
    print(" -", column)

print()
print("Spatial feature statistics:")

print(
    result[
        [
            "worker_cpu_mean_spatial",
            "worker_cpu_max_spatial",
            "worker_cpu_std_spatial",
            "cpu_spatial_imbalance"
        ]
    ].describe()
)

print()
print("First 5 rows:")

print(
    result[
        [
            "experiment",
            "trial",
            "worker_cpu_mean_spatial",
            "worker_cpu_max_spatial",
            "worker_cpu_std_spatial",
            "cpu_spatial_imbalance",
            "request_to_ready_ms"
        ]
    ].head().to_string(index=False)
)

print()
print("Last 5 rows:")

print(
    result[
        [
            "experiment",
            "trial",
            "worker_cpu_mean_spatial",
            "worker_cpu_max_spatial",
            "worker_cpu_std_spatial",
            "cpu_spatial_imbalance",
            "request_to_ready_ms"
        ]
    ].tail().to_string(index=False)
)

print()
print("SUCCESS: all 150 rows processed.")
