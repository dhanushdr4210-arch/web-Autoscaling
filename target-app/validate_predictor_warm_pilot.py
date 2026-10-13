import json
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from kubernetes import client, config
from sklearn.ensemble import GradientBoostingRegressor


# ============================================================
# CONFIG
# ============================================================

DATASET = "v3_master_150_v3_spatial_features.csv"
TARGET = "request_to_ready_ms"

DEPLOYMENT = "startscale-event"
NAMESPACE = "default"
LABEL_SELECTOR = "app=startscale-event"

TARGET_NODE = "autoscaling-worker"
TARGET_IMAGE = "localhost:5000/startscale-target:v1"

TRIALS = 5
COOLDOWN_SECONDS = 5

GLOBAL_P90_ADJUSTMENT_MS = 576.94

FEATURES = [
    "node_cpu_pct_v2",
    "node_memory_pct_v2",
    "worker_cpu_mean",
    "worker_memory_mean",
    "worker_memory_max",
    "worker_memory_std",
    "node_cpu_x_memory",
    "cluster_cpu_x_memory",
    "cpu_spatial_imbalance",
    "image_cache_state",
]


# ============================================================
# KUBERNETES
# ============================================================

print()
print("=" * 80)
print("STARTSCALE PREDICTOR VALIDATION - WARM PILOT")
print("=" * 80)

config.load_kube_config()

core_api = client.CoreV1Api()
apps_api = client.AppsV1Api()
custom_api = client.CustomObjectsApi()


# ============================================================
# LOAD DATASET + TRAIN MODELS
# ============================================================

df = pd.read_csv(DATASET)

X = df[FEATURES].copy()
y = df[TARGET].astype(float)

X["image_cache_state"] = X["image_cache_state"].map(
    {"cold": 0, "warm": 1}
).fillna(X["image_cache_state"])

p50_model = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.50,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

p90_model = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.90,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

p50_model.fit(X, y)
p90_model.fit(X, y)

print()
print(f"Training dataset : {DATASET}")
print(f"Training samples  : {len(df)}")
print("Models ready.")


# ============================================================
# HELPERS
# ============================================================

def get_metrics():
    result = custom_api.list_cluster_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        plural="nodes",
    )

    return result["items"]


def parse_cpu(value):
    value = str(value)

    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000_000

    if value.endswith("u"):
        return float(value[:-1]) / 1_000_000

    if value.endswith("m"):
        return float(value[:-1]) / 1000

    return float(value)


def parse_memory(value):
    value = str(value)

    if value.endswith("Ki"):
        return float(value[:-2]) / 1024

    if value.endswith("Mi"):
        return float(value[:-2])

    if value.endswith("Gi"):
        return float(value[:-2]) * 1024

    if value.endswith("Ti"):
        return float(value[:-2]) * 1024 * 1024

    return float(value) / (1024 * 1024)


def get_node_capacity(node_name):
    nodes = core_api.list_node().items

    for node in nodes:
        if node.metadata.name == node_name:
            cpu = parse_cpu(
                node.status.capacity["cpu"]
            )

            memory = parse_memory(
                node.status.capacity["memory"]
            )

            return cpu, memory

    raise RuntimeError(
        f"Node not found: {node_name}"
    )


def get_live_state():
    metrics = get_metrics()

    node_cpu_pct = None
    node_memory_pct = None

    worker_cpu = []
    worker_memory = []

    target_cpu_pct = None
    target_memory_pct = None

    for item in metrics:

        name = item["metadata"]["name"]

        cpu = parse_cpu(
            item["usage"]["cpu"]
        )

        memory = parse_memory(
            item["usage"]["memory"]
        )

        capacity_cpu, capacity_memory = get_node_capacity(name)

        cpu_pct = (
            cpu / capacity_cpu
        ) * 100

        memory_pct = (
            memory / capacity_memory
        ) * 100

        if name == TARGET_NODE:

            target_cpu_pct = cpu_pct
            target_memory_pct = memory_pct

        if name.startswith("autoscaling-worker"):

            worker_cpu.append(cpu_pct)
            worker_memory.append(memory_pct)

    if target_cpu_pct is None:
        raise RuntimeError(
            "Target worker metrics not found."
        )

    worker_cpu = np.array(worker_cpu)
    worker_memory = np.array(worker_memory)

    worker_cpu_mean = float(
        np.mean(worker_cpu)
    )

    worker_cpu_max = float(
        np.max(worker_cpu)
    )

    worker_cpu_std = float(
        np.std(worker_cpu)
    )

    worker_memory_mean = float(
        np.mean(worker_memory)
    )

    worker_memory_max = float(
        np.max(worker_memory)
    )

    worker_memory_std = float(
        np.std(worker_memory)
    )

    node_cpu_x_memory = (
        target_cpu_pct *
        target_memory_pct
    )

    cluster_cpu_x_memory = (
        worker_cpu_mean *
        worker_memory_mean
    )

    cpu_spatial_imbalance = (
        worker_cpu_max -
        worker_cpu_mean
    ) * 100

    return {
        "node_cpu_pct_v2": target_cpu_pct,
        "node_memory_pct_v2": target_memory_pct,
        "worker_cpu_mean": worker_cpu_mean,
        "worker_memory_mean": worker_memory_mean,
        "worker_memory_max": worker_memory_max,
        "worker_memory_std": worker_memory_std,
        "node_cpu_x_memory": node_cpu_x_memory,
        "cluster_cpu_x_memory": cluster_cpu_x_memory,
        "cpu_spatial_imbalance": cpu_spatial_imbalance,
        "image_cache_state": 1,
    }


def get_pod():
    pods = core_api.list_namespaced_pod(
        namespace=NAMESPACE,
        label_selector=LABEL_SELECTOR,
    ).items

    pods = [
        p for p in pods
        if p.metadata.deletion_timestamp is None
    ]

    if not pods:
        return None

    return max(
        pods,
        key=lambda p: p.metadata.creation_timestamp
    )


def wait_for_ready(timeout=60):
    start = time.perf_counter()

    while True:

        pod = get_pod()

        if pod:

            conditions = pod.status.conditions or []

            ready = any(
                c.type == "Ready"
                and c.status == "True"
                for c in conditions
            )

            if ready:
                return pod

        if time.perf_counter() - start > timeout:
            raise TimeoutError(
                "Pod did not become Ready."
            )

        time.sleep(0.1)


def set_replicas(count):
    body = {
        "spec": {
            "replicas": count
        }
    }

    apps_api.patch_namespaced_deployment_scale(
        name=DEPLOYMENT,
        namespace=NAMESPACE,
        body=body,
    )


def ensure_zero():
    set_replicas(0)

    deadline = time.time() + 60

    while time.time() < deadline:

        pods = core_api.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=LABEL_SELECTOR,
        ).items

        active = [
            p for p in pods
            if p.metadata.deletion_timestamp is None
        ]

        if len(active) == 0:
            return

        time.sleep(0.2)

    raise TimeoutError(
        "Existing pod did not disappear."
    )


# ============================================================
# SAFETY CHECK
# ============================================================

print()
print("Checking deployment state...")

ensure_zero()

deployment = apps_api.read_namespaced_deployment(
    name=DEPLOYMENT,
    namespace=NAMESPACE,
)

print(
    "Deployment replicas:",
    deployment.spec.replicas
)

print(
    "Validation mode: WARM CACHE"
)

print(
    "Trials:",
    TRIALS
)


# ============================================================
# VALIDATION LOOP
# ============================================================

results = []

try:

    for trial in range(1, TRIALS + 1):

        print()
        print("-" * 80)
        print(f"TRIAL {trial}/{TRIALS}")
        print("-" * 80)

        ensure_zero()

        # ----------------------------------------------------
        # Capture live state BEFORE scale request
        # ----------------------------------------------------

        live_state = get_live_state()

        X_live = pd.DataFrame(
            [live_state],
            columns=FEATURES,
        )

        p50_prediction = float(
            p50_model.predict(X_live)[0]
        )

        raw_p90_prediction = float(
            p90_model.predict(X_live)[0]
        )

        calibrated_p90_prediction = (
            raw_p90_prediction +
            GLOBAL_P90_ADJUSTMENT_MS
        )

        print()
        print("LIVE FEATURES")

        for feature in FEATURES:
            print(
                f"{feature:<28}: "
                f"{live_state[feature]:.4f}"
            )

        print()
        print(
            f"P50 prediction              : "
            f"{p50_prediction:.2f} ms"
        )

        print(
            f"Raw P90 prediction          : "
            f"{raw_p90_prediction:.2f} ms"
        )

        print(
            f"Calibrated P90 prediction   : "
            f"{calibrated_p90_prediction:.2f} ms"
        )

        # ----------------------------------------------------
        # SCALE REQUEST TIMESTAMP
        # ----------------------------------------------------

        request_perf = time.perf_counter()
        request_epoch = time.time()

        set_replicas(1)

        # ----------------------------------------------------
        # WAIT FOR READY
        # ----------------------------------------------------

        pod = wait_for_ready()

        ready_perf = time.perf_counter()

        actual_ms = (
            ready_perf -
            request_perf
        ) * 1000

        pod_name = pod.metadata.name
        node_name = pod.spec.node_name

        print()
        print(
            f"Pod                     : {pod_name}"
        )

        print(
            f"Node                    : {node_name}"
        )

        print(
            f"Actual request→ready    : "
            f"{actual_ms:.2f} ms"
        )

        print(
            f"P50 prediction error    : "
            f"{actual_ms - p50_prediction:.2f} ms"
        )

        print(
            f"P90 margin               : "
            f"{calibrated_p90_prediction - actual_ms:.2f} ms"
        )

        results.append(
            {
                "trial": trial,
                "cache_state": "warm",
                "request_epoch": request_epoch,
                "pod_name": pod_name,
                "node_name": node_name,

                **live_state,

                "predicted_p50_ms":
                    p50_prediction,

                "predicted_p90_raw_ms":
                    raw_p90_prediction,

                "p90_adjustment_ms":
                    GLOBAL_P90_ADJUSTMENT_MS,

                "predicted_p90_calibrated_ms":
                    calibrated_p90_prediction,

                "actual_request_to_ready_ms":
                    actual_ms,

                "p50_error_ms":
                    actual_ms - p50_prediction,

                "p90_margin_ms":
                    calibrated_p90_prediction - actual_ms,
            }
        )

        # ----------------------------------------------------
        # Return to zero
        # ----------------------------------------------------

        ensure_zero()

        print(
            f"Cooldown: "
            f"{COOLDOWN_SECONDS} seconds"
        )

        time.sleep(COOLDOWN_SECONDS)

finally:

    print()
    print("SAFETY RESTORE")
    ensure_zero()


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

output = (
    "predictor_validation_warm_pilot.csv"
)

result_df.to_csv(
    output,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("WARM PILOT SUMMARY")
print("=" * 80)

print(
    f"Trials completed : "
    f"{len(result_df)}"
)

if len(result_df) > 0:

    actual = result_df[
        "actual_request_to_ready_ms"
    ]

    p50 = result_df[
        "predicted_p50_ms"
    ]

    p90 = result_df[
        "predicted_p90_calibrated_ms"
    ]

    mae = np.mean(
        np.abs(actual - p50)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - p50) ** 2
        )
    )

    p90_coverage = np.mean(
        actual <= p90
    ) * 100

    print(
        f"Actual mean          : "
        f"{actual.mean():.2f} ms"
    )

    print(
        f"P50 MAE              : "
        f"{mae:.2f} ms"
    )

    print(
        f"P50 RMSE             : "
        f"{rmse:.2f} ms"
    )

    print(
        f"P90 coverage         : "
        f"{p90_coverage:.2f}%"
    )

    print(
        f"Actual min           : "
        f"{actual.min():.2f} ms"
    )

    print(
        f"Actual max           : "
        f"{actual.max():.2f} ms"
    )

print()
print(
    f"Saved: {output}"
)

print()
print("=" * 80)
print("WARM PILOT COMPLETE")
print("=" * 80)