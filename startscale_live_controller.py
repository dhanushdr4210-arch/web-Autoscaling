import json
import numpy as np
import pandas as pd

from kubernetes import client, config
from sklearn.ensemble import GradientBoostingRegressor


# =============================================================================
# STARTSCALE LIVE SHADOW CONTROLLER
# =============================================================================

DATASET = "v3_master_150_v3_spatial_features.csv"

TARGET = "request_to_ready_ms"

TARGET_NODE_NAME = "autoscaling-worker"

TARGET_IMAGE = "localhost:5000/startscale-target:v1"

GLOBAL_P90_ADJUSTMENT_MS = 576.94

SAFETY_MARGIN_MS = 500


# =============================================================================
# MODEL FEATURES
# =============================================================================

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


# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 80)
print("STARTSCALE LIVE SHADOW CONTROLLER")
print("=" * 80)
print()


# =============================================================================
# LOAD KUBERNETES
# =============================================================================

print("Loading Kubernetes configuration...")

try:
    config.load_kube_config()
except Exception:
    config.load_incluster_config()


core_api = client.CoreV1Api()
custom_api = client.CustomObjectsApi()


# =============================================================================
# LOAD DATASET
# =============================================================================

print()
print("=" * 80)
print("LOADING STARTSCALE DATASET")
print("=" * 80)

df = pd.read_csv(DATASET)

print(f"Dataset : {DATASET}")
print(f"Samples : {len(df)}")
print(f"Features: {len(FEATURES)}")


# =============================================================================
# CHECK FEATURES
# =============================================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )


if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found in dataset."
    )


# =============================================================================
# CACHE ENCODING
# =============================================================================

print()
print("=" * 80)
print("CACHE ENCODING")
print("=" * 80)

print("cold -> 0")
print("warm -> 1")

df["image_cache_state"] = (
    df["image_cache_state"]
    .astype(str)
    .str.lower()
    .map(
        {
            "cold": 0,
            "warm": 1,
        }
    )
)

if df["image_cache_state"].isna().any():
    raise ValueError(
        "Invalid image_cache_state value found in dataset."
    )


# =============================================================================
# PREPARE TRAINING DATA
# =============================================================================

X = df[FEATURES].copy()

y = df[TARGET].astype(float).copy()


# =============================================================================
# TRAIN P50 MODEL
# =============================================================================

print()
print("=" * 80)
print("TRAINING P50 MODEL")
print("=" * 80)

p50_model = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.50,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

p50_model.fit(X, y)


# =============================================================================
# TRAIN P90 MODEL
# =============================================================================

print()
print("=" * 80)
print("TRAINING P90 MODEL")
print("=" * 80)

p90_model = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.90,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

p90_model.fit(X, y)


print()
print("MODEL READY")


# =============================================================================
# CPU PARSER
# =============================================================================

def parse_cpu_to_cores(cpu_value):
    """
    Convert Kubernetes CPU quantities to CPU cores.
    """

    if cpu_value is None:
        return 0.0

    value = str(cpu_value).strip()

    try:
        if value.endswith("n"):
            return float(value[:-1]) / 1_000_000_000

        if value.endswith("u"):
            return float(value[:-1]) / 1_000_000

        if value.endswith("m"):
            return float(value[:-1]) / 1000

        return float(value)

    except Exception:
        return 0.0


# =============================================================================
# MEMORY PARSER
# =============================================================================

def parse_memory_to_bytes(memory_value):
    """
    Convert Kubernetes memory quantities to bytes.
    """

    if memory_value is None:
        return 0.0

    value = str(memory_value).strip()

    units = {
        "Ki": 1024,
        "Mi": 1024 ** 2,
        "Gi": 1024 ** 3,
        "Ti": 1024 ** 4,
        "Pi": 1024 ** 5,
        "K": 1000,
        "M": 1000 ** 2,
        "G": 1000 ** 3,
        "T": 1000 ** 4,
        "P": 1000 ** 5,
    }

    for unit, multiplier in units.items():

        if value.endswith(unit):

            number = value[:-len(unit)]

            try:
                return float(number) * multiplier
            except Exception:
                return 0.0

    try:
        return float(value)
    except Exception:
        return 0.0


# =============================================================================
# NODE CAPACITY HELPERS
# =============================================================================

def get_node_cpu_capacity(node):
    """
    Return allocatable CPU in cores.
    """

    try:
        cpu = node.status.allocatable.get("cpu")

        return parse_cpu_to_cores(cpu)

    except Exception:
        return 0.0


def get_node_memory_capacity(node):
    """
    Return allocatable memory in bytes.
    """

    try:
        memory = node.status.allocatable.get("memory")

        return parse_memory_to_bytes(memory)

    except Exception:
        return 0.0


# =============================================================================
# GET LIVE KUBERNETES STATE
# =============================================================================

def get_live_cluster_state():

    print()
    print("=" * 80)
    print("READING LIVE KUBERNETES STATE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Get nodes
    # -------------------------------------------------------------------------

    nodes_response = core_api.list_node()

    nodes = nodes_response.items

    worker_nodes = []

    for node in nodes:

        node_name = node.metadata.name

        if "control-plane" not in node_name:
            worker_nodes.append(node)

    if not worker_nodes:
        raise RuntimeError(
            "No worker nodes detected."
        )

    print(
        f"Worker nodes detected: {len(worker_nodes)}"
    )

    # -------------------------------------------------------------------------
    # Metrics Server
    # -------------------------------------------------------------------------

    print()
    print("Reading Metrics Server...")

    metrics_response = custom_api.list_cluster_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        plural="nodes",
    )

    metrics_items = metrics_response.get(
        "items",
        []
    )

    if not metrics_items:
        raise RuntimeError(
            "Metrics Server returned no node metrics."
        )

    # -------------------------------------------------------------------------
    # Build metric lookup
    # -------------------------------------------------------------------------

    metrics_by_node = {}

    for item in metrics_items:

        node_name = item.get(
            "metadata",
            {}
        ).get(
            "name"
        )

        usage = item.get(
            "usage",
            {}
        )

        cpu = parse_cpu_to_cores(
            usage.get("cpu")
        )

        memory = parse_memory_to_bytes(
            usage.get("memory")
        )

        metrics_by_node[node_name] = {
            "cpu_cores": cpu,
            "memory_bytes": memory,
        }

    # -------------------------------------------------------------------------
    # Calculate worker percentages
    # -------------------------------------------------------------------------

    worker_cpu_pct = []
    worker_memory_pct = []

    worker_state = {}

    for node in worker_nodes:

        node_name = node.metadata.name

        metric = metrics_by_node.get(
            node_name,
            {}
        )

        cpu_cores = metric.get(
            "cpu_cores",
            0.0
        )

        memory_bytes = metric.get(
            "memory_bytes",
            0.0
        )

        cpu_capacity = get_node_cpu_capacity(
            node
        )

        memory_capacity = get_node_memory_capacity(
            node
        )

        if cpu_capacity > 0:

            cpu_pct = (
                cpu_cores /
                cpu_capacity
            ) * 100.0

        else:

            cpu_pct = 0.0

        if memory_capacity > 0:

            memory_pct = (
                memory_bytes /
                memory_capacity
            ) * 100.0

        else:

            memory_pct = 0.0

        worker_cpu_pct.append(
            cpu_pct
        )

        worker_memory_pct.append(
            memory_pct
        )

        worker_state[node_name] = {
            "cpu_cores": cpu_cores,
            "cpu_pct": cpu_pct,
            "memory_bytes": memory_bytes,
            "memory_pct": memory_pct,
        }

    worker_cpu = np.array(
        worker_cpu_pct,
        dtype=float
    )

    worker_memory = np.array(
        worker_memory_pct,
        dtype=float
    )

    # -------------------------------------------------------------------------
    # Worker aggregate statistics
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Target node
    # -------------------------------------------------------------------------

    target_node = None

    for node in worker_nodes:

        if node.metadata.name == TARGET_NODE_NAME:

            target_node = node
            break

    if target_node is None:

        raise RuntimeError(
            f"Target node '{TARGET_NODE_NAME}' not found."
        )

    target_node_name = target_node.metadata.name

    target_state = worker_state.get(
        target_node_name
    )

    if target_state is None:

        raise RuntimeError(
            "Target node metrics not available."
        )

    target_cpu_pct = float(
        target_state["cpu_pct"]
    )

    target_memory_pct = float(
        target_state["memory_pct"]
    )

    # -------------------------------------------------------------------------
    # Cluster aggregate percentages
    # -------------------------------------------------------------------------

    cluster_cpu_pct = float(
        np.sum(worker_cpu)
    )

    cluster_memory_pct = float(
        np.sum(worker_memory)
    )

    # -------------------------------------------------------------------------
    # Interaction features
    # -------------------------------------------------------------------------

    node_cpu_x_memory = (
        target_cpu_pct *
        target_memory_pct
    )

    cluster_cpu_x_memory = (
        cluster_cpu_pct *
        cluster_memory_pct
    )

    # -------------------------------------------------------------------------
    # Spatial imbalance
    # -------------------------------------------------------------------------

    if worker_cpu_mean > 0:

        cpu_spatial_imbalance = (
            worker_cpu_std /
            worker_cpu_mean
        ) * 100.0

    else:

        cpu_spatial_imbalance = 0.0

    # -------------------------------------------------------------------------
    # Cache detection
    # -------------------------------------------------------------------------

    print()
    print("Checking target image cache...")

    cached_images = []

    try:

        node_images = (
            target_node.status.images
            or []
        )

        for image_status in node_images:

            image_names = (
                image_status.names
                or []
            )

            for image_name in image_names:

                cached_images.append(
                    str(image_name)
                )

    except Exception as error:

        print(
            "WARNING: Could not read node image list."
        )

        print(
            f"Reason: {error}"
        )

    image_cache_state = int(
        TARGET_IMAGE in cached_images
    )

    if image_cache_state == 1:

        cache_label = "warm"

    else:

        cache_label = "cold"

    # -------------------------------------------------------------------------
    # Print live state
    # -------------------------------------------------------------------------

    print()
    print("-" * 80)
    print("LIVE CLUSTER STATE")
    print("-" * 80)

    print(
        f"Target node             : {target_node_name}"
    )

    print(
        f"Target node CPU         : "
        f"{target_cpu_pct:.2f}%"
    )

    print(
        f"Target node memory      : "
        f"{target_memory_pct:.2f}%"
    )

    print(
        f"Worker CPU mean         : "
        f"{worker_cpu_mean:.2f}%"
    )

    print(
        f"Worker CPU max          : "
        f"{worker_cpu_max:.2f}%"
    )

    print(
        f"Worker CPU std          : "
        f"{worker_cpu_std:.2f}%"
    )

    print(
        f"Worker memory mean      : "
        f"{worker_memory_mean:.2f}%"
    )

    print(
        f"Worker memory max       : "
        f"{worker_memory_max:.2f}%"
    )

    print(
        f"Worker memory std       : "
        f"{worker_memory_std:.2f}%"
    )

    print(
        f"Worker CPU total        : "
        f"{cluster_cpu_pct:.2f}%"
    )

    print(
        f"Worker memory total     : "
        f"{cluster_memory_pct:.2f}%"
    )

    print(
        f"CPU spatial imbalance   : "
        f"{cpu_spatial_imbalance:.2f}%"
    )

    print(
        f"Node CPU x memory       : "
        f"{node_cpu_x_memory:.2f}"
    )

    print(
        f"Cluster CPU x memory    : "
        f"{cluster_cpu_x_memory:.2f}"
    )

    print(
        f"Image cache state       : "
        f"{cache_label}"
    )

    print(
        f"Image cache encoding    : "
        f"{image_cache_state}"
    )

    print("-" * 80)

    return {
        "target_node": target_node_name,

        "node_cpu_pct_v2": target_cpu_pct,

        "node_memory_pct_v2": target_memory_pct,

        "worker_cpu_mean": worker_cpu_mean,

        "worker_cpu_max": worker_cpu_max,

        "worker_cpu_std": worker_cpu_std,

        "worker_memory_mean": worker_memory_mean,

        "worker_memory_max": worker_memory_max,

        "worker_memory_std": worker_memory_std,

        "node_cpu_x_memory": node_cpu_x_memory,

        "cluster_cpu_x_memory": cluster_cpu_x_memory,

        "cpu_spatial_imbalance": cpu_spatial_imbalance,

        "image_cache_state": image_cache_state,

        "image_cache_label": cache_label,

        "worker_state": worker_state,
    }


# =============================================================================
# BUILD MODEL INPUT
# =============================================================================

def build_model_input(state):

    row = {

        "node_cpu_pct_v2":
            state["node_cpu_pct_v2"],

        "node_memory_pct_v2":
            state["node_memory_pct_v2"],

        "worker_cpu_mean":
            state["worker_cpu_mean"],

        "worker_memory_mean":
            state["worker_memory_mean"],

        "worker_memory_max":
            state["worker_memory_max"],

        "worker_memory_std":
            state["worker_memory_std"],

        "node_cpu_x_memory":
            state["node_cpu_x_memory"],

        "cluster_cpu_x_memory":
            state["cluster_cpu_x_memory"],

        "cpu_spatial_imbalance":
            state["cpu_spatial_imbalance"],

        "image_cache_state":
            state["image_cache_state"],
    }

    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


# =============================================================================
# PREDICT STARTUP LATENCY
# =============================================================================

def predict_startup_latency(state):

    model_input = build_model_input(
        state
    )

    p50_prediction = float(
        p50_model.predict(
            model_input
        )[0]
    )

    raw_p90_prediction = float(
        p90_model.predict(
            model_input
        )[0]
    )

    calibrated_p90_prediction = (
        raw_p90_prediction +
        GLOBAL_P90_ADJUSTMENT_MS
    )

    return (
        p50_prediction,
        raw_p90_prediction,
        calibrated_p90_prediction,
    )


# =============================================================================
# SHADOW SCALING DECISION
# =============================================================================

def scaling_decision(
    demand_eta_ms,
    calibrated_p90_ms,
):

    required_lead_time_ms = (
        calibrated_p90_ms +
        SAFETY_MARGIN_MS
    )

    prepare_window_ms = (
        required_lead_time_ms +
        2000
    )

    if demand_eta_ms <= required_lead_time_ms:

        decision = "SCALE_NOW"

    elif demand_eta_ms <= prepare_window_ms:

        decision = "PREPARE_SCALE"

    else:

        decision = "WAIT"

    return (
        required_lead_time_ms,
        decision,
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    live_state = get_live_cluster_state()

    print()
    print("=" * 80)
    print("STARTSCALE STARTUP-LATENCY PREDICTION")
    print("=" * 80)

    (
        p50_prediction,
        raw_p90_prediction,
        calibrated_p90_prediction,
    ) = predict_startup_latency(
        live_state
    )

    print()
    print(
        f"P50 predicted delay       : "
        f"{p50_prediction:.2f} ms"
    )

    print(
        f"Raw P90 predicted delay   : "
        f"{raw_p90_prediction:.2f} ms"
    )

    print(
        f"P90 calibration adjustment: "
        f"{GLOBAL_P90_ADJUSTMENT_MS:.2f} ms"
    )

    print(
        f"Calibrated P90 delay      : "
        f"{calibrated_p90_prediction:.2f} ms"
    )

    print(
        f"Safety margin             : "
        f"{SAFETY_MARGIN_MS:.2f} ms"
    )

    required_lead_time = (
        calibrated_p90_prediction +
        SAFETY_MARGIN_MS
    )

    print(
        f"Required scale lead time  : "
        f"{required_lead_time:.2f} ms"
    )

    # -------------------------------------------------------------------------
    # Shadow decision tests
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SHADOW DECISION TEST")
    print("=" * 80)

    test_etas = [
        2000,
        4000,
        6000,
        8000,
        10000,
        12000,
    ]

    decision_rows = []

    for eta in test_etas:

        lead_time, decision = scaling_decision(
            eta,
            calibrated_p90_prediction,
        )

        print(
            f"Demand ETA {eta:>5} ms -> "
            f"{decision}"
        )

        decision_rows.append(
            {
                "demand_eta_ms": eta,
                "predicted_p50_ms":
                    p50_prediction,
                "predicted_p90_raw_ms":
                    raw_p90_prediction,
                "predicted_p90_calibrated_ms":
                    calibrated_p90_prediction,
                "safety_margin_ms":
                    SAFETY_MARGIN_MS,
                "required_lead_time_ms":
                    lead_time,
                "decision":
                    decision,
                "cache_state":
                    live_state["image_cache_label"],
            }
        )

    # -------------------------------------------------------------------------
    # Save prediction
    # -------------------------------------------------------------------------

    prediction_output = pd.DataFrame(
        [
            {
                "target_node":
                    live_state["target_node"],

                "target_image":
                    TARGET_IMAGE,

                "cache_state":
                    live_state["image_cache_label"],

                "image_cache_state":
                    live_state["image_cache_state"],

                "node_cpu_pct":
                    live_state["node_cpu_pct_v2"],

                "node_memory_pct":
                    live_state["node_memory_pct_v2"],

                "worker_cpu_mean":
                    live_state["worker_cpu_mean"],

                "worker_cpu_max":
                    live_state["worker_cpu_max"],

                "worker_cpu_std":
                    live_state["worker_cpu_std"],

                "worker_memory_mean":
                    live_state["worker_memory_mean"],

                "worker_memory_max":
                    live_state["worker_memory_max"],

                "worker_memory_std":
                    live_state["worker_memory_std"],

                "cpu_spatial_imbalance":
                    live_state["cpu_spatial_imbalance"],

                "node_cpu_x_memory":
                    live_state["node_cpu_x_memory"],

                "cluster_cpu_x_memory":
                    live_state["cluster_cpu_x_memory"],

                "predicted_p50_ms":
                    p50_prediction,

                "predicted_p90_raw_ms":
                    raw_p90_prediction,

                "p90_adjustment_ms":
                    GLOBAL_P90_ADJUSTMENT_MS,

                "predicted_p90_calibrated_ms":
                    calibrated_p90_prediction,

                "safety_margin_ms":
                    SAFETY_MARGIN_MS,

                "required_scale_lead_time_ms":
                    required_lead_time,

            }
        ]
    )

    prediction_output.to_csv(
        "startscale_live_prediction.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save shadow decisions
    # -------------------------------------------------------------------------

    decisions_output = pd.DataFrame(
        decision_rows
    )

    decisions_output.to_csv(
        "startscale_shadow_decisions.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        "Saved: startscale_live_prediction.csv"
    )

    print(
        "Saved: startscale_shadow_decisions.csv"
    )

    print()
    print("=" * 80)
    print("SHADOW MODE ONLY")
    print("=" * 80)

    print(
        "No Kubernetes replicas were changed."
    )

    print(
        "No live scaling action was executed."
    )

    print(
        "Controller validation completed."
    )

    print("=" * 80)
    print()