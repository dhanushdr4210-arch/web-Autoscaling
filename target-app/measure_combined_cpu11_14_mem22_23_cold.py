import csv
import json
import subprocess
import time
from datetime import datetime, timezone

import requests


# ============================================================
# CONFIG
# ============================================================

NAMESPACE = "default"

EVENT_DEPLOYMENT = "startscale-event"

IMAGE = "localhost:5000/startscale-target:cold-v2"

EXPERIMENT = "V3_Combined_CPU11_14_Mem22_23_COLD"

OUTPUT = "v3_combined_cpu11_14_mem22_23_cold.csv"

TRIALS = 10

TARGET_REPLICAS = 1

READY_TIMEOUT = 60

POLL_INTERVAL = 0.2

METRICS_URL = (
    "http://127.0.0.1:8001/apis/metrics.k8s.io/v1beta1/nodes"
)

NODE_CPU_CAPACITY_M = 16000.0

NODE_MEMORY_CAPACITY_KI = 7970460.0


# ============================================================
# COMMAND HELPERS
# ============================================================

def run(cmd):

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print("COMMAND FAILED:")
        print(" ".join(cmd))
        print(result.stdout)
        print(result.stderr)

        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}"
        )

    return result.stdout.strip()


def kubectl(*args):

    return run(
        ["kubectl", *args]
    )


# ============================================================
# METRICS
# ============================================================

def get_node_metrics():

    response = requests.get(
        METRICS_URL,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    result = {}

    for item in data["items"]:

        name = item["metadata"]["name"]

        cpu = item["usage"]["cpu"]
        memory = item["usage"]["memory"]

        if cpu.endswith("n"):
            cpu_m = float(cpu[:-1]) / 1_000_000

        elif cpu.endswith("u"):
            cpu_m = float(cpu[:-1]) / 1000

        elif cpu.endswith("m"):
            cpu_m = float(cpu[:-1])

        else:
            cpu_m = float(cpu) * 1000

        if memory.endswith("Ki"):
            memory_mi = float(
                memory[:-2]
            ) / 1024

        elif memory.endswith("Mi"):
            memory_mi = float(
                memory[:-2]
            )

        elif memory.endswith("Gi"):
            memory_mi = float(
                memory[:-2]
            ) * 1024

        else:
            memory_mi = float(memory) / (
                1024 * 1024
            )

        result[name] = {
            "cpu_m": cpu_m,
            "memory_mi": memory_mi,
            "cpu_pct": (
                cpu_m /
                NODE_CPU_CAPACITY_M *
                100
            ),
            "memory_pct": (
                memory_mi /
                (NODE_MEMORY_CAPACITY_KI / 1024) *
                100
            )
        }

    return result


# ============================================================
# REMOVE COLD IMAGE
# ============================================================

def remove_cold_image():

    print("\nRemoving cold-v2 from workers...")

    workers = [
        "autoscaling-worker",
        "autoscaling-worker2",
        "autoscaling-worker3"
    ]

    for worker in workers:

        result = subprocess.run(
            [
                "docker",
                "exec",
                worker,
                "crictl",
                "rmi",
                IMAGE
            ],
            capture_output=True,
            text=True
        )

        print(
            worker,
            "returncode=",
            result.returncode
        )

    print("Cold image removal attempted.")


# ============================================================
# VERIFY COLD IMAGE
# ============================================================

def verify_cold_image_absent():

    workers = [
        "autoscaling-worker",
        "autoscaling-worker2",
        "autoscaling-worker3"
    ]

    for worker in workers:

        result = subprocess.run(
            [
                "docker",
                "exec",
                worker,
                "crictl",
                "images"
            ],
            capture_output=True,
            text=True
        )

        if IMAGE.split(":")[-1] in result.stdout:

            print(
                "WARNING:",
                worker,
                "may still contain cold-v2"
            )


# ============================================================
# WAIT FOR REPLICA STATE
# ============================================================

def wait_for_replicas(target):

    deadline = time.time() + READY_TIMEOUT

    while time.time() < deadline:

        output = kubectl(
            "get",
            "deployment",
            EVENT_DEPLOYMENT,
            "-o",
            "json"
        )

        data = json.loads(output)

        replicas = data["status"].get(
            "readyReplicas",
            0
        )

        if replicas == target:

            return

        time.sleep(
            POLL_INTERVAL
        )

    raise TimeoutError(
        f"Deployment did not reach {target} ready replicas"
    )


# ============================================================
# WAIT FOR POD READY
# ============================================================

def wait_for_ready_pod():

    deadline = time.time() + READY_TIMEOUT

    while time.time() < deadline:

        output = kubectl(
            "get",
            "pods",
            "-l",
            "app=startscale-event",
            "-o",
            "json"
        )

        data = json.loads(output)

        for pod in data["items"]:

            conditions = pod["status"].get(
                "conditions",
                []
            )

            ready = any(
                c["type"] == "Ready"
                and c["status"] == "True"
                for c in conditions
            )

            if ready:

                return pod

        time.sleep(
            POLL_INTERVAL
        )

    raise TimeoutError(
        "Pod did not become Ready"
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 90)
print(
    "STARTSCALE V3 — "
    "CPU11–14 / MEM22–23 / COLD CACHE"
)
print("=" * 90)

print(
    "\nExperiment:",
    EXPERIMENT
)

print(
    "Output:",
    OUTPUT
)


# Ensure event deployment is scaled to zero

print("\nScaling event deployment to 0...")

kubectl(
    "scale",
    "deployment",
    EVENT_DEPLOYMENT,
    "--replicas=0"
)

wait_for_replicas(0)


# Verify calibration pod exists

print("\nChecking calibration...")

calibration = kubectl(
    "get",
    "pods",
    "-l",
    "app=startscale-calibration-cpu11-14-mem22-23",
    "-o",
    "wide"
)

print(calibration)


# ============================================================
# CSV
# ============================================================

fieldnames = [
    "experiment",
    "trial",
    "scale_from",
    "scale_to",
    "scale_request_time",
    "pod_name",
    "node_name",
    "image",
    "cpu_request",
    "memory_request",
    "configured_startup_delay_ms",
    "pre_scale_node_cpu_m",
    "pre_scale_node_memory_mi",
    "pre_scale_node_cpu_pct",
    "pre_scale_node_memory_pct",
    "pre_scale_all_nodes_json",
    "request_to_ready_ms",
    "image_cache_state",
    "container_started"
]


with open(
    OUTPUT,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    # ========================================================
    # TRIALS
    # ========================================================

    for trial in range(
        1,
        TRIALS + 1
    ):

        print("\n")
        print("=" * 70)
        print(
            f"TRIAL {trial}/{TRIALS}"
        )
        print("=" * 70)

        # --------------------------------------------
        # Make sure deployment is zero
        # --------------------------------------------

        kubectl(
            "scale",
            "deployment",
            EVENT_DEPLOYMENT,
            "--replicas=0"
        )

        wait_for_replicas(0)

        # --------------------------------------------
        # Remove cold image
        # --------------------------------------------

        remove_cold_image()

        verify_cold_image_absent()

        # --------------------------------------------
        # Capture pre-scale metrics
        # --------------------------------------------

        metrics = get_node_metrics()

        worker_name = "autoscaling-worker"

        if worker_name not in metrics:

            raise RuntimeError(
                "autoscaling-worker not found in metrics"
            )

        target = metrics[worker_name]

        print(
            f"Pre-scale CPU: "
            f"{target['cpu_m']:.2f}m "
            f"({target['cpu_pct']:.2f}%)"
        )

        print(
            f"Pre-scale memory: "
            f"{target['memory_mi']:.2f}Mi "
            f"({target['memory_pct']:.2f}%)"
        )

        # --------------------------------------------
        # Request scale
        # --------------------------------------------

        request_time = datetime.now(
            timezone.utc
        )

        request_epoch = time.perf_counter()

        kubectl(
            "scale",
            "deployment",
            EVENT_DEPLOYMENT,
            "--replicas=1"
        )

        # --------------------------------------------
        # Wait for Ready
        # --------------------------------------------

        pod = wait_for_ready_pod()

        ready_epoch = time.perf_counter()

        latency_ms = (
            ready_epoch -
            request_epoch
        ) * 1000

        pod_name = pod["metadata"]["name"]

        node_name = pod["spec"].get(
            "nodeName",
            ""
        )

        print(
            f"Pod: {pod_name}"
        )

        print(
            f"Node: {node_name}"
        )

        print(
            f"Request → Ready: "
            f"{latency_ms:.2f} ms"
        )

        # --------------------------------------------
        # Container start timestamp
        # --------------------------------------------

        container_started = ""

        container_statuses = (
            pod["status"].get(
                "containerStatuses",
                []
            )
        )

        if container_statuses:

            state = container_statuses[0].get(
                "state",
                {}
            )

            running = state.get(
                "running"
            )

            if running:

                container_started = (
                    running.get(
                        "startedAt",
                        ""
                    )
                )

        # --------------------------------------------
        # Save row
        # --------------------------------------------

        writer.writerow({

            "experiment":
                EXPERIMENT,

            "trial":
                trial,

            "scale_from":
                0,

            "scale_to":
                1,

            "scale_request_time":
                request_time.isoformat(),

            "pod_name":
                pod_name,

            "node_name":
                node_name,

            "image":
                IMAGE,

            "cpu_request":
                "2",

            "memory_request":
                "1600Mi",

            "configured_startup_delay_ms":
                3000,

            "pre_scale_node_cpu_m":
                round(
                    target["cpu_m"],
                    2
                ),

            "pre_scale_node_memory_mi":
                round(
                    target["memory_mi"],
                    2
                ),

            "pre_scale_node_cpu_pct":
                round(
                    target["cpu_pct"],
                    2
                ),

            "pre_scale_node_memory_pct":
                round(
                    target["memory_pct"],
                    2
                ),

            "pre_scale_all_nodes_json":
                json.dumps(
                    metrics
                ),

            "request_to_ready_ms":
                round(
                    latency_ms,
                    2
                ),

            "image_cache_state":
                "cold",

            "container_started":
                container_started
        })

        f.flush()

        # --------------------------------------------
        # Return to zero before next trial
        # --------------------------------------------

        kubectl(
            "scale",
            "deployment",
            EVENT_DEPLOYMENT,
            "--replicas=0"
        )

        wait_for_replicas(0)

        time.sleep(2)


print("\n")
print("=" * 90)
print("EXPERIMENT COMPLETE")
print("=" * 90)

print(
    f"Saved: {OUTPUT}"
)