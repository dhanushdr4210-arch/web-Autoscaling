import csv
import json
import subprocess
import time
from datetime import datetime, timezone

DEPLOYMENT = "startscale-cold"
LABEL = "app=startscale-cold"

BASE_REPLICAS = 0
TARGET_REPLICAS = 1
TRIALS = 10

POLL_INTERVAL = 0.2
READY_TIMEOUT = 120

IMAGE = "localhost:5000/startscale-target:cold-v2"
OUTPUT = "startup_latency_cold_dataset.csv"

WORKERS = [
    "autoscaling-worker",
    "autoscaling-worker2",
    "autoscaling-worker3",
]


def run_cmd(args, check=True):
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


def kubectl(*args, check=True):
    return run_cmd(["kubectl", *args], check=check)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def get_pods():
    raw = kubectl("get", "pods", "-l", LABEL, "-o", "json")
    return json.loads(raw)["items"]


def get_ready_pods():
    ready = []

    for pod in get_pods():
        conditions = pod.get("status", {}).get("conditions", [])

        is_ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in conditions
        )

        if is_ready:
            ready.append(pod)

    return ready


def wait_for_ready_replicas(count, timeout=READY_TIMEOUT):
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        ready = get_ready_pods()

        if len(ready) >= count:
            return ready

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Timed out waiting for {count} Ready pod(s).")


def get_node_metrics():
    raw = kubectl("top", "nodes", "--no-headers")

    metrics = {}

    for line in raw.splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        node = parts[0]

        metrics[node] = {
            "cpu_cores": parts[1],
            "cpu_pct": parts[2],
            "memory": parts[3],
            "memory_pct": parts[4],
        }

    return metrics


def delete_cached_cold_image():
    print("Removing cold-v2 from worker image caches...")

    for worker in WORKERS:
        result = subprocess.run(
            [
                "docker",
                "exec",
                worker,
                "crictl",
                "rmi",
                IMAGE,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"  Removed from {worker}")
        else:
            print(f"  Not present on {worker}")


def get_pod_container_info(pod):
    containers = pod.get("status", {}).get("containerStatuses", [])

    if not containers:
        return {
            "image_id": "",
            "container_started": "",
        }

    c = containers[0]

    state = c.get("state", {})
    running = state.get("running", {})

    return {
        "image_id": c.get("imageID", ""),
        "container_started": running.get("startedAt", ""),
    }


def main():
    print("=" * 55)
    print(" StartScale Cold-Image Startup Measurement")
    print("=" * 55)
    print(f"Trials: {TRIALS}")
    print(f"Deployment: {DEPLOYMENT}")
    print(f"Image: {IMAGE}")
    print()

    kubectl("scale", "deployment", DEPLOYMENT, "--replicas=0")
    wait_for_ready_replicas(0)

    fieldnames = [
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

    rows = []

    for trial in range(1, TRIALS + 1):
        print()
        print(f"========== TRIAL {trial}/{TRIALS} ==========")

        # Ensure deployment is at zero before each trial.
        kubectl("scale", "deployment", DEPLOYMENT, "--replicas=0")
        wait_for_ready_replicas(0)

        # Remove cold image from every worker.
        delete_cached_cold_image()

        # Collect node metrics immediately before scale-up.
        metrics = get_node_metrics()

        print("Pre-scale node metrics:")

        for node, values in metrics.items():
            print(
                f"  {node}: "
                f"CPU={values['cpu_pct']} "
                f"MEM={values['memory_pct']}"
            )

        old_names = {
            pod["metadata"]["name"]
            for pod in get_pods()
        }

        # High-resolution local stopwatch.
        start_time = time.monotonic()
        request_time = utc_now()

        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            "--replicas=1",
        )

        new_pod = None

        deadline = time.monotonic() + READY_TIMEOUT

        while time.monotonic() < deadline:
            for pod in get_ready_pods():

                name = pod["metadata"]["name"]

                if name not in old_names:
                    new_pod = pod
                    break

            if new_pod is not None:
                break

            time.sleep(POLL_INTERVAL)

        if new_pod is None:
            raise TimeoutError(
                f"Trial {trial}: new cold pod did not become Ready."
            )

        latency_ms = (time.monotonic() - start_time) * 1000

        pod_name = new_pod["metadata"]["name"]
        node_name = new_pod["spec"].get("nodeName", "")

        container = new_pod["spec"]["containers"][0]

        cpu_request = (
            container.get("resources", {})
            .get("requests", {})
            .get("cpu", "")
        )

        memory_request = (
            container.get("resources", {})
            .get("requests", {})
            .get("memory", "")
        )

        env = {
            item["name"]: item.get("value", "")
            for item in container.get("env", [])
        }

        startup_delay = env.get("STARTUP_DELAY_MS", "")

        runtime_info = get_pod_container_info(new_pod)

        selected_metrics = metrics.get(
            node_name,
            {
                "cpu_cores": "",
                "cpu_pct": "",
                "memory": "",
                "memory_pct": "",
            },
        )

        print(f"New pod: {pod_name}")
        print(f"Node: {node_name}")
        print(f"Request -> Ready: {latency_ms:.2f} ms")

        rows.append(
            {
                "trial": trial,
                "scale_from": BASE_REPLICAS,
                "scale_to": TARGET_REPLICAS,
                "scale_request_time": request_time,
                "pod_name": pod_name,
                "node_name": node_name,
                "image": IMAGE,
                "image_id": runtime_info["image_id"],
                "cpu_request": cpu_request,
                "memory_request": memory_request,
                "configured_startup_delay_ms": startup_delay,
                "pre_scale_node_cpu_cores": selected_metrics["cpu_cores"],
                "pre_scale_node_cpu_pct": selected_metrics["cpu_pct"],
                "pre_scale_node_memory": selected_metrics["memory"],
                "pre_scale_node_memory_pct": selected_metrics["memory_pct"],
                "pre_scale_all_nodes_json": json.dumps(
                    metrics,
                    separators=(",", ":"),
                ),
                "request_to_ready_ms": round(latency_ms, 2),
                "image_cache_state": "cold",
                "container_started": runtime_info["container_started"],
            }
        )

        # Return to zero.
        kubectl("scale", "deployment", DEPLOYMENT, "--replicas=0")
        wait_for_ready_replicas(0)

        print("Returned to zero replicas.")

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 55)
    print(" Cold-image measurement complete")
    print(f"Dataset: {OUTPUT}")
    print("=" * 55)


if __name__ == "__main__":
    main()