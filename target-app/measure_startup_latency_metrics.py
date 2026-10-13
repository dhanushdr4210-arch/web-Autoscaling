import csv
import json
import subprocess
import time
from datetime import datetime, timezone


DEPLOYMENT = "startscale-target"
LABEL = "app=startscale-target"

BASE_REPLICAS = 3
TARGET_REPLICAS = 4
TRIALS = 10

POLL_INTERVAL = 0.1
READY_TIMEOUT = 60

OUTPUT = "startup_latency_metrics_dataset.csv"


def kubectl(*args):
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def utc_now():
    return datetime.now(timezone.utc)


def get_pods():
    raw = kubectl(
        "get",
        "pods",
        "-l",
        LABEL,
        "-o",
        "json",
    )
    return json.loads(raw)["items"]


def get_ready_pods():
    pods = get_pods()
    ready = []

    for pod in pods:
        conditions = pod.get("status", {}).get("conditions", [])

        is_ready = any(
            c.get("type") == "Ready"
            and c.get("status") == "True"
            for c in conditions
        )

        if is_ready:
            ready.append(pod)

    return ready


def wait_for_ready_replicas(count):
    deadline = time.monotonic() + READY_TIMEOUT

    while time.monotonic() < deadline:
        pods = get_ready_pods()

        if len(pods) == count:
            return pods

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"Timed out waiting for {count} Ready replicas."
    )


def get_node_metrics():
    """
    Returns:
        {
            node_name: {
                "cpu_cores": "...",
                "cpu_pct": "...",
                "memory": "...",
                "memory_pct": "..."
            }
        }
    """
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


def get_container_info(pod):
    containers = pod.get("spec", {}).get("containers", [])

    if not containers:
        return "", "", "", ""

    container = containers[0]

    image = container.get("image", "")

    resources = container.get("resources", {})
    requests = resources.get("requests", {})

    cpu_request = requests.get("cpu", "")
    memory_request = requests.get("memory", "")

    startup_delay = ""

    for env in container.get("env", []):
        if env.get("name") == "STARTUP_DELAY_MS":
            startup_delay = env.get("value", "")
            break

    return image, cpu_request, memory_request, startup_delay


def find_new_pod(old_names):
    pods = get_pods()

    candidates = [
        p
        for p in pods
        if p["metadata"]["name"] not in old_names
    ]

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: p["metadata"].get(
            "creationTimestamp", ""
        )
    )

    return candidates[-1]


print("=" * 55)
print(" StartScale Metrics-Aware Startup Measurement")
print("=" * 55)
print(f"Trials: {TRIALS}")
print(f"Baseline replicas: {BASE_REPLICAS}")
print(f"Target replicas:   {TARGET_REPLICAS}")
print()

# Confirm Metrics Server works.
print("Checking Metrics Server...")
initial_metrics = get_node_metrics()

if not initial_metrics:
    raise RuntimeError(
        "kubectl top nodes returned no metrics."
    )

print(
    f"Metrics available for {len(initial_metrics)} nodes."
)
print()

fieldnames = [
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
    "pre_scale_node_cpu_cores",
    "pre_scale_node_cpu_pct",
    "pre_scale_node_memory",
    "pre_scale_node_memory_pct",
    "pre_scale_all_nodes_json",
    "request_to_ready_ms",
    "image_cache_assumption",
]

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

    for trial in range(1, TRIALS + 1):

        print(f"========== TRIAL {trial}/{TRIALS} ==========")

        # Return to baseline.
        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            f"--replicas={BASE_REPLICAS}",
        )

        wait_for_ready_replicas(BASE_REPLICAS)

        baseline_pods = get_ready_pods()

        old_names = {
            pod["metadata"]["name"]
            for pod in baseline_pods
        }

        # Capture node pressure immediately before scaling.
        pre_scale_metrics = get_node_metrics()

        print("Pre-scale node metrics:")

        for node, values in pre_scale_metrics.items():
            print(
                f"  {node}: "
                f"CPU={values['cpu_pct']} "
                f"MEM={values['memory_pct']}"
            )

        print()

        # Start local high-resolution timer.
        scale_request = utc_now()
        start_monotonic = time.monotonic()

        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            f"--replicas={TARGET_REPLICAS}",
        )

        new_pod = None

        deadline = time.monotonic() + READY_TIMEOUT

        while time.monotonic() < deadline:

            candidate = find_new_pod(old_names)

            if candidate is not None:

                conditions = candidate.get(
                    "status", {}
                ).get("conditions", [])

                ready = any(
                    c.get("type") == "Ready"
                    and c.get("status") == "True"
                    for c in conditions
                )

                if ready:
                    new_pod = candidate
                    break

            time.sleep(POLL_INTERVAL)

        if new_pod is None:
            raise TimeoutError(
                f"Trial {trial}: new pod did not become Ready."
            )

        end_monotonic = time.monotonic()

        request_to_ready_ms = (
            end_monotonic - start_monotonic
        ) * 1000

        pod_name = new_pod["metadata"]["name"]

        node_name = new_pod.get(
            "spec", {}
        ).get("nodeName", "")

        image, cpu_request, memory_request, startup_delay = (
            get_container_info(new_pod)
        )

        # Match the selected node against the PRE-SCALE snapshot.
        selected_node_metrics = pre_scale_metrics.get(
            node_name,
            {}
        )

        row = {
            "trial": trial,
            "scale_from": BASE_REPLICAS,
            "scale_to": TARGET_REPLICAS,
            "scale_request_time": scale_request.isoformat(),
            "pod_name": pod_name,
            "node_name": node_name,
            "image": image,
            "cpu_request": cpu_request,
            "memory_request": memory_request,
            "configured_startup_delay_ms": startup_delay,

            "pre_scale_node_cpu_cores":
                selected_node_metrics.get(
                    "cpu_cores", ""
                ),

            "pre_scale_node_cpu_pct":
                selected_node_metrics.get(
                    "cpu_pct", ""
                ),

            "pre_scale_node_memory":
                selected_node_metrics.get(
                    "memory", ""
                ),

            "pre_scale_node_memory_pct":
                selected_node_metrics.get(
                    "memory_pct", ""
                ),

            "pre_scale_all_nodes_json":
                json.dumps(
                    pre_scale_metrics,
                    separators=(",", ":")
                ),

            "request_to_ready_ms":
                round(request_to_ready_ms, 2),

            "image_cache_assumption":
                "warm",
        }

        writer.writerow(row)
        f.flush()

        print(f"New pod: {pod_name}")
        print(f"Node: {node_name}")
        print(
            f"Request -> Ready: "
            f"{request_to_ready_ms:.2f} ms"
        )

        print()

        # Return to baseline.
        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            f"--replicas={BASE_REPLICAS}",
        )

        wait_for_ready_replicas(BASE_REPLICAS)

        print("Returned to baseline.")
        print()

print("=" * 55)
print("Measurement complete")
print(f"Dataset: {OUTPUT}")
print("=" * 55)