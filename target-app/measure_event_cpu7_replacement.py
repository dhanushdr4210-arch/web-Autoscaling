import csv
import json
import subprocess
import time
from datetime import datetime, timezone

DEPLOYMENT = "startscale-event"
LABEL = "app=startscale-event"

BASE_REPLICAS = 0
TARGET_REPLICAS = 1
TRIALS = 1

POLL_INTERVAL = 0.2
READY_TIMEOUT = 120

IMAGE = "localhost:5000/startscale-target:v1"
CACHE_STATE = "warm"

OUTPUT = "v3_cpu7_replacement.csv"

def kubectl(*args):
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


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
    ready = []

    for pod in get_pods():
        conditions = pod.get(
            "status", {}
        ).get("conditions", [])

        is_ready = any(
            c.get("type") == "Ready"
            and c.get("status") == "True"
            for c in conditions
        )

        if is_ready:
            ready.append(pod)

    return ready


def wait_until_no_pods(timeout=120):
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        if len(get_pods()) == 0:
            return

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        "Timed out waiting for all event pods to disappear."
    )


def get_node_metrics():
    raw = kubectl(
        "top",
        "nodes",
        "--no-headers",
    )

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
    statuses = pod.get(
        "status", {}
    ).get(
        "containerStatuses", []
    )

    if not statuses:
        return "", ""

    container = statuses[0]

    image_id = container.get(
        "imageID",
        "",
    )

    running = container.get(
        "state", {}
    ).get(
        "running",
        {},
    )

    return image_id, running.get(
        "startedAt",
        "",
    )


def wait_for_new_ready_pod(old_names):
    deadline = (
        time.monotonic()
        + READY_TIMEOUT
    )

    while time.monotonic() < deadline:

        for pod in get_ready_pods():

            name = pod["metadata"]["name"]

            if name not in old_names:
                return pod

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        "New pod did not become Ready."
    )


def main():

    print("=" * 60)
    print(" StartScale V3 Standardized Startup Measurement")
    print("=" * 60)

    print(f"Deployment : {DEPLOYMENT}")
    print(f"Image      : {IMAGE}")
    print(f"Transition : {BASE_REPLICAS} -> {TARGET_REPLICAS}")
    print(f"Cache      : {CACHE_STATE}")
    print(f"Trials     : {TRIALS}")
    print()

    # Ensure clean starting state.
    kubectl(
        "scale",
        "deployment",
        DEPLOYMENT,
        "--replicas=0",
    )

    wait_until_no_pods()

    fieldnames = [
        "experiment",
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
        print(
            f"========== TRIAL {trial}/{TRIALS} =========="
        )

        # Return to a completely empty state.
        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            "--replicas=0",
        )

        wait_until_no_pods()

        # Capture the exact pre-scale environment.
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

        request_time = datetime.now(
            timezone.utc
        ).isoformat()

        start_time = time.monotonic()

        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            "--replicas=1",
        )

        new_pod = wait_for_new_ready_pod(
            old_names
        )

        latency_ms = (
            time.monotonic()
            - start_time
        ) * 1000

        pod_name = (
            new_pod["metadata"]["name"]
        )

        node_name = (
            new_pod["spec"]
            .get("nodeName", "")
        )

        container = (
            new_pod["spec"]
            ["containers"][0]
        )

        resources = container.get(
            "resources",
            {},
        )

        requests = resources.get(
            "requests",
            {},
        )

        cpu_request = requests.get(
            "cpu",
            "",
        )

        memory_request = requests.get(
            "memory",
            "",
        )

        env = {
            item["name"]: item.get(
                "value",
                "",
            )
            for item in container.get(
                "env",
                [],
            )
        }

        startup_delay = env.get(
            "STARTUP_DELAY_MS",
            "",
        )

        image_id, container_started = (
            get_container_info(new_pod)
        )

        selected = metrics.get(
            node_name,
            {
                "cpu_cores": "",
                "cpu_pct": "",
                "memory": "",
                "memory_pct": "",
            },
        )

        print(
            f"New pod: {pod_name}"
        )

        print(
            f"Node: {node_name}"
        )

        print(
            f"Request -> Ready: "
            f"{latency_ms:.2f} ms"
        )

        rows.append(
            {
                "experiment":
                    "V3_LowPressure_Warm",

                "trial":
                    trial,

                "scale_from":
                    BASE_REPLICAS,

                "scale_to":
                    TARGET_REPLICAS,

                "scale_request_time":
                    request_time,

                "pod_name":
                    pod_name,

                "node_name":
                    node_name,

                "image":
                    IMAGE,

                "image_id":
                    image_id,

                "cpu_request":
                    cpu_request,

                "memory_request":
                    memory_request,

                "configured_startup_delay_ms":
                    startup_delay,

                "pre_scale_node_cpu_cores":
                    selected["cpu_cores"],

                "pre_scale_node_cpu_pct":
                    selected["cpu_pct"],

                "pre_scale_node_memory":
                    selected["memory"],

                "pre_scale_node_memory_pct":
                    selected["memory_pct"],

                "pre_scale_all_nodes_json":
                    json.dumps(
                        metrics,
                        separators=(",", ":"),
                    ),

                "request_to_ready_ms":
                    round(
                        latency_ms,
                        2,
                    ),

                "image_cache_state":
                    CACHE_STATE,

                "container_started":
                    container_started,
            }
        )

        # End this trial at zero replicas.
        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            "--replicas=0",
        )

        wait_until_no_pods()

        print(
            "Returned to zero replicas."
        )

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
    print("=" * 60)
    print(" V3 measurement complete")
    print(f"Dataset: {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
