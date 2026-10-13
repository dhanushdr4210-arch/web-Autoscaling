import csv
import json
import subprocess
import time
from datetime import datetime, timezone

DEPLOYMENT = "startscale-event"
LABEL = "app=startscale-event"

BASE_REPLICAS = 0
TARGET_REPLICAS = 1
TRIALS = 10

POLL_INTERVAL = 0.2
READY_TIMEOUT = 120

IMAGE = "localhost:5000/startscale-target:v1"
CACHE_STATE = "warm"

EXPERIMENT = "V3_Combined_CPU11_14_Mem22_23"
OUTPUT = "v3_combined_cpu11_14_mem22_23.csv"


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
        conditions = pod.get("status", {}).get("conditions", [])

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


def parse_cpu_to_millicores(cpu):
    """
    Kubernetes Metrics API CPU:
      123456789n -> 123.456789 millicores
      500m       -> 500 millicores
      2          -> 2000 millicores
    """
    cpu = str(cpu).strip()

    if cpu.endswith("n"):
        return float(cpu[:-1]) / 1_000_000

    if cpu.endswith("u"):
        return float(cpu[:-1]) / 1_000

    if cpu.endswith("m"):
        return float(cpu[:-1])

    return float(cpu) * 1000


def parse_memory_to_mib(memory):
    """
    Kubernetes Metrics API memory:
      1745564Ki -> MiB
      2Gi       -> MiB
      500Mi     -> MiB
    """
    memory = str(memory).strip()

    if memory.endswith("Ki"):
        return float(memory[:-2]) / 1024

    if memory.endswith("Mi"):
        return float(memory[:-2])

    if memory.endswith("Gi"):
        return float(memory[:-2]) * 1024

    if memory.endswith("Ti"):
        return float(memory[:-2]) * 1024 * 1024

    return float(memory) / (1024 * 1024)


def get_node_metrics():
    """
    Read raw Kubernetes Metrics API.

    This intentionally does NOT use `kubectl top nodes`,
    because the project calibration uses the raw Metrics API.
    """

    raw = kubectl(
        "get",
        "--raw",
        "/apis/metrics.k8s.io/v1beta1/nodes",
    )

    data = json.loads(raw)

    metrics = {}

    for item in data.get("items", []):
        node = item["metadata"]["name"]

        usage = item.get("usage", {})

        cpu_raw = usage.get("cpu", "")
        memory_raw = usage.get("memory", "")

        cpu_m = parse_cpu_to_millicores(cpu_raw)
        memory_mib = parse_memory_to_mib(memory_raw)

        # kind worker capacity = 16 CPU
        cpu_pct = (cpu_m / 16000.0) * 100.0

        # Actual node capacity used in this project.
        # 7970460Ki ≈ 7783.2 MiB
        node_memory_mib = 7970460 / 1024
        memory_pct = (
            memory_mib / node_memory_mib
        ) * 100.0

        metrics[node] = {
            "cpu_raw": cpu_raw,
            "memory_raw": memory_raw,
            "cpu_cores": round(cpu_m / 1000, 4),
            "cpu_pct": round(cpu_pct, 2),
            "memory_mib": round(memory_mib, 2),
            "memory_pct": round(memory_pct, 2),
        }

    return metrics


def get_container_info(pod):
    statuses = (
        pod.get("status", {})
        .get("containerStatuses", [])
    )

    if not statuses:
        return "", ""

    container = statuses[0]

    image_id = container.get("imageID", "")

    running = (
        container.get("state", {})
        .get("running", {})
    )

    return image_id, running.get("startedAt", "")


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

    print("=" * 70)
    print(" StartScale V3 CPU11-14 / MEM22-23 Measurement")
    print("=" * 70)

    print(f"Deployment : {DEPLOYMENT}")
    print(f"Image      : {IMAGE}")
    print(f"Transition : {BASE_REPLICAS} -> {TARGET_REPLICAS}")
    print(f"Cache      : {CACHE_STATE}")
    print(f"Trials     : {TRIALS}")
    print(f"Experiment : {EXPERIMENT}")
    print(f"Output     : {OUTPUT}")
    print()

    # Make sure measurement deployment starts at zero.
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
        print("=" * 70)
        print(f" TRIAL {trial}/{TRIALS}")
        print("=" * 70)

        # Ensure event deployment is back at zero.
        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            "--replicas=0",
        )

        wait_until_no_pods()

        # Capture exact environment before scale request.
        metrics = get_node_metrics()

        print("Pre-scale RAW Metrics API:")

        for node, values in metrics.items():
            print(
                f"  {node}: "
                f"CPU={values['cpu_raw']} "
                f"({values['cpu_pct']:.2f}%) | "
                f"MEM={values['memory_raw']} "
                f"({values['memory_pct']:.2f}%)"
            )

        old_names = {
            pod["metadata"]["name"]
            for pod in get_pods()
        }

        request_time = datetime.now(
            timezone.utc
        ).isoformat()

        start_time = time.monotonic()

        # 0 -> 1
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

        pod_name = new_pod["metadata"]["name"]

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
                "memory_mib": "",
                "memory_pct": "",
            },
        )

        print()
        print(f"New pod              : {pod_name}")
        print(f"Node                 : {node_name}")
        print(
            f"Request -> Ready     : "
            f"{latency_ms:.2f} ms"
        )
        print(
            f"Pre-scale CPU        : "
            f"{selected['cpu_pct']}%"
        )
        print(
            f"Pre-scale Memory     : "
            f"{selected['memory_pct']}%"
        )

        rows.append(
            {
                "experiment":
                    EXPERIMENT,

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
                    selected["memory_raw"]
                    if node_name in metrics
                    else "",

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

        # Return event deployment to zero.
        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            "--replicas=0",
        )

        wait_until_no_pods()

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
    print("=" * 70)
    print(" V3 measurement complete")
    print(f" Dataset: {OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()