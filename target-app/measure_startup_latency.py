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

POLL_INTERVAL = 0.2
READY_TIMEOUT = 60

OUTPUT = "startup_latency_dataset.csv"


def run_kubectl(*args):
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()


def now_utc():
    return datetime.now(timezone.utc)


def parse_k8s_time(value):
    if not value:
        return None

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    return datetime.fromisoformat(value)


def get_pods():
    output = run_kubectl(
        "get",
        "pods",
        "-l",
        LABEL,
        "-o",
        "json"
    )
    return json.loads(output)["items"]


def ready_pods():
    pods = get_pods()

    result = []

    for pod in pods:
        conditions = pod.get("status", {}).get("conditions", [])

        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in conditions
        )

        if ready:
            result.append(pod)

    return result


def wait_for_replicas(count, timeout=READY_TIMEOUT):
    deadline = time.time() + timeout

    while time.time() < deadline:
        pods = ready_pods()

        if len(pods) == count:
            return pods

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"Timed out waiting for {count} Ready replicas."
    )


def get_image(pod):
    containers = pod.get("spec", {}).get("containers", [])

    if containers:
        return containers[0].get("image", "")

    return ""


def get_startup_delay_ms(pod):
    containers = pod.get("spec", {}).get("containers", [])

    if not containers:
        return ""

    env = containers[0].get("env", [])

    for item in env:
        if item.get("name") == "STARTUP_DELAY_MS":
            return item.get("value", "")

    return ""


def get_cpu_request(pod):
    containers = pod.get("spec", {}).get("containers", [])

    if not containers:
        return ""

    resources = containers[0].get("resources", {})
    requests = resources.get("requests", {})

    return requests.get("cpu", "")


def get_memory_request(pod):
    containers = pod.get("spec", {}).get("containers", [])

    if not containers:
        return ""

    resources = containers[0].get("resources", {})
    requests = resources.get("memory", {})

    return requests.get("memory", "")


def find_new_pod(old_names):
    pods = get_pods()

    new_pods = [
        p for p in pods
        if p["metadata"]["name"] not in old_names
    ]

    if not new_pods:
        return None

    new_pods.sort(
        key=lambda p: p["metadata"].get("creationTimestamp", "")
    )

    return new_pods[-1]


print("========================================")
print(" StartScale Startup-Latency Measurement")
print("========================================")
print(f"Trials: {TRIALS}")
print(f"Baseline replicas: {BASE_REPLICAS}")
print(f"Target replicas:   {TARGET_REPLICAS}")
print()

# Make sure the cluster starts from baseline.
print("Setting baseline replicas...")
run_kubectl(
    "scale",
    "deployment",
    DEPLOYMENT,
    f"--replicas={BASE_REPLICAS}"
)

wait_for_replicas(BASE_REPLICAS)

print("Baseline ready.")
print()

fieldnames = [
    "trial",
    "scale_from",
    "scale_to",
    "scale_request_time",
    "scale_api_return_time",
    "pod_name",
    "node_name",
    "image",
    "image_cache_assumption",
    "pod_creation_time",
    "container_started_time",
    "pod_ready_time",
    "request_to_creation_ms",
    "request_to_container_start_ms",
    "request_to_ready_ms",
    "container_start_to_ready_ms",
    "configured_startup_delay_ms",
    "cpu_request",
    "memory_request",
]

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for trial in range(1, TRIALS + 1):

        print(f"========== TRIAL {trial}/{TRIALS} ==========")

        # Ensure baseline.
        run_kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            f"--replicas={BASE_REPLICAS}"
        )

        baseline_pods = wait_for_replicas(BASE_REPLICAS)

        old_names = {
            p["metadata"]["name"]
            for p in baseline_pods
        }

        print(f"Baseline pods: {len(old_names)}")

        # Initiate scale-up.
        scale_request = now_utc()

        run_kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            f"--replicas={TARGET_REPLICAS}"
        )

        scale_return = now_utc()

        print(
            "Scale request:",
            scale_request.isoformat()
        )

        # Wait for a new Ready pod.
        deadline = time.time() + READY_TIMEOUT

        new_pod = None

        while time.time() < deadline:

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
            print("ERROR: new pod did not become Ready.")
            raise TimeoutError(
                f"Trial {trial}: new pod did not become Ready."
            )

        metadata = new_pod["metadata"]
        status = new_pod.get("status", {})
        spec = new_pod.get("spec", {})

        pod_name = metadata["name"]
        node_name = spec.get("nodeName", "")

        creation_time = parse_k8s_time(
            metadata.get("creationTimestamp")
        )

        container_started = None

        container_statuses = status.get(
            "containerStatuses",
            []
        )

        if container_statuses:

            state = container_statuses[0].get(
                "state",
                {}
            )

            running = state.get("running")

            if running:
                container_started = parse_k8s_time(
                    running.get("startedAt")
                )

        ready_time = None

        for condition in status.get("conditions", []):

            if condition.get("type") == "Ready":
                if condition.get("status") == "True":
                    ready_time = parse_k8s_time(
                        condition.get("lastTransitionTime")
                    )

        request_to_creation = None
        request_to_container = None
        request_to_ready = None
        container_to_ready = None

        if creation_time:
            request_to_creation = (
                creation_time - scale_request
            ).total_seconds() * 1000

        if container_started:
            request_to_container = (
                container_started - scale_request
            ).total_seconds() * 1000

        if ready_time:
            request_to_ready = (
                ready_time - scale_request
            ).total_seconds() * 1000

        if container_started and ready_time:
            container_to_ready = (
                ready_time - container_started
            ).total_seconds() * 1000

        row = {
            "trial": trial,
            "scale_from": BASE_REPLICAS,
            "scale_to": TARGET_REPLICAS,
            "scale_request_time": scale_request.isoformat(),
            "scale_api_return_time": scale_return.isoformat(),
            "pod_name": pod_name,
            "node_name": node_name,
            "image": get_image(new_pod),
            "image_cache_assumption": "warm",
            "pod_creation_time": creation_time.isoformat()
            if creation_time else "",
            "container_started_time": container_started.isoformat()
            if container_started else "",
            "pod_ready_time": ready_time.isoformat()
            if ready_time else "",
            "request_to_creation_ms": round(
                request_to_creation, 2
            ) if request_to_creation is not None else "",
            "request_to_container_start_ms": round(
                request_to_container, 2
            ) if request_to_container is not None else "",
            "request_to_ready_ms": round(
                request_to_ready, 2
            ) if request_to_ready is not None else "",
            "container_start_to_ready_ms": round(
                container_to_ready, 2
            ) if container_to_ready is not None else "",
            "configured_startup_delay_ms":
                get_startup_delay_ms(new_pod),
            "cpu_request":
                get_cpu_request(new_pod),
            "memory_request":
                get_memory_request(new_pod),
        }

        writer.writerow(row)
        f.flush()

        print(f"New pod: {pod_name}")
        print(f"Node: {node_name}")
        print(
            f"Request → Ready: "
            f"{row['request_to_ready_ms']} ms"
        )
        print()

        # Return to baseline before next trial.
        run_kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            f"--replicas={BASE_REPLICAS}"
        )

        wait_for_replicas(BASE_REPLICAS)

        print("Returned to baseline.")
        print()

print("========================================")
print("Measurement complete.")
print(f"Dataset: {OUTPUT}")
print("========================================")