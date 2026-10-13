import csv
import json
import subprocess
import time
from datetime import datetime, timezone


# ============================================================
# CONFIGURATION
# ============================================================

DEPLOYMENT = "startscale-cache-event"
LABEL = "app=startscale-cache-event"

BASE_REPLICAS = 0
TARGET_REPLICAS = 1
TRIALS = 10

POLL_INTERVAL = 0.2
READY_TIMEOUT = 180

IMAGE = "localhost:5000/startscale-target:cold-v2"
CACHE_STATE = "cold"

EXPERIMENT = "V3_COMBINED_CPU06_08_MEM16_18_COLD"

OUTPUT = "v3_combined_cpu06_08_mem16_18_cold.csv"

WORKERS = [
    "autoscaling-worker",
    "autoscaling-worker2",
    "autoscaling-worker3",
]


# ============================================================
# KUBECTL
# ============================================================

def kubectl(*args):
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


# ============================================================
# DOCKER / CRICTL
# ============================================================

def docker_exec(
    node,
    *args,
):
    result = subprocess.run(
        [
            "docker",
            "exec",
            node,
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    return result


# ============================================================
# VERIFY IMAGE ABSENT
# ============================================================

def image_present_on_worker(
    node,
    image,
):
    """
    Check the complete crictl image list and compare
    repository + tag exactly.

    We do NOT use:
        crictl images -q IMAGE

    because in this environment that lookup did not behave
    as expected and returned the full image list.
    """

    if ":" not in image:
        raise ValueError(
            f"Image must contain repository:tag: {image}"
        )

    repository, tag = image.rsplit(
        ":",
        1,
    )

    result = docker_exec(
        node,
        "crictl",
        "images",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Could not inspect images on {node}.\n"
            f"{result.stderr.strip()}"
        )

    for line in result.stdout.splitlines():

        parts = line.split()

        if len(parts) < 2:
            continue

        repo = parts[0]
        found_tag = parts[1]

        if (
            repo == repository
            and found_tag == tag
        ):
            return True

    return False


# ============================================================
# REMOVE COLD IMAGE FROM ALL WORKERS
# ============================================================

def remove_image_from_all_workers():

    print()
    print("=" * 70)
    print("Removing cold-v2 from all worker nodes")
    print("=" * 70)

    for node in WORKERS:

        result = docker_exec(
            node,
            "crictl",
            "rmi",
            IMAGE,
        )

        output = (
            result.stdout.strip()
            or result.stderr.strip()
        )

        print(
            f"{node}: {output}"
        )

    print()
    print(
        "Verifying cold-v2 is absent..."
    )

    remaining = []

    for node in WORKERS:

        if image_present_on_worker(
            node,
            IMAGE,
        ):
            remaining.append(node)

    if remaining:

        raise RuntimeError(
            "Image still present on: "
            + ", ".join(remaining)
        )

    print(
        "Verified: cold-v2 absent on all workers."
    )


# ============================================================
# POD HELPERS
# ============================================================

def get_pods():

    raw = kubectl(
        "get",
        "pods",
        "-l",
        LABEL,
        "-o",
        "json",
    )

    return json.loads(
        raw
    )["items"]


def get_ready_pods():

    ready = []

    for pod in get_pods():

        conditions = (
            pod
            .get(
                "status",
                {},
            )
            .get(
                "conditions",
                [],
            )
        )

        is_ready = any(
            condition.get("type") == "Ready"
            and condition.get("status") == "True"
            for condition in conditions
        )

        if is_ready:
            ready.append(pod)

    return ready


def wait_until_no_pods(
    timeout=120,
):

    start = time.monotonic()

    while (
        time.monotonic() - start
        < timeout
    ):

        pods = get_pods()

        if len(pods) == 0:
            return

        time.sleep(
            POLL_INTERVAL
        )

    raise TimeoutError(
        "Timed out waiting for "
        "cache-event pods to disappear."
    )


# ============================================================
# NODE METRICS
# ============================================================

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


# ============================================================
# CONTAINER INFORMATION
# ============================================================

def get_container_info(
    pod,
):

    statuses = (
        pod
        .get(
            "status",
            {},
        )
        .get(
            "containerStatuses",
            [],
        )
    )

    if not statuses:
        return "", ""

    container = statuses[0]

    image_id = container.get(
        "imageID",
        "",
    )

    running = (
        container
        .get(
            "state",
            {},
        )
        .get(
            "running",
            {},
        )
    )

    return (
        image_id,
        running.get(
            "startedAt",
            "",
        ),
    )


# ============================================================
# WAIT FOR NEW READY POD
# ============================================================

def wait_for_new_ready_pod(
    old_names,
):

    deadline = (
        time.monotonic()
        + READY_TIMEOUT
    )

    while (
        time.monotonic()
        < deadline
    ):

        for pod in get_ready_pods():

            name = (
                pod["metadata"]["name"]
            )

            if name not in old_names:

                return pod

        time.sleep(
            POLL_INTERVAL
        )

    raise TimeoutError(
        "New cold-cache pod did not "
        "become Ready within the timeout."
    )


# ============================================================
# DEPLOYMENT CHECK
# ============================================================

def verify_deployment():

    try:

        raw = kubectl(
            "get",
            "deployment",
            DEPLOYMENT,
            "-o",
            "json",
        )

        deployment = json.loads(raw)

    except Exception as exc:

        raise RuntimeError(
            f"Deployment '{DEPLOYMENT}' "
            f"was not found."
        ) from exc

    replicas = (
        deployment
        .get(
            "spec",
            {},
        )
        .get(
            "replicas",
            None,
        )
    )

    print(
        f"Deployment found: {DEPLOYMENT}"
    )

    print(
        f"Configured replicas: {replicas}"
    )


# ============================================================
# PRESSURE CHECK
# ============================================================

def verify_pressure_pods():

    raw = kubectl(
        "get",
        "pods",
        "-l",
        "app=startscale-combined-cpu06-08-mem16",
        "-o",
        "json",
    )

    pods = json.loads(raw)["items"]

    if len(pods) != 3:

        raise RuntimeError(
            "Expected 3 pressure pods, "
            f"found {len(pods)}."
        )

    for pod in pods:

        phase = (
            pod
            .get(
                "status",
                {},
            )
            .get(
                "phase",
                "",
            )
        )

        if phase != "Running":

            raise RuntimeError(
                f"Pressure pod "
                f"{pod['metadata']['name']} "
                f"is not Running."
            )

    print(
        "Pressure condition: "
        "3/3 pods Running."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "StartScale V3 - "
        "Matched Cold Cache Experiment"
    )
    print("=" * 75)

    print(
        f"Experiment : {EXPERIMENT}"
    )

    print(
        f"Deployment : {DEPLOYMENT}"
    )

    print(
        f"Image      : {IMAGE}"
    )

    print(
        "Condition  : "
        "CPU 6-8% + Memory 16-18%"
    )

    print(
        f"Cache      : {CACHE_STATE}"
    )

    print(
        f"Transition : "
        f"{BASE_REPLICAS} -> {TARGET_REPLICAS}"
    )

    print(
        f"Trials     : {TRIALS}"
    )

    print(
        f"Output     : {OUTPUT}"
    )

    print()

    # --------------------------------------------------------
    # PRE-FLIGHT
    # --------------------------------------------------------

    print(
        "Running pre-flight checks..."
    )

    verify_deployment()

    verify_pressure_pods()

    metrics = get_node_metrics()

    print()
    print(
        "Current worker metrics:"
    )

    for node in WORKERS:

        values = metrics.get(
            node,
            {},
        )

        print(
            f"  {node}: "
            f"CPU={values.get('cpu_pct', '?')} "
            f"MEM={values.get('memory_pct', '?')}"
        )

    # --------------------------------------------------------
    # Ensure deployment starts at zero.
    # --------------------------------------------------------

    kubectl(
        "scale",
        "deployment",
        DEPLOYMENT,
        "--replicas=0",
    )

    wait_until_no_pods()

    print()
    print(
        "Deployment confirmed at 0 replicas."
    )

    # --------------------------------------------------------
    # CSV fields
    # --------------------------------------------------------

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

    # ========================================================
    # TRIAL LOOP
    # ========================================================

    for trial in range(
        1,
        TRIALS + 1,
    ):

        print()
        print(
            "=" * 65
        )
        print(
            f"TRIAL {trial}/{TRIALS}"
        )
        print(
            "=" * 65
        )

        # ----------------------------------------------------
        # Force zero replicas.
        # ----------------------------------------------------

        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            "--replicas=0",
        )

        wait_until_no_pods()

        # ----------------------------------------------------
        # Make the image genuinely cold.
        # ----------------------------------------------------

        remove_image_from_all_workers()

        # ----------------------------------------------------
        # Capture pre-scale metrics.
        # ----------------------------------------------------

        metrics = get_node_metrics()

        print()
        print(
            "Pre-scale node metrics:"
        )

        for node in WORKERS:

            values = metrics.get(
                node,
                {},
            )

            print(
                f"  {node}: "
                f"CPU={values.get('cpu_pct', '?')} "
                f"MEM={values.get('memory_pct', '?')}"
            )

        # ----------------------------------------------------
        # Existing pods before scale.
        # ----------------------------------------------------

        old_names = {
            pod["metadata"]["name"]
            for pod in get_pods()
        }

        # ----------------------------------------------------
        # Start timing immediately before scale command.
        # ----------------------------------------------------

        request_time = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        start_time = (
            time.monotonic()
        )

        # ----------------------------------------------------
        # Scale 0 -> 1.
        # ----------------------------------------------------

        kubectl(
            "scale",
            "deployment",
            DEPLOYMENT,
            "--replicas=1",
        )

        # ----------------------------------------------------
        # Wait for new pod Ready.
        # ----------------------------------------------------

        new_pod = (
            wait_for_new_ready_pod(
                old_names
            )
        )

        latency_ms = (
            time.monotonic()
            - start_time
        ) * 1000

        # ----------------------------------------------------
        # Pod information
        # ----------------------------------------------------

        pod_name = (
            new_pod["metadata"]["name"]
        )

        node_name = (
            new_pod["spec"]
            .get(
                "nodeName",
                "",
            )
        )

        container = (
            new_pod["spec"]
            ["containers"][0]
        )

        resources = (
            container
            .get(
                "resources",
                {},
            )
        )

        requests = (
            resources
            .get(
                "requests",
                {},
            )
        )

        cpu_request = (
            requests.get(
                "cpu",
                "",
            )
        )

        memory_request = (
            requests.get(
                "memory",
                "",
            )
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
            get_container_info(
                new_pod
            )
        )

        # ----------------------------------------------------
        # Selected node metrics
        # ----------------------------------------------------

        selected = metrics.get(
            node_name,
            {
                "cpu_cores": "",
                "cpu_pct": "",
                "memory": "",
                "memory_pct": "",
            },
        )

        print()
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

        print(
            f"Image ID: {image_id}"
        )

        # ----------------------------------------------------
        # Add row
        # ----------------------------------------------------

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
                    selected["memory"],

                "pre_scale_node_memory_pct":
                    selected["memory_pct"],

                "pre_scale_all_nodes_json":
                    json.dumps(
                        metrics,
                        separators=(
                            ",",
                            ":",
                        ),
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

        # ----------------------------------------------------
        # Return to zero.
        # ----------------------------------------------------

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

    # ========================================================
    # SAVE DATASET
    # ========================================================

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    latencies = [
        row["request_to_ready_ms"]
        for row in rows
    ]

    print()
    print("=" * 75)
    print(
        "MATCHED COLD-CACHE EXPERIMENT COMPLETE"
    )
    print("=" * 75)

    print(
        f"Trials : {len(rows)}"
    )

    print(
        f"Mean latency : "
        f"{sum(latencies) / len(latencies):.2f} ms"
    )

    print(
        f"Min latency  : "
        f"{min(latencies):.2f} ms"
    )

    print(
        f"Max latency  : "
        f"{max(latencies):.2f} ms"
    )

    print(
        f"Dataset : {OUTPUT}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()