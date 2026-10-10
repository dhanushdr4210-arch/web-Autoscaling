import argparse
import csv
import time
import subprocess
from datetime import datetime, timezone

from kubernetes import client, config


def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat() if dt else ""


def seconds_between(a, b):
    if a is None or b is None:
        return None
    return (b - a).total_seconds()


def wait_for_pod(v1, namespace, label_selector, timeout):
    start = time.time()

    while time.time() - start < timeout:

        pods = v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=label_selector
        )

        if pods.items:
            return sorted(
                pods.items,
                key=lambda p: p.metadata.creation_timestamp or now_utc()
            )[-1]

        time.sleep(0.2)

    raise TimeoutError(
        "Timed out waiting for pod creation"
    )


def wait_for_ready(v1, namespace, pod_name, timeout):

    start = time.time()

    scheduled_time = None
    started_time = None
    ready_time = None

    while time.time() - start < timeout:

        pod = v1.read_namespaced_pod(
            name=pod_name,
            namespace=namespace
        )

        # Pod lifecycle conditions
        for cond in (pod.status.conditions or []):

            if (
                cond.type == "PodScheduled"
                and cond.status == "True"
            ):
                if scheduled_time is None:
                    scheduled_time = cond.last_transition_time

            if (
                cond.type == "Ready"
                and cond.status == "True"
            ):
                if ready_time is None:
                    ready_time = cond.last_transition_time

        # Container start time
        for cs in (pod.status.container_statuses or []):

            if cs.state and cs.state.running:

                if (
                    cs.state.running.started_at
                    and started_time is None
                ):
                    started_time = cs.state.running.started_at

        if ready_time is not None:
            return (
                pod,
                scheduled_time,
                started_time,
                ready_time
            )

        time.sleep(0.2)

    raise TimeoutError(
        "Timed out waiting for pod Ready"
    )


def scale_to(apps, deployment, namespace, replicas):

    apps.patch_namespaced_deployment_scale(
        name=deployment,
        namespace=namespace,
        body={
            "spec": {
                "replicas": replicas
            }
        }
    )


def wait_for_zero(
    apps,
    deployment,
    namespace,
    timeout=60
):

    start = time.time()

    while time.time() - start < timeout:

        dep = apps.read_namespaced_deployment(
            deployment,
            namespace
        )

        if (dep.spec.replicas or 0) == 0:
            return

        time.sleep(0.5)

    raise TimeoutError(
        "Deployment did not return to 0 replicas"
    )


def make_worker3_cold():

    image = "localhost:5000/startscale-target:v1"

    print()
    print("Removing target image from worker3...")

    result = subprocess.run(
        [
            "docker",
            "exec",
            "autoscaling-worker3",
            "crictl",
            "rmi",
            image
        ],
        capture_output=True,
        text=True
    )

    # crictl may return non-zero if image is already absent.
    # That is acceptable because we verify the actual state below.
    if result.stdout.strip():
        print(result.stdout.strip())

    if result.stderr.strip():
        print(result.stderr.strip())

    # Verify image is absent
    check = subprocess.run(
        [
            "docker",
            "exec",
            "autoscaling-worker3",
            "crictl",
            "images"
        ],
        capture_output=True,
        text=True
    )

    if "startscale-target" in check.stdout:
        raise RuntimeError(
            "SAFETY STOP: target image is still present "
            "on autoscaling-worker3."
        )

    print(
        "Worker3 image removed -> "
        "COLD CACHE CONFIRMED"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deployment",
        default="startscale-event"
    )

    parser.add_argument(
        "--namespace",
        default="default"
    )

    parser.add_argument(
        "--label",
        default="app=startscale-event"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10
    )

    parser.add_argument(
        "--cooldown",
        type=int,
        default=5
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120
    )

    parser.add_argument(
        "--out",
        default="controlled_cold_trials_v2.csv"
    )

    args = parser.parse_args()

    # Load Kubernetes configuration
    config.load_kube_config()

    apps = client.AppsV1Api()
    v1 = client.CoreV1Api()

    print()
    print("================================================")
    print(" StartScale Controlled Cold Trials v2")
    print("================================================")
    print(f"Deployment : {args.deployment}")
    print(f"Node       : autoscaling-worker3")
    print(f"Trials     : {args.trials}")
    print(f"Cooldown   : {args.cooldown}s")
    print(f"Output     : {args.out}")
    print()

    # ------------------------------------------------
    # SAFETY CHECK
    # ------------------------------------------------

    dep = apps.read_namespaced_deployment(
        args.deployment,
        args.namespace
    )

    current_replicas = dep.spec.replicas or 0

    print(
        f"Current replicas : {current_replicas}"
    )

    if current_replicas != 0:

        raise RuntimeError(
            "SAFETY STOP: deployment replicas are not 0."
        )

    # ------------------------------------------------
    # VERIFY NODE PINNING
    # ------------------------------------------------

    node_name = (
        dep.spec.template.spec.node_name
    )

    print(
        f"Deployment node  : {node_name}"
    )

    if node_name != "autoscaling-worker3":

        raise RuntimeError(
            "SAFETY STOP: deployment is not pinned "
            "to autoscaling-worker3."
        )

    print(
        "Worker3 pinning verified."
    )

    # ------------------------------------------------
    # CSV COLUMNS
    # ------------------------------------------------

    fieldnames = [

        "trial",

        "cache_state",

        "request_time_utc",

        "pod_created_time_utc",

        "scheduled_time_utc",

        "started_time_utc",

        "ready_time_utc",

        "pod_name",

        "node",

        "scheduled_to_started_ms",

        "started_to_ready_ms",

        "request_to_ready_ms"
    ]

    successful = 0

    # ------------------------------------------------
    # CREATE CSV
    # ------------------------------------------------

    with open(
        args.out,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        f.flush()

        try:

            # ========================================
            # TRIAL LOOP
            # ========================================

            for trial in range(
                1,
                args.trials + 1
            ):

                print()
                print("----------------------------------------------")
                print(
                    f" TRIAL {trial}/{args.trials}"
                )
                print("----------------------------------------------")

                # ------------------------------------
                # Make absolutely sure deployment = 0
                # ------------------------------------

                scale_to(
                    apps,
                    args.deployment,
                    args.namespace,
                    0
                )

                wait_for_zero(
                    apps,
                    args.deployment,
                    args.namespace
                )

                # Give Kubernetes time to terminate
                # previous pod completely
                time.sleep(2)

                # ------------------------------------
                # Force cold cache
                # ------------------------------------

                make_worker3_cold()

                # Small settling period
                time.sleep(1)

                # ------------------------------------
                # Exact scale request timestamp
                # ------------------------------------

                request_time = now_utc()

                print()
                print(
                    f"Request time : "
                    f"{iso(request_time)}"
                )

                # ------------------------------------
                # SCALE 0 -> 1
                # ------------------------------------

                scale_to(
                    apps,
                    args.deployment,
                    args.namespace,
                    1
                )

                print(
                    "Scale request sent."
                )

                # ------------------------------------
                # Detect newest pod
                # ------------------------------------

                pod = wait_for_pod(
                    v1,
                    args.namespace,
                    args.label,
                    args.timeout
                )

                pod_created_time = (
                    pod.metadata.creation_timestamp
                )

                # ------------------------------------
                # Wait for Ready
                # ------------------------------------

                (
                    pod,
                    scheduled_time,
                    started_time,
                    ready_time
                ) = wait_for_ready(
                    v1,
                    args.namespace,
                    pod.metadata.name,
                    args.timeout
                )

                node = pod.spec.node_name

                # ------------------------------------
                # Calculate timings
                # ------------------------------------

                scheduled_to_started = (
                    seconds_between(
                        scheduled_time,
                        started_time
                    )
                )

                started_to_ready = (
                    seconds_between(
                        started_time,
                        ready_time
                    )
                )

                request_to_ready = (
                    seconds_between(
                        request_time,
                        ready_time
                    )
                )

                # ------------------------------------
                # Safety: verify correct node
                # ------------------------------------

                if node != "autoscaling-worker3":

                    raise RuntimeError(
                        f"SAFETY STOP: pod scheduled on "
                        f"{node}, expected "
                        f"autoscaling-worker3."
                    )

                # ------------------------------------
                # Print result
                # ------------------------------------

                print(
                    f"Pod     : "
                    f"{pod.metadata.name}"
                )

                print(
                    f"Node    : "
                    f"{node}"
                )

                print(
                    f"Scheduled -> Started : "
                    f"{scheduled_to_started:.3f}s"
                )

                print(
                    f"Started -> Ready    : "
                    f"{started_to_ready:.3f}s"
                )

                print(
                    f"Request -> Ready    : "
                    f"{request_to_ready:.3f}s"
                )

                # ------------------------------------
                # Save CSV row
                # ------------------------------------

                writer.writerow({

                    "trial":
                        trial,

                    "cache_state":
                        "cold",

                    "request_time_utc":
                        iso(request_time),

                    "pod_created_time_utc":
                        iso(pod_created_time),

                    "scheduled_time_utc":
                        iso(scheduled_time),

                    "started_time_utc":
                        iso(started_time),

                    "ready_time_utc":
                        iso(ready_time),

                    "pod_name":
                        pod.metadata.name,

                    "node":
                        node or "",

                    "scheduled_to_started_ms":
                        f"{scheduled_to_started * 1000:.2f}",

                    "started_to_ready_ms":
                        f"{started_to_ready * 1000:.2f}",

                    "request_to_ready_ms":
                        f"{request_to_ready * 1000:.2f}"
                })

                f.flush()

                successful += 1

                # ------------------------------------
                # Return deployment to 0
                # ------------------------------------

                print(
                    "Returning deployment to 0..."
                )

                scale_to(
                    apps,
                    args.deployment,
                    args.namespace,
                    0
                )

                wait_for_zero(
                    apps,
                    args.deployment,
                    args.namespace
                )

                print(
                    f"Trial {trial} complete."
                )

                # ------------------------------------
                # Cooldown
                # ------------------------------------

                if trial < args.trials:

                    print(
                        f"Cooldown: "
                        f"{args.cooldown}s"
                    )

                    time.sleep(
                        args.cooldown
                    )

        finally:

            # ========================================
            # SAFETY RESTORE
            # ========================================

            try:

                scale_to(
                    apps,
                    args.deployment,
                    args.namespace,
                    0
                )

                print()
                print(
                    "Safety restore: replicas = 0"
                )

            except Exception as e:

                print(
                    "WARNING: could not restore "
                    f"replicas: {e}"
                )

    # ------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------

    print()
    print("================================================")
    print(" Collection complete")
    print("================================================")

    print(
        f"Successful trials : "
        f"{successful}/{args.trials}"
    )

    print(
        f"Output             : "
        f"{args.out}"
    )

    print(
        "Deployment replicas : 0"
    )

    print()
    print(
        "Cold-cache experiment finished."
    )
    print()


if __name__ == "__main__":
    main()