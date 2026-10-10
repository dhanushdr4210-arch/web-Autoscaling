import argparse
import csv
import time
from datetime import datetime, timezone

from kubernetes import client, config


def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    if dt is None:
        return ""
    return dt.astimezone(timezone.utc).isoformat()


def seconds_between(a, b):
    if a is None or b is None:
        return None
    return (b - a).total_seconds()


def wait_for_pod(v1, namespace, label_selector, timeout=60):
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

    raise TimeoutError("Timed out waiting for pod creation")


def wait_for_ready(v1, namespace, pod_name, timeout=120):

    start = time.time()

    scheduled_time = None
    started_time = None
    ready_time = None

    while time.time() - start < timeout:

        pod = v1.read_namespaced_pod(
            name=pod_name,
            namespace=namespace
        )

        for cond in (pod.status.conditions or []):

            if cond.type == "PodScheduled" and cond.status == "True":
                if scheduled_time is None:
                    scheduled_time = cond.last_transition_time

            if cond.type == "Ready" and cond.status == "True":
                if ready_time is None:
                    ready_time = cond.last_transition_time

        for cs in (pod.status.container_statuses or []):

            if cs.state and cs.state.running:
                if cs.state.running.started_at and started_time is None:
                    started_time = cs.state.running.started_at

        if ready_time is not None:
            return pod, scheduled_time, started_time, ready_time

        time.sleep(0.2)

    raise TimeoutError("Timed out waiting for pod Ready")


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
        "--out",
        default="controlled_startscale_trial.csv"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120
    )

    args = parser.parse_args()

    config.load_kube_config()

    apps = client.AppsV1Api()
    v1 = client.CoreV1Api()

    print()
    print("==============================================")
    print(" StartScale Controlled Measurement v2")
    print("==============================================")
    print(f"Deployment : {args.deployment}")
    print(f"Namespace  : {args.namespace}")
    print(f"Label      : {args.label}")
    print()

    # ------------------------------------------------
    # SAFETY CHECK
    # ------------------------------------------------

    dep = apps.read_namespaced_deployment(
        args.deployment,
        args.namespace
    )

    replicas = dep.spec.replicas or 0

    print(f"Current replicas : {replicas}")

    if replicas != 0:
        raise RuntimeError(
            "SAFETY STOP: deployment replicas are not 0."
        )

    # ------------------------------------------------
    # CHECK FOR OLD PODS
    # ------------------------------------------------

    old_pods = v1.list_namespaced_pod(
        args.namespace,
        label_selector=args.label
    )

    for pod in old_pods.items:

        print(f"Deleting old pod: {pod.metadata.name}")

        v1.delete_namespaced_pod(
            name=pod.metadata.name,
            namespace=args.namespace
        )

    time.sleep(1)

    # ------------------------------------------------
    # REQUEST TIMESTAMP
    # ------------------------------------------------

    request_time = now_utc()

    print()
    print(f"REQUEST_TIME_UTC : {iso(request_time)}")

    # ------------------------------------------------
    # SCALE
    # ------------------------------------------------

    apps.patch_namespaced_deployment_scale(
        name=args.deployment,
        namespace=args.namespace,
        body={
            "spec": {
                "replicas": 1
            }
        }
    )

    print("Scale request sent.")

    # ------------------------------------------------
    # POD CREATION
    # ------------------------------------------------

    pod = wait_for_pod(
        v1,
        args.namespace,
        args.label,
        timeout=args.timeout
    )

    pod_created_time = pod.metadata.creation_timestamp

    print()
    print(f"POD_NAME         : {pod.metadata.name}")
    print(f"POD_CREATED_UTC  : {iso(pod_created_time)}")

    # ------------------------------------------------
    # POD LIFECYCLE
    # ------------------------------------------------

    pod, scheduled_time, started_time, ready_time = wait_for_ready(
        v1,
        args.namespace,
        pod.metadata.name,
        timeout=args.timeout
    )

    node = pod.spec.node_name

    print(f"NODE             : {node}")
    print(f"SCHEDULED_UTC    : {iso(scheduled_time)}")
    print(f"STARTED_UTC      : {iso(started_time)}")
    print(f"READY_UTC        : {iso(ready_time)}")

    # ------------------------------------------------
    # CALCULATE VALID METRICS
    # ------------------------------------------------

    request_to_ready = seconds_between(
        request_time,
        ready_time
    )

    scheduled_to_started = seconds_between(
        scheduled_time,
        started_time
    )

    started_to_ready = seconds_between(
        started_time,
        ready_time
    )

    # Kubernetes creationTimestamp has only coarse
    # timestamp precision in this API response.
    # Therefore DO NOT calculate request -> pod_created.

    print()
    print("--------------- TIMINGS ----------------")

    print(
        f"Scheduled -> Started : "
        f"{scheduled_to_started:.3f}s"
    )

    print(
        f"Started -> Ready : "
        f"{started_to_ready:.3f}s"
    )

    print(
        f"REQUEST -> READY : "
        f"{request_to_ready:.3f}s"
    )

    print("-----------------------------------------")

    # ------------------------------------------------
    # SAVE
    # ------------------------------------------------

    fieldnames = [
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

    with open(args.out, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerow({
            "request_time_utc": iso(request_time),
            "pod_created_time_utc": iso(pod_created_time),
            "scheduled_time_utc": iso(scheduled_time),
            "started_time_utc": iso(started_time),
            "ready_time_utc": iso(ready_time),
            "pod_name": pod.metadata.name,
            "node": node or "",
            "scheduled_to_started_ms":
                f"{scheduled_to_started * 1000:.2f}",
            "started_to_ready_ms":
                f"{started_to_ready * 1000:.2f}",
            "request_to_ready_ms":
                f"{request_to_ready * 1000:.2f}"
        })

    print()
    print(f"Saved result -> {args.out}")

    # ------------------------------------------------
    # RESTORE
    # ------------------------------------------------

    print()
    print("Scaling deployment back to 0...")

    apps.patch_namespaced_deployment_scale(
        name=args.deployment,
        namespace=args.namespace,
        body={
            "spec": {
                "replicas": 0
            }
        }
    )

    print("Deployment restored to 0 replicas.")
    print("Done.")


if __name__ == "__main__":
    main()