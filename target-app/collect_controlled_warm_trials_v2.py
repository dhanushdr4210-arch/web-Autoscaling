import argparse
import csv
import time
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

    raise TimeoutError("Timed out waiting for pod creation")


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


def scale_to(apps, deployment, namespace, replicas):
    apps.patch_namespaced_deployment_scale(
        name=deployment,
        namespace=namespace,
        body={"spec": {"replicas": replicas}}
    )


def wait_for_zero(apps, deployment, namespace, timeout=60):
    start = time.time()

    while time.time() - start < timeout:

        dep = apps.read_namespaced_deployment(
            deployment,
            namespace
        )

        if (dep.spec.replicas or 0) == 0:
            return

        time.sleep(0.5)

    raise TimeoutError("Deployment did not return to 0 replicas")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--deployment", default="startscale-event")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--label", default="app=startscale-event")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--out",
        default="controlled_warm_trials_v2.csv"
    )

    args = parser.parse_args()

    config.load_kube_config()

    apps = client.AppsV1Api()
    v1 = client.CoreV1Api()

    print()
    print("==============================================")
    print(" StartScale Controlled Warm Trials v2")
    print("==============================================")
    print(f"Deployment : {args.deployment}")
    print(f"Trials     : {args.trials}")
    print(f"Cooldown   : {args.cooldown}s")
    print()

    # Safety check
    dep = apps.read_namespaced_deployment(
        args.deployment,
        args.namespace
    )

    current_replicas = dep.spec.replicas or 0

    print(f"Current replicas : {current_replicas}")

    if current_replicas != 0:
        raise RuntimeError(
            "SAFETY STOP: deployment replicas are not 0."
        )

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

    with open(args.out, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        f.flush()

        successful = 0

        try:

            for trial in range(1, args.trials + 1):

                print()
                print("----------------------------------------------")
                print(f" TRIAL {trial}/{args.trials}")
                print("----------------------------------------------")

                # Ensure zero replicas
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

                # Allow previous pod termination to settle
                time.sleep(2)

                # Exact request timestamp
                request_time = now_utc()

                print(
                    f"Request time : {iso(request_time)}"
                )

                # Scale up
                scale_to(
                    apps,
                    args.deployment,
                    args.namespace,
                    1
                )

                print("Scale request sent.")

                # Detect pod
                pod = wait_for_pod(
                    v1,
                    args.namespace,
                    args.label,
                    args.timeout
                )

                pod_created_time = pod.metadata.creation_timestamp

                # Lifecycle
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

                scheduled_to_started = seconds_between(
                    scheduled_time,
                    started_time
                )

                started_to_ready = seconds_between(
                    started_time,
                    ready_time
                )

                request_to_ready = seconds_between(
                    request_time,
                    ready_time
                )

                print(f"Pod     : {pod.metadata.name}")
                print(f"Node    : {node}")
                print(
                    f"Started -> Ready : "
                    f"{started_to_ready:.3f}s"
                )
                print(
                    f"Request -> Ready : "
                    f"{request_to_ready:.3f}s"
                )

                writer.writerow({
                    "trial": trial,
                    "cache_state": "warm",
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

                f.flush()

                successful += 1

                # Return to zero before next trial
                print("Returning deployment to 0...")

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

                if trial < args.trials:
                    print(
                        f"Cooldown: {args.cooldown}s"
                    )
                    time.sleep(args.cooldown)

        finally:

            # Safety restore even if an error occurs
            try:
                scale_to(
                    apps,
                    args.deployment,
                    args.namespace,
                    0
                )
            except Exception as e:
                print(
                    f"WARNING: could not restore replicas: {e}"
                )

    print()
    print("==============================================")
    print(" Collection complete")
    print("==============================================")
    print(f"Successful trials : {successful}/{args.trials}")
    print(f"Output             : {args.out}")
    print("Deployment replicas: 0")
    print()


if __name__ == "__main__":
    main()