"""
StartScale v2 - Measurement Harness (Step 6)

Watches Kubernetes pod events for a given label selector and records, per pod:
  - scheduled_time   : when the scheduler assigned it to a node
  - created_time     : when the container was created
  - started_time     : when the container process started running
  - ready_time       : when the readiness probe first passed (pod added to Service endpoints)
  - node             : which node it landed on
  - readiness_latency_s : ready_time - started_time  (the core signal for StartScale's novelty claim)
  - total_latency_s     : ready_time - scheduled_time

Also polls replica count over time (from the Deployment) so you can plot
"replica count vs time" alongside readiness latency for each scaling
strategy (HPA / LSTM-only / LSTM+RL) later.

Usage:
    pip install kubernetes
    python measure_harness.py --label app=startscale-target --namespace default \
        --deployment startscale-target --out results.csv --replica-out replicas.csv \
        --duration 300

Requires: your kubeconfig already points at the kind cluster (kubectl works),
since this reuses the same config kubectl uses.
"""

import argparse
import csv
import time
import threading
from datetime import datetime, timezone

from kubernetes import client, config, watch


def iso(dt):
    if dt is None:
        return ""
    return dt.astimezone(timezone.utc).isoformat()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class PodRecord:
    def __init__(self, name):
        self.name = name
        self.node = None
        self.scheduled_time = None
        self.created_time = None
        self.started_time = None
        self.ready_time = None
        self.written = False


def watch_pods(v1, label_selector, namespace, out_path, duration, stop_event):
    records = {}
    fieldnames = [
        "pod_name", "node", "scheduled_time", "created_time", "started_time",
        "ready_time", "readiness_latency_s", "total_latency_s",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        w = watch.Watch()
        start = time.time()
        try:
            for event in w.stream(
                v1.list_namespaced_pod,
                namespace=namespace,
                label_selector=label_selector,
                timeout_seconds=duration,
            ):
                if stop_event.is_set():
                    w.stop()
                    break
                pod = event["object"]
                name = pod.metadata.name
                rec = records.setdefault(name, PodRecord(name))

                if pod.spec.node_name:
                    rec.node = pod.spec.node_name

                # Scheduled: PodScheduled condition True
                for cond in (pod.status.conditions or []):
                    if cond.type == "PodScheduled" and cond.status == "True" and rec.scheduled_time is None:
                        rec.scheduled_time = cond.last_transition_time
                    if cond.type == "Ready" and cond.status == "True" and rec.ready_time is None:
                        rec.ready_time = cond.last_transition_time

                # Container created/started from container statuses
                for cs in (pod.status.container_statuses or []):
                    if cs.state and cs.state.running:
                        if rec.started_time is None:
                            rec.started_time = cs.state.running.started_at
                        if rec.created_time is None:
                            rec.created_time = cs.state.running.started_at

                # Once ready, write the row (once per pod)
                if rec.ready_time and not rec.written:
                    readiness_latency = None
                    total_latency = None
                    if rec.started_time:
                        readiness_latency = (rec.ready_time - rec.started_time).total_seconds()
                    if rec.scheduled_time:
                        total_latency = (rec.ready_time - rec.scheduled_time).total_seconds()
                    writer.writerow({
                        "pod_name": rec.name,
                        "node": rec.node or "",
                        "scheduled_time": iso(rec.scheduled_time),
                        "created_time": iso(rec.created_time),
                        "started_time": iso(rec.started_time),
                        "ready_time": iso(rec.ready_time),
                        "readiness_latency_s": f"{readiness_latency:.3f}" if readiness_latency is not None else "",
                        "total_latency_s": f"{total_latency:.3f}" if total_latency is not None else "",
                    })
                    f.flush()
                    rec.written = True
                    print(f"[pod-ready] {rec.name} on {rec.node}: "
                          f"readiness_latency={readiness_latency:.3f}s" if readiness_latency is not None
                          else f"[pod-ready] {rec.name} on {rec.node}: readiness_latency=unknown")

                if time.time() - start > duration:
                    w.stop()
                    break
        except Exception as e:
            print(f"[measure_harness] pod watch stopped: {e}")


def poll_replicas(apps_v1, namespace, deployment_name, out_path, duration, stop_event, interval=2.0):
    fieldnames = ["timestamp", "desired_replicas", "ready_replicas", "available_replicas"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        start = time.time()
        while time.time() - start < duration and not stop_event.is_set():
            try:
                dep = apps_v1.read_namespaced_deployment(deployment_name, namespace)
                status = dep.status
                writer.writerow({
                    "timestamp": now_iso(),
                    "desired_replicas": dep.spec.replicas or 0,
                    "ready_replicas": status.ready_replicas or 0,
                    "available_replicas": status.available_replicas or 0,
                })
                f.flush()
            except Exception as e:
                print(f"[measure_harness] replica poll error: {e}")
            time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="Label selector, e.g. app=startscale-target")
    ap.add_argument("--namespace", default="default")
    ap.add_argument("--deployment", required=True, help="Deployment name to poll replica counts")
    ap.add_argument("--out", default="pod_readiness.csv", help="Per-pod readiness CSV output path")
    ap.add_argument("--replica-out", default="replicas.csv", help="Replica-count-over-time CSV output path")
    ap.add_argument("--duration", type=int, default=300, help="How long to run, in seconds")
    args = ap.parse_args()

    config.load_kube_config()  # uses the same kubeconfig as kubectl
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()

    stop_event = threading.Event()

    replica_thread = threading.Thread(
        target=poll_replicas,
        args=(apps_v1, args.namespace, args.deployment, args.replica_out, args.duration, stop_event),
        daemon=True,
    )
    replica_thread.start()

    print(f"[measure_harness] watching pods with label '{args.label}' for {args.duration}s ...")
    print(f"[measure_harness] pod readiness -> {args.out}")
    print(f"[measure_harness] replica counts -> {args.replica_out}")

    try:
        watch_pods(v1, args.label, args.namespace, args.out, args.duration, stop_event)
    except KeyboardInterrupt:
        print("\n[measure_harness] stopped by user")
    finally:
        stop_event.set()
        replica_thread.join(timeout=5)

    print("[measure_harness] done.")


if __name__ == "__main__":
    main()
