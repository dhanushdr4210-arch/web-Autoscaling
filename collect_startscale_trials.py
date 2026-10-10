import csv
import datetime
import subprocess
import time


DEPLOYMENT = "startscale-event"
LABEL = "app=startscale-event"
TRIALS = 10
OUTPUT = "startscale_10_trials.csv"

# Current live-model predictions from our validated controller run
P50_PREDICTION_MS = 6731.83
P90_PREDICTION_MS = 7615.14

rows = []

print("=" * 80)
print("STARTSCALE 10-TRIAL CONTROLLED VALIDATION")
print("=" * 80)

print(f"Trials              : {TRIALS}")
print(f"P50 prediction      : {P50_PREDICTION_MS:.2f} ms")
print(f"Calibrated P90      : {P90_PREDICTION_MS:.2f} ms")
print(f"Output              : {OUTPUT}")
print()

# Make sure the deployment starts at zero
subprocess.run(
    ["kubectl", "scale", "deployment", DEPLOYMENT, "--replicas=0"],
    check=True,
    capture_output=True,
    text=True
)

time.sleep(3)

for trial in range(1, TRIALS + 1):

    print("-" * 80)
    print(f"TRIAL {trial}/{TRIALS}")
    print("-" * 80)

    # Ensure previous pod is gone
    subprocess.run(
        ["kubectl", "scale", "deployment", DEPLOYMENT, "--replicas=0"],
        check=True,
        capture_output=True,
        text=True
    )

    time.sleep(3)

    # Exact measurement start
    request_time = datetime.datetime.now(datetime.timezone.utc)

    print("Request time:", request_time.isoformat())

    # Trigger scale-up
    subprocess.run(
        ["kubectl", "scale", "deployment", DEPLOYMENT, "--replicas=1"],
        check=True,
        capture_output=True,
        text=True
    )

    # Wait until pod becomes Ready
    result = subprocess.run(
        [
            "kubectl",
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            LABEL,
            "--timeout=60s"
        ],
        capture_output=True,
        text=True
    )

    ready_time = datetime.datetime.now(datetime.timezone.utc)

    if result.returncode != 0:
        print("ERROR: Pod did not become Ready.")
        print(result.stderr.strip())

        rows.append({
            "trial": trial,
            "request_time_utc": request_time.isoformat(),
            "ready_time_utc": "",
            "request_to_ready_ms": "",
            "pod_name": "",
            "node_name": "",
            "cache_state": "warm",
            "p50_prediction_ms": P50_PREDICTION_MS,
            "p90_prediction_ms": P90_PREDICTION_MS,
            "status": "FAILED"
        })

        subprocess.run(
            ["kubectl", "scale", "deployment", DEPLOYMENT, "--replicas=0"],
            check=True,
            capture_output=True,
            text=True
        )

        continue

    latency_ms = round(
        (ready_time - request_time).total_seconds() * 1000,
        2
    )

    # Get pod name and node name
    pod_result = subprocess.run(
        [
            "kubectl",
            "get",
            "pod",
            "-l",
            LABEL,
            "-o",
            "jsonpath={.items[0].metadata.name}"
        ],
        capture_output=True,
        text=True,
        check=True
    )

    pod_name = pod_result.stdout.strip()

    node_result = subprocess.run(
        [
            "kubectl",
            "get",
            "pod",
            pod_name,
            "-o",
            "jsonpath={.spec.nodeName}"
        ],
        capture_output=True,
        text=True,
        check=True
    )

    node_name = node_result.stdout.strip()

    print("Pod:", pod_name)
    print("Node:", node_name)
    print("Ready time:", ready_time.isoformat())
    print(f"Actual latency: {latency_ms:.2f} ms")

    rows.append({
        "trial": trial,
        "request_time_utc": request_time.isoformat(),
        "ready_time_utc": ready_time.isoformat(),
        "request_to_ready_ms": latency_ms,
        "pod_name": pod_name,
        "node_name": node_name,
        "cache_state": "warm",
        "p50_prediction_ms": P50_PREDICTION_MS,
        "p90_prediction_ms": P90_PREDICTION_MS,
        "status": "SUCCESS"
    })

    # Reset to zero
    subprocess.run(
        ["kubectl", "scale", "deployment", DEPLOYMENT, "--replicas=0"],
        check=True,
        capture_output=True,
        text=True
    )

    time.sleep(3)

print()
print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "trial",
            "request_time_utc",
            "ready_time_utc",
            "request_to_ready_ms",
            "pod_name",
            "node_name",
            "cache_state",
            "p50_prediction_ms",
            "p90_prediction_ms",
            "status"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

successful = [
    r["request_to_ready_ms"]
    for r in rows
    if r["status"] == "SUCCESS"
]

print(f"Successful trials : {len(successful)}/{TRIALS}")

if successful:
    print(f"Minimum latency   : {min(successful):.2f} ms")
    print(f"Maximum latency   : {max(successful):.2f} ms")
    print(
        f"Mean latency      : "
        f"{sum(successful) / len(successful):.2f} ms"
    )

    sorted_values = sorted(successful)
    median = sorted_values[len(sorted_values) // 2]

    print(f"Median latency    : {median:.2f} ms")

print()
print(f"Saved: {OUTPUT}")
print()
print("IMPORTANT:")
print("This is a controlled warm-cache validation.")
print("No StartScale replicas were automatically controlled.")