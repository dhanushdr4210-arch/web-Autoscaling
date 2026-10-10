import csv
import datetime
import subprocess
import time


DEPLOYMENT = "startscale-event"
LABEL = "app=startscale-event"
WORKER = "autoscaling-worker3"
IMAGE = "localhost:5000/startscale-target:v1"

TRIALS = 10
OUTPUT = "startscale_10_cold_trials.csv"

P50_PREDICTION_MS = 6731.83
P90_PREDICTION_MS = 7615.14

rows = []

print("=" * 80)
print("STARTSCALE 10-TRIAL COLD-CACHE VALIDATION")
print("=" * 80)

# Make sure deployment starts at zero
subprocess.run(
    ["kubectl", "scale", "deployment", DEPLOYMENT, "--replicas=0"],
    check=True,
    capture_output=True,
    text=True
)

time.sleep(3)

for trial in range(1, TRIALS + 1):

    print()
    print("-" * 80)
    print(f"TRIAL {trial}/{TRIALS}")
    print("-" * 80)

    # Ensure deployment is scaled down
    subprocess.run(
        ["kubectl", "scale", "deployment", DEPLOYMENT, "--replicas=0"],
        check=True,
        capture_output=True,
        text=True
    )

    time.sleep(3)

    # Remove image from target worker
    remove = subprocess.run(
        [
            "docker",
            "exec",
            WORKER,
            "crictl",
            "rmi",
            IMAGE
        ],
        capture_output=True,
        text=True
    )

    if remove.returncode != 0:
        print("Image removal warning:")
        print(remove.stderr.strip())

    # Verify image is absent
    check = subprocess.run(
        [
            "docker",
            "exec",
            WORKER,
            "crictl",
            "images"
        ],
        capture_output=True,
        text=True
    )

    if IMAGE in check.stdout:
        print("ERROR: Image is still present on worker3.")
        print("Skipping this trial.")

        rows.append({
            "trial": trial,
            "request_time_utc": "",
            "ready_time_utc": "",
            "request_to_ready_ms": "",
            "pod_name": "",
            "node_name": WORKER,
            "cache_state": "cold_failed",
            "p50_prediction_ms": P50_PREDICTION_MS,
            "p90_prediction_ms": P90_PREDICTION_MS,
            "status": "FAILED_CACHE_CHECK"
        })

        continue

    print("Cache state: COLD")

    # Record exact request time
    request_time = datetime.datetime.now(datetime.timezone.utc)

    print("Request time:", request_time.isoformat())

    # Trigger scale-up
    subprocess.run(
        [
            "kubectl",
            "scale",
            "deployment",
            DEPLOYMENT,
            "--replicas=1"
        ],
        check=True,
        capture_output=True,
        text=True
    )

    # Wait for Ready
    result = subprocess.run(
        [
            "kubectl",
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            LABEL,
            "--timeout=120s"
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
            "node_name": WORKER,
            "cache_state": "cold",
            "p50_prediction_ms": P50_PREDICTION_MS,
            "p90_prediction_ms": P90_PREDICTION_MS,
            "status": "FAILED_READY"
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

    # Get pod name
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

    # Get actual node
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
        "cache_state": "cold",
        "p50_prediction_ms": P50_PREDICTION_MS,
        "p90_prediction_ms": P90_PREDICTION_MS,
        "status": "SUCCESS"
    })

    # Scale down
    subprocess.run(
        [
            "kubectl",
            "scale",
            "deployment",
            DEPLOYMENT,
            "--replicas=0"
        ],
        check=True,
        capture_output=True,
        text=True
    )

    time.sleep(3)


print()
print("=" * 80)
print("SAVING COLD-CACHE RESULTS")
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
print("Every trial verifies the image is absent before scaling.")
print("This produces controlled cold-cache startup measurements.")