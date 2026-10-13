import csv
import random
import time
from pathlib import Path

from locust import HttpUser, task, LoadTestShape


TRACE_FILE = Path(__file__).parent / "azure_replay_20min.csv"

SAFE_MIN_RPS = 25.0
SAFE_MAX_RPS = 40.0
REPLAY_USERS = 20


def load_trace():
    rows = []

    with open(TRACE_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                (
                    int(row["second"]),
                    float(row["requests_per_second"]),
                )
            )

    if not rows:
        raise RuntimeError("Azure replay trace is empty")

    values = [rps for _, rps in rows]
    rmin = min(values)
    rmax = max(values)

    scaled = []

    for second, rps in rows:
        normalized = (rps - rmin) / (rmax - rmin)

        target_rps = SAFE_MIN_RPS + normalized * (
            SAFE_MAX_RPS - SAFE_MIN_RPS
        )

        scaled.append((second, target_rps))

    return scaled


TRACE = load_trace()

print(
    f"Loaded Azure trace: {len(TRACE)} seconds | "
    f"original RPS: {min(x[1] for x in TRACE):.1f}-"
    f"{max(x[1] for x in TRACE):.1f} | "
    f"scaled RPS: {SAFE_MIN_RPS:.1f}-"
    f"{SAFE_MAX_RPS:.1f}"
)


class AzureReplayUser(HttpUser):

    def on_start(self):
        # Each user records its own start time.
        self.replay_start = time.monotonic()

        # Small random offset prevents all 20 users from firing
        # at exactly the same instant.
        time.sleep(random.uniform(0.0, 0.2))

    @task
    def replay_workload(self):

        elapsed = time.monotonic() - self.replay_start

        index = int(elapsed)

        if index >= len(TRACE):
            return

        target_rps = TRACE[index][1]

        # Distribute target RPS across the fixed user pool.
        per_user_rps = target_rps / REPLAY_USERS

        if per_user_rps <= 0:
            time.sleep(0.1)
            return

        interval = 1.0 / per_user_rps

        request_start = time.monotonic()

        self.client.get(
            "/work?ms=50",
            name="/work"
        )

        elapsed_request = time.monotonic() - request_start

        # Pace the next request.
        sleep_time = interval - elapsed_request

        if sleep_time > 0:
            time.sleep(sleep_time)


class AzureReplayShape(LoadTestShape):

    time_limit = len(TRACE)

    def tick(self):

        elapsed = int(self.get_run_time())

        if elapsed >= self.time_limit:
            return None

        return REPLAY_USERS, REPLAY_USERS
