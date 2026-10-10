"""
StartScale v2 - Pilot Dataset Traffic Replay (Step 7)

Replays a preprocessed trace (default: NASA Jul95 per-second arrivals) against
the target app's /work endpoint, using Locust's LoadTestShape to drive the
number of concurrent simulated users up and down to approximate the trace's
request rate over time.

This does NOT try to hit the exact historical RPS (NASA peaked at ~20 RPS,
your target app can handle far more) -- it RESCALES the trace to a pilot-scale
range so you get a small, fast, repeatable pilot run before committing to a
long/expensive full-scale replay in the later "larger dataset" step.

Usage:
    pip install locust pandas pyarrow
    locust -f locustfile.py --host http://localhost:8080 --headless \
        --run-time 5m --csv pilot_run

    (run `kubectl port-forward svc/startscale-target 8080:80` in another
    window first, and start measure_harness.py in a third window at the
    same time, so you capture readiness latency + replica counts alongside
    this traffic run)

Env vars:
    TRACE_PATH   path to the arrivals parquet file (default: nasa_jul95_arrivals_1s.parquet)
    PILOT_MINUTES  how many minutes of the trace to replay, compressed to run in this long (default: 5)
    MAX_USERS    the peak simulated user count the trace's peak RPS maps to (default: 20)
    WORK_MS      how much synthetic CPU work each /work request asks for (default: 50)
"""

import os
import time
import pandas as pd
from locust import HttpUser, task, between
from locust import LoadTestShape

TRACE_PATH = os.environ.get("TRACE_PATH", "nasa_jul95_arrivals_1s.parquet")
PILOT_MINUTES = float(os.environ.get("PILOT_MINUTES", "5"))
MAX_USERS = int(os.environ.get("MAX_USERS", "20"))
WORK_MS = int(os.environ.get("WORK_MS", "50"))


def _load_shape_schedule():
    """Loads the trace and produces a list of (duration_s, user_count) steps,
    rescaling the trace's own RPS range onto [1, MAX_USERS], and time-compressing
    it so the whole trace plays out over PILOT_MINUTES minutes regardless of the
    trace's original duration."""
    df = pd.read_parquet(TRACE_PATH)
    # Expect a column with per-second request counts; try common names.
    rate_col = None
    for candidate in ("requests_per_second", "rps", "count", "requests"):
        if candidate in df.columns:
            rate_col = candidate
            break
    if rate_col is None:
        # fall back to the first numeric column
        numeric_cols = df.select_dtypes("number").columns
        if len(numeric_cols) == 0:
            raise ValueError(f"No numeric column found in {TRACE_PATH} to use as rate.")
        rate_col = numeric_cols[0]

    rates = df[rate_col].fillna(0).to_numpy()
    if len(rates) == 0:
        raise ValueError(f"Trace {TRACE_PATH} is empty.")

    rmin, rmax = rates.min(), rates.max()
    span = (rmax - rmin) or 1.0

    # Downsample/compress: pick N steps evenly across the trace so the pilot
    # run finishes in PILOT_MINUTES minutes with ~2s per step.
    step_seconds = 2.0
    total_pilot_seconds = PILOT_MINUTES * 60.0
    n_steps = max(1, int(total_pilot_seconds / step_seconds))
    idxs = [int(i * (len(rates) - 1) / max(1, n_steps - 1)) for i in range(n_steps)]

    schedule = []
    for i in idxs:
        r = rates[i]
        scaled_users = 1 + int(((r - rmin) / span) * (MAX_USERS - 1))
        schedule.append((step_seconds, scaled_users))
    return schedule


_SCHEDULE = None


def get_schedule():
    global _SCHEDULE
    if _SCHEDULE is None:
        _SCHEDULE = _load_shape_schedule()
        total_s = sum(s[0] for s in _SCHEDULE)
        print(f"[locustfile] loaded {len(_SCHEDULE)} steps from {TRACE_PATH}, "
              f"pilot run will last ~{total_s:.0f}s, users range 1..{MAX_USERS}")
    return _SCHEDULE


class TracedLoadShape(LoadTestShape):
    """Drives Locust's user count over time according to the rescaled trace."""

    def tick(self):
        schedule = get_schedule()
        run_time = self.get_run_time()
        cumulative = 0.0
        for duration, users in schedule:
            cumulative += duration
            if run_time < cumulative:
                return (users, users)  # (user_count, spawn_rate)
        return None  # stop -- schedule exhausted


class TargetAppUser(HttpUser):
    wait_time = between(0.1, 0.3)

    @task
    def hit_work_endpoint(self):
        self.client.get(f"/work?ms={WORK_MS}", name="/work")
