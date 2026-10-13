"""
StartScale v2 - Larger Dataset Traffic Replay (Step 8)

Extends the Step 7 pilot script: same rescale-and-compress approach, but now
supports any of your preprocessed traces via TRACE_PATH + TRACE_RATE_COL, runs
longer by default, and allows a higher MAX_USERS ceiling since this is meant
to actually stress the cluster (not just prove the pipeline works).

Usage examples:
    # NASA (per-second arrivals)
    $env:TRACE_PATH="nasa_jul95_arrivals_1s.parquet"
    $env:PILOT_MINUTES="20"
    $env:MAX_USERS="40"
    locust -f locustfile.py --host http://localhost:8080 --headless --run-time 20m --csv run_nasa

    # WorldCup98 (per-minute counts -- note different time resolution)
    $env:TRACE_PATH="worldcup98_full_1min.parquet"
    $env:PILOT_MINUTES="20"
    $env:MAX_USERS="40"
    locust -f locustfile.py --host http://localhost:8080 --headless --run-time 20m --csv run_worldcup98

    # Azure Functions (per-minute aggregate invocation counts)
    $env:TRACE_PATH="azure_functions_aggregate_1min.parquet"
    $env:PILOT_MINUTES="20"
    $env:MAX_USERS="40"
    locust -f locustfile.py --host http://localhost:8080 --headless --run-time 20m --csv run_azure

Env vars:
    TRACE_PATH     path to the trace parquet file (default: nasa_jul95_arrivals_1s.parquet)
    TRACE_RATE_COL force a specific column name to use as the rate signal (optional;
                   auto-detected otherwise -- see _load_shape_schedule)
    PILOT_MINUTES  minutes to compress/stretch the trace into (default: 20 for this step)
    MAX_USERS      peak simulated user count the trace's peak rate maps to (default: 40)
    WORK_MS        synthetic CPU work per /work request, in ms (default: 50)
    STEP_SECONDS   how many seconds each load-shape step lasts (default: 5 -- coarser
                   than the pilot's 2s, since longer runs don't need per-2s granularity)
"""

import os
import pandas as pd
from locust import HttpUser, task, between
from locust import LoadTestShape

TRACE_PATH = os.environ.get("TRACE_PATH", "nasa_jul95_arrivals_1s.parquet")
TRACE_RATE_COL = os.environ.get("TRACE_RATE_COL", "")
PILOT_MINUTES = float(os.environ.get("PILOT_MINUTES", "20"))
MAX_USERS = int(os.environ.get("MAX_USERS", "40"))
WORK_MS = int(os.environ.get("WORK_MS", "50"))
STEP_SECONDS = float(os.environ.get("STEP_SECONDS", "5"))

# Known rate-column names across your preprocessed traces -- checked in order.
KNOWN_RATE_COLS = [
    "requests_per_second",  # NASA arrivals_1s
    "rps",
    "count",                # WorldCup98 full_1min
    "requests",
    "invocation_count",     # Azure Functions aggregate
    "aggregate_invocations",
    "requests_per_minute",  # WorldCup98 full_1min
    "invocations",
    "invocations_per_minute",
    "total_invocations",
]


def _load_shape_schedule():
    df = pd.read_parquet(TRACE_PATH)

    rate_col = TRACE_RATE_COL or None
    if rate_col and rate_col not in df.columns:
        raise ValueError(f"TRACE_RATE_COL='{rate_col}' not found in {TRACE_PATH}. "
                          f"Available columns: {list(df.columns)}")
    if not rate_col:
        for candidate in KNOWN_RATE_COLS:
            if candidate in df.columns:
                rate_col = candidate
                break
    if not rate_col:
        index_like = {"minute", "index", "timestamp", "time", "second", "idx", "level_0"}
        numeric_cols = [c for c in df.select_dtypes("number").columns
                        if c.lower() not in index_like]
        if len(numeric_cols) == 0:
            raise ValueError(f"No usable numeric column found in {TRACE_PATH}. "
                              f"Available columns: {list(df.columns)}")
        # pick the column with the highest variance (most 'bursty' signal),
        # skipping monotonic index-like columns
        candidates = [c for c in numeric_cols if not df[c].is_monotonic_increasing]
        pool = candidates or numeric_cols
        rate_col = max(pool, key=lambda c: float(df[c].fillna(0).var()))
        print(f"[locustfile] WARNING: no known rate column found, "
              f"auto-picked '{rate_col}' from {list(df.columns)}. "
              f"Set TRACE_RATE_COL explicitly if this is wrong.")

    rates = df[rate_col].fillna(0).to_numpy()
    if len(rates) == 0:
        raise ValueError(f"Trace {TRACE_PATH} is empty.")

    rmin, rmax = rates.min(), rates.max()
    span = (rmax - rmin) or 1.0

    total_seconds = PILOT_MINUTES * 60.0
    n_steps = max(1, int(total_seconds / STEP_SECONDS))
    idxs = [int(i * (len(rates) - 1) / max(1, n_steps - 1)) for i in range(n_steps)]

    schedule = []
    for i in idxs:
        r = rates[i]
        scaled_users = 1 + int(((r - rmin) / span) * (MAX_USERS - 1))
        schedule.append((STEP_SECONDS, scaled_users))
    return schedule, rate_col


_SCHEDULE = None
_RATE_COL = None


def get_schedule():
    global _SCHEDULE, _RATE_COL
    if _SCHEDULE is None:
        _SCHEDULE, _RATE_COL = _load_shape_schedule()
        total_s = sum(s[0] for s in _SCHEDULE)
        print(f"[locustfile] trace={TRACE_PATH} rate_col={_RATE_COL} "
              f"steps={len(_SCHEDULE)} run_length~{total_s:.0f}s users_range=1..{MAX_USERS}")
    return _SCHEDULE


class TracedLoadShape(LoadTestShape):
    def tick(self):
        schedule = get_schedule()
        run_time = self.get_run_time()
        cumulative = 0.0
        for duration, users in schedule:
            cumulative += duration
            if run_time < cumulative:
                return (users, users)
        return None


class TargetAppUser(HttpUser):
    wait_time = between(0.1, 0.3)

    @task
    def hit_work_endpoint(self):
        self.client.get(f"/work?ms={WORK_MS}", name="/work")
