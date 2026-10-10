"""
StartScale v2 - Target Application

A minimal FastAPI service used as the scaling target in the autoscaling
experiments. It exposes:
  - GET  /healthz   -> liveness probe (always fast, no dependency on load)
  - GET  /readyz    -> readiness probe (fails until warmup is complete,
                       used to measure pod readiness latency)
  - GET  /work      -> CPU-bound endpoint that does real work proportional
                       to `ms`, so HPA/CPU metrics and request latency
                       both respond to load
  - GET  /metrics   -> Prometheus-format metrics (request count, latency,
                       in-flight requests) for the monitoring phase

Startup simulates a configurable "cold start" delay via STARTUP_DELAY_MS,
so later phases (startup-latency prediction) have a real, controllable
signal to predict and compensate for.
"""

import os
import time
import math
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse

# ---- Config (all overridable via env vars, set from the K8s Deployment) ----
STARTUP_DELAY_MS = int(os.environ.get("STARTUP_DELAY_MS", "0"))
POD_NAME = os.environ.get("POD_NAME", "unknown-pod")
NODE_NAME = os.environ.get("NODE_NAME", "unknown-node")

_ready = False
_start_time = time.time()

# ---- Minimal in-process metrics (no external deps needed for phase 4) ----
_lock = threading.Lock()
_request_count = 0
_in_flight = 0
_total_latency_s = 0.0


def _simulate_warmup():
    """Runs in a background thread so the process can still accept TCP
    connections immediately (matching real container cold-start behaviour),
    but /readyz will report not-ready until this completes."""
    global _ready
    if STARTUP_DELAY_MS > 0:
        time.sleep(STARTUP_DELAY_MS / 1000.0)
    _ready = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_simulate_warmup, daemon=True).start()
    yield


app = FastAPI(title="StartScale Target App", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    """Liveness: process is up and can serve HTTP at all."""
    return {"status": "alive", "pod": POD_NAME}


@app.get("/readyz")
def readyz(response: Response):
    """Readiness: gates traffic until simulated warmup finishes.
    Kubernetes readinessProbe should point here."""
    if not _ready:
        response.status_code = 503
        return {"status": "not-ready", "pod": POD_NAME}
    return {"status": "ready", "pod": POD_NAME}


def _cpu_burn(ms: int):
    """Busy-loop CPU work for approximately `ms` milliseconds.
    Real CPU cycles so container CPU metrics genuinely rise under load."""
    end = time.time() + (ms / 1000.0)
    x = 0.0001
    while time.time() < end:
        x = math.sqrt(x * x + 1.0)
    return x


@app.get("/work")
def work(ms: int = 50):
    """CPU-bound endpoint. `ms` controls how much synthetic work is done
    per request -- the traffic generator (Locust) varies request rate,
    this controls per-request cost."""
    global _request_count, _in_flight, _total_latency_s
    with _lock:
        _in_flight += 1
    t0 = time.time()
    ms = max(0, min(ms, 5000))  # clamp to avoid pathological requests
    result = _cpu_burn(ms)
    elapsed = time.time() - t0
    with _lock:
        _in_flight -= 1
        _request_count += 1
        _total_latency_s += elapsed
    return {
        "pod": POD_NAME,
        "node": NODE_NAME,
        "requested_ms": ms,
        "elapsed_ms": round(elapsed * 1000, 2),
        "checksum": round(result, 6),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus text-format metrics for the monitoring phase."""
    with _lock:
        count = _request_count
        in_flight = _in_flight
        total_latency = _total_latency_s
    uptime = time.time() - _start_time
    avg_latency = (total_latency / count) if count else 0.0
    lines = [
        "# HELP app_requests_total Total handled /work requests",
        "# TYPE app_requests_total counter",
        f"app_requests_total {count}",
        "# HELP app_in_flight_requests Current in-flight /work requests",
        "# TYPE app_in_flight_requests gauge",
        f"app_in_flight_requests {in_flight}",
        "# HELP app_avg_latency_seconds Average /work request latency",
        "# TYPE app_avg_latency_seconds gauge",
        f"app_avg_latency_seconds {avg_latency:.6f}",
        "# HELP app_uptime_seconds Seconds since process start",
        "# TYPE app_uptime_seconds gauge",
        f"app_uptime_seconds {uptime:.2f}",
        "# HELP app_ready Whether the pod has finished simulated warmup (1/0)",
        "# TYPE app_ready gauge",
        f"app_ready {1 if _ready else 0}",
    ]
    return "\n".join(lines) + "\n"


@app.get("/")
def root():
    return {
        "service": "startscale-target-app",
        "pod": POD_NAME,
        "node": NODE_NAME,
        "ready": _ready,
    }
