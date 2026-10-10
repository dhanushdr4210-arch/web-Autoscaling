# StartScale - Kubernetes Autoscaling Research

## Overview
Research project exploring startup-latency-aware scaling for Kubernetes workloads.

## Target Application
main.py provides these FastAPI endpoints:
- /healthz - Liveness check
- /readyz - Readiness after simulated warm-up
- /work?ms=50 - CPU-bound synthetic workload
- /metrics - Application metrics in Prometheus format

STARTUP_DELAY_MS controls simulated warm-up delay. This is not a direct measurement of actual Kubernetes container startup latency.

## Requirements
- Python
- Docker
- Kubernetes and kubectl
- kind or another Kubernetes cluster

## Install Python Dependencies
Run from the target application directory:

```powershell
python -m pip install -r requirements.txt
```

## Run Locally
Start the API server:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs to inspect the API.

## Research Components
- `model_v3.py` - Model implementation
- `train_startscale_model.py` - Model training
- `quantile_readiness_prediction.py` - Readiness prediction utilities
- `startscale_live_controller.py` - Live scaling controller
- `startscale_shadow_controller.py` - Shadow controller
- `measure_harness.py` and `measure_startup_latency.py` - Measurement utilities
- `startup_delay_analysis.py` and `tail_condition_analysis.py` - Analysis scripts
- `validate_startscale_grouped.py` and `validate_stratified_v3.py` - Validation scripts
- Kubernetes YAML files - Deployment and monitoring configurations

## Research Limitations
Model performance improvements must be demonstrated through reproducible experiments against suitable baselines. Simulated warm-up delay, pod readiness latency, and actual container startup latency are different measurements.

## Data and Secrets
Do not commit large datasets, generated experiment logs, virtual environments, or credentials unless specifically reviewed and needed.

