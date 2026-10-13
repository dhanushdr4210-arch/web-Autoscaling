FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV STARTUP_DELAY_MS=0
EXPOSE 8000

# POD_NAME / NODE_NAME are injected by the Kubernetes Deployment via the
# Downward API (see the deployment manifest) -- default here so it also
# runs standalone with `docker run`.
ENV POD_NAME=local
ENV NODE_NAME=local

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
