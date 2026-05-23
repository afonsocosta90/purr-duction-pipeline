"""Measure cold-start latency of the FastAPI inference container.

Three numbers, all from the user-visible perspective (i.e. wall-clock
from outside the container — what someone hitting the API after
`docker compose up` actually waits for):

  1. *Container ready* — `docker compose up -d api` to first
     `GET /health` returning 200. Includes Docker container start,
     uvicorn boot, FastAPI lifespan handler, and `load_model()`.
  2. *First /predict (warm-up)* — wall-clock latency of the very first
     `POST /predict` call against the fresh process. This captures any
     JIT / lazy-init work that doesn't happen during model load.
  3. *Total cold-start* — sum of the two above.

Run from the repo root with Docker Desktop running:

    poetry run python scripts/measure_cold_start.py
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COMPOSE = ["docker", "compose", "-f", "demo/docker-compose.demo.yml"]
HEALTH_URL = "http://localhost:3000/health"
PREDICT_URL = "http://localhost:3000/predict"
SAMPLE = Path("demo/demo_data/cat_sample_1.jpg")
POLL_INTERVAL_S = 0.05
TIMEOUT_S = 120.0


def _health_ok() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=1) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _post_predict() -> int:
    boundary = "----catopsboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="cat.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + SAMPLE.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        PREDICT_URL,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status


def main() -> None:
    if not SAMPLE.exists():
        sys.exit(f"Sample image missing: {SAMPLE}")

    print("[1/3] Tearing down API container...")
    subprocess.run(COMPOSE + ["down", "api"], check=False)

    print("[2/3] Starting API container (timing cold start)...")
    t0 = time.perf_counter()
    subprocess.run(COMPOSE + ["up", "-d", "api"], check=True)

    deadline = t0 + TIMEOUT_S
    while not _health_ok():
        if time.perf_counter() > deadline:
            sys.exit(f"Timed out after {TIMEOUT_S:.0f}s waiting for /health")
        time.sleep(POLL_INTERVAL_S)
    t_ready = time.perf_counter()

    print("[3/3] Sending first /predict...")
    t_p0 = time.perf_counter()
    status = _post_predict()
    t_p1 = time.perf_counter()
    if status != 200:
        sys.exit(f"First /predict returned HTTP {status}")

    container_ready_ms = (t_ready - t0) * 1000
    first_predict_ms = (t_p1 - t_p0) * 1000
    total_ms = (t_p1 - t0) * 1000

    print()
    print(f"Container up -> /health 200:  {container_ready_ms:8.1f} ms   (model load)")
    print(f"First /predict (warm-up):     {first_predict_ms:8.1f} ms")
    print(f"Total cold-start to served:   {total_ms:8.1f} ms")


if __name__ == "__main__":
    main()
