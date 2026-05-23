"""Measure end-to-end /predict latency (p50 / p95 / p99).

Sends N sequential POST /predict requests against a running API and
sorts the client-side wall-clock timings. Reports percentiles + mean,
plus the same percentiles read from the FastAPI Prometheus histogram
(`http_request_duration_seconds`) so you can cross-check client-side
vs server-side.

Requires the API to be running:

    python demo/launch.py up     # or: make demo / make serve

Then from the repo root:

    poetry run python scripts/measure_latency.py
"""

from __future__ import annotations

import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PREDICT_URL = "http://localhost:3000/predict"
METRICS_URL = "http://localhost:3000/metrics"
SAMPLE = Path("demo/demo_data/cat_sample_1.jpg")
N_WARMUP = 10
N_RUN = 200


def _post_predict(img: bytes) -> tuple[int, float]:
    boundary = "----catopsboundary"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="cat.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + img
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        PREDICT_URL,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()
        status = resp.status
    t1 = time.perf_counter()
    return status, (t1 - t0) * 1000


def _pct(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(
        len(sorted_values) - 1, int(round((p / 100.0) * (len(sorted_values) - 1)))
    )
    return sorted_values[idx]


def _server_percentiles_from_prom() -> dict[str, float] | None:
    """Compute p50/p95/p99 from the Prometheus histogram exposed at /metrics.

    Parses `http_request_duration_seconds_bucket{...handler="/predict",...}`
    cumulative counts and applies linear interpolation inside the matched
    bucket. Returns None if the histogram isn't present (e.g. instrumentator
    disabled).
    """
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=5) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return None

    buckets: list[tuple[float, float]] = []  # (upper_bound_seconds, cumulative_count)
    total = None
    for line in text.splitlines():
        if "http_request_duration_seconds_bucket{" not in line:
            continue
        if 'handler="/predict"' not in line:
            continue
        # Match: ...le="0.05"} 12
        try:
            le_str = line.split('le="', 1)[1].split('"', 1)[0]
            count_str = line.rsplit(" ", 1)[1]
            le = float("inf") if le_str == "+Inf" else float(le_str)
            buckets.append((le, float(count_str)))
        except (IndexError, ValueError):
            continue
    for line in text.splitlines():
        if (
            line.startswith("http_request_duration_seconds_count{")
            and 'handler="/predict"' in line
        ):
            try:
                total = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                pass

    if not buckets or total is None or total == 0:
        return None
    buckets.sort()

    def quantile(q: float) -> float:
        target = q * total
        prev_le, prev_cum = 0.0, 0.0
        for le, cum in buckets:
            if cum >= target:
                # Linear interpolation inside the bucket.
                span = cum - prev_cum
                if span <= 0 or le == float("inf"):
                    return prev_le * 1000
                frac = (target - prev_cum) / span
                return (prev_le + frac * (le - prev_le)) * 1000
            prev_le, prev_cum = le, cum
        return buckets[-1][0] * 1000

    return {"p50": quantile(0.50), "p95": quantile(0.95), "p99": quantile(0.99)}


def main() -> None:
    if not SAMPLE.exists():
        sys.exit(f"Sample image missing: {SAMPLE}")
    img = SAMPLE.read_bytes()

    print(f"[warmup] {N_WARMUP} requests...")
    for _ in range(N_WARMUP):
        status, _ = _post_predict(img)
        if status != 200:
            sys.exit(f"Warmup failed: HTTP {status}")

    print(f"[run] {N_RUN} sequential requests...")
    latencies: list[float] = []
    for i in range(N_RUN):
        status, ms = _post_predict(img)
        if status != 200:
            sys.exit(f"Run failed at i={i}: HTTP {status}")
        latencies.append(ms)
    latencies.sort()

    mean_ms = statistics.mean(latencies)
    p50, p95, p99 = _pct(latencies, 50), _pct(latencies, 95), _pct(latencies, 99)

    print()
    print(f"Client-side end-to-end (n={N_RUN}, single-threaded)")
    print(f"  mean : {mean_ms:7.1f} ms")
    print(f"  p50  : {p50:7.1f} ms")
    print(f"  p95  : {p95:7.1f} ms")
    print(f"  p99  : {p99:7.1f} ms")

    server = _server_percentiles_from_prom()
    if server:
        print()
        print("Server-side (from /metrics http_request_duration_seconds histogram)")
        print(f"  p50  : {server['p50']:7.1f} ms")
        print(f"  p95  : {server['p95']:7.1f} ms")
        print(f"  p99  : {server['p99']:7.1f} ms")
    else:
        print("\n(server-side histogram unavailable — skipped)")


if __name__ == "__main__":
    main()
