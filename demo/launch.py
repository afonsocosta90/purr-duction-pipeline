#!/usr/bin/env python3
"""Cross-platform launcher for the "Am I a Cat?" demo stack.

One command, every OS - no Make, no Bash, no POSIX shell required. Runs
identically from Windows PowerShell/cmd and macOS/Linux terminals.

    python demo/launch.py             # fetch model + start the 4-service stack
    python demo/launch.py down        # stop the stack
    python demo/launch.py logs        # follow service logs
    python demo/launch.py reset       # tear down volumes + images, rebuild
    python demo/launch.py fetch-model # just download the model checkpoint

Only the Python standard library is used, so this runs before any
`poetry install`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Avoid UnicodeEncodeError crashes on legacy Windows code pages.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_REL = "demo/docker-compose.demo.yml"

MODEL_PATH = REPO_ROOT / "models" / "best_model.pt"
MODEL_URL = (
    "https://github.com/afonsocosta90/purr-duction-pipeline"
    "/releases/download/v1.1/best_model.pt"
)
MODEL_SHA256 = "793a0817c6d38e95b5eaaddb5bb5734b0a597296d5f469f6e9437d3abc46fa47"

# The API reads features_config.json at startup to build its inference
# transform. It is a `features` pipeline output and is not committed
# (data/processed/ is gitignored), so a fresh clone has nothing to bind-mount
# into the API container — without it the container crash-loops on startup.
# These values are the canonical config for the released model: they mirror
# configs/data.yaml plus the ImageNet normalization stats hardcoded in
# features/build_features.py.
FEATURES_CONFIG_PATH = REPO_ROOT / "data" / "processed" / "features_config.json"
FEATURES_CONFIG = {
    "resize": {"target_size": [224, 224], "interpolation": "bilinear"},
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "split_seed": 42,
}

# (GHCR image, local tag the compose file expects, compose service name)
IMAGES = [
    ("ghcr.io/afonsocosta90/purr-duction-pipeline:latest", "catops-api:demo", "api"),
    (
        "ghcr.io/afonsocosta90/purr-duction-pipeline-demo:latest",
        "catops-demo:latest",
        "streamlit",
    ),
]

SESSION_LOGS = [
    REPO_ROOT / "monitoring" / "feedback_log.csv",
    REPO_ROOT / "monitoring" / "inference_log.csv",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run `docker compose -f demo/... <args>` from the repo root."""
    return subprocess.run(
        ["docker", "compose", "-f", COMPOSE_REL, *args],
        cwd=str(REPO_ROOT),
        check=check,
    )


def ensure_docker() -> None:
    """Exit with a clear message if Docker is missing or not running."""
    if shutil.which("docker") is None:
        sys.exit(
            "[X] Docker is not installed or not on PATH.\n"
            "    Install Docker Desktop: "
            "https://www.docker.com/products/docker-desktop/\n"
            "    Then start it and re-run this command."
        )
    probe = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if probe.returncode != 0:
        sys.exit(
            "[X] Docker is installed but not running.\n"
            "    Start Docker Desktop, wait for it to finish starting, then "
            "re-run this command."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model() -> None:
    """Download the model checkpoint from the GitHub release if it is missing.

    Only downloads when absent: an existing best_model.pt may be the user's
    own freshly trained model (a DVC pipeline output), so it is never touched.
    """
    if MODEL_PATH.exists():
        print(f"[OK] Model checkpoint present - {MODEL_PATH.relative_to(REPO_ROOT)}")
        return

    print("[..] Model checkpoint missing - downloading release weights...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_name(MODEL_PATH.name + ".tmp")

    def _progress(block_num: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block_num * block_size * 100 // total)
            print(f"\r     {pct:3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, tmp, _progress)
        print()
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(f"[X] Download failed ({exc}).\n    Manual URL: {MODEL_URL}")

    digest = _sha256(tmp)
    if digest != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        sys.exit(
            "[X] Checksum mismatch - discarding download.\n"
            f"    expected {MODEL_SHA256}\n    got      {digest}"
        )

    tmp.replace(MODEL_PATH)
    print(f"[OK] Model checkpoint ready - {MODEL_PATH.relative_to(REPO_ROOT)}")


def ensure_features_config() -> None:
    """Write data/processed/features_config.json if it is missing.

    The API container reads this file at startup; without it the container
    crash-loops with FileNotFoundError. It is normally produced by the
    `features` pipeline stage, so it is written only when absent — a real
    `dvc repro` output is never clobbered.
    """
    rel = FEATURES_CONFIG_PATH.relative_to(REPO_ROOT)
    if FEATURES_CONFIG_PATH.exists():
        print(f"[OK] Features config present - {rel}")
        return

    FEATURES_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_CONFIG_PATH.write_text(json.dumps(FEATURES_CONFIG, indent=2))
    print(f"[OK] Features config written - {rel}")


def _prepare_image(ghcr_ref: str, local_tag: str, service: str) -> None:
    """Pull the pre-built GHCR image and tag it; fall back to a local build."""
    print(f"  Fetching {service} image...")
    pulled = subprocess.run(
        ["docker", "pull", ghcr_ref], capture_output=True, text=True
    ).returncode
    if pulled == 0:
        subprocess.run(["docker", "tag", ghcr_ref, local_tag], check=True)
        print(f"  [OK] Pulled {ghcr_ref}")
    else:
        print(f"  Pre-built image unavailable - building {service} locally...")
        _compose("build", service)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def up() -> None:
    print("\n=== Am I a Cat? - MLOps Portfolio Demo ===\n")
    ensure_docker()
    fetch_model()
    ensure_features_config()

    print("  Clearing previous session logs...")
    for log in SESSION_LOGS:
        log.unlink(missing_ok=True)

    for ghcr_ref, local_tag, service in IMAGES:
        _prepare_image(ghcr_ref, local_tag, service)

    _compose("up", "-d")

    print("\n  [OK] Services are starting. Available shortly at:\n")
    print("     Streamlit UI   ->  http://localhost:8501")
    print("     FastAPI docs   ->  http://localhost:3000/docs")
    print("     Grafana        ->  http://localhost:3001  (admin / catops)")
    print("     Prometheus     ->  http://localhost:9090\n")
    print("  Follow logs : python demo/launch.py logs")
    print("  Stop        : python demo/launch.py down\n")


def down() -> None:
    ensure_docker()
    _compose("down")
    for log in SESSION_LOGS:
        log.unlink(missing_ok=True)
    print("  Session logs cleared.")


def logs() -> None:
    ensure_docker()
    _compose("logs", "-f", check=False)


def reset() -> None:
    ensure_docker()
    print("[!] This removes all Prometheus and Grafana data volumes.")
    try:
        if input("    Continue? [y/N] ").strip().lower() != "y":
            print("    Aborted.")
            return
    except EOFError:
        pass  # non-interactive: proceed
    _compose("down", "-v", "--remove-orphans", check=False)
    subprocess.run(
        ["docker", "rmi", "catops-api:demo", "catops-demo:latest"],
        capture_output=True,
    )
    up()


COMMANDS = {
    "up": up,
    "down": down,
    "logs": logs,
    "reset": reset,
    "fetch-model": fetch_model,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Am I a Cat? cross-platform demo launcher."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="up",
        choices=list(COMMANDS),
        help="up (default) | down | logs | reset | fetch-model",
    )
    args = parser.parse_args()
    try:
        COMMANDS[args.command]()
    except subprocess.CalledProcessError as exc:
        sys.exit(f"[X] Command failed: {' '.join(exc.cmd)} (exit {exc.returncode})")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
