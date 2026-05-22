#!/usr/bin/env bash
# Cross-platform demo launcher - macOS/Linux entry point.
# Usage:  ./demo.sh [up|down|logs|reset|fetch-model]   (default: up)
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
    exec python3 demo/launch.py "$@"
elif command -v python >/dev/null 2>&1; then
    exec python demo/launch.py "$@"
else
    echo "Python 3 not found. Install Python 3.12+ and re-run." >&2
    exit 1
fi
