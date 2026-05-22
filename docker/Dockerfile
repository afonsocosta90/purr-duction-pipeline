FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Install directly into the system Python — no venv needed in a container,
# and it avoids bind-mount of /app clobbering an in-project .venv.
RUN poetry config virtualenvs.create false

# torch/torchvision are not yet tracked in pyproject.toml; install them
# directly from the PyTorch CPU wheel index before poetry resolves the rest.
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction

# Copy source code
COPY . .

# Drop root privileges
RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 3000

# Single worker: the custom Prometheus Counter/Histogram/Gauge live in a
# per-process registry, so multiple workers would split and corrupt /metrics.
CMD ["uvicorn", "src.catops.serving.service:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1", "--access-log"]
