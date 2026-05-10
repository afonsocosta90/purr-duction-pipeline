FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Place the venv at /opt/venv so a bind-mount of /app doesn't overwrite it
RUN poetry config virtualenvs.path /opt/venv

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction

# Copy source code
COPY . .

# Drop root privileges
RUN useradd --create-home --uid 1001 appuser \
    && chown -R appuser:appuser /app /opt/venv
USER appuser

EXPOSE 3000

CMD ["/opt/venv/bin/uvicorn", "src.catops.serving.service:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "2", "--access-log"]
