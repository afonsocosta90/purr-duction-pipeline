FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Place the venv inside /app so it survives a USER switch
RUN poetry config virtualenvs.in-project true

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction

# Copy source code
COPY . .

# Drop root privileges
RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 3000

# Use the in-project venv directly — no `poetry run` needed after USER switch
CMD ["/app/.venv/bin/uvicorn", "src.catops.serving.service:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "2", "--access-log"]
