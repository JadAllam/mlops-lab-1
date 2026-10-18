# Stage 1 - build the virtual environment from the lockfile only, so this layer
# is cached until pyproject.toml or uv.lock actually change.
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2 - runtime. Only the finished venv and the source come across, so the
# uv binary and the build caches stay out of the final image.
FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
