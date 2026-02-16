# syntax=docker/dockerfile:1.7

# ---------- Builder ----------
FROM python:3.13-slim AS builder

# Install uv (copy the static binary from Astral's image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Speed & determinism
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 1) Install ONLY transitive dependencies (cacheable)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable

# 2) Copy the project and install the app itself (non-editable)
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# ---------- Runtime ----------
FROM python:3.13-slim AS runtime

# Use the built virtualenv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Use the venv by default
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Run as non-root
RUN useradd --create-home appuser
USER appuser

WORKDIR /app
EXPOSE 8000

# If your ASGI app is app.main:app (src/app/main.py), this Just Works:
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
