# deploy/Dockerfile
#
# Multi-stage build for the observability dashboard (the deployable web
# surface of this project -- the harness/eval layers are libraries invoked
# by CI or by whatever process runs the agent, not a standalone service).
#
# Build from the repo root so the build context includes harness/, eval/,
# and observability/:
#   docker build -f deploy/Dockerfile -t harborline .
# Run:
#   docker run -p 8000:8000 -e GROQ_API_KEY=... harborline

# ---- Stage 1: install dependencies into a venv ----
FROM python:3.12-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: runtime image ----
FROM python:3.12-slim AS runtime

# Minimal OS deps: curl for the healthcheck, nothing else. Keeps the
# runtime image's attack surface small compared to installing full build
# tooling into the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY harness/ ./harness/
COPY eval/ ./eval/
COPY observability/ ./observability/

RUN mkdir -p /app/eval/reports && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "observability.dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]