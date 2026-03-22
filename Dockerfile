# syntax=docker/dockerfile:1

FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.13_pytorch_release_2.9.1 AS base

LABEL org.opencontainers.image.source=https://github.com/burnsco/insanely-fast-whisper-rocm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    ROCM_PATH=/opt/rocm \
    TORCHAUDIO_USE_SOUNDFILE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock openapi.yaml README.md /app/

FROM base AS runtime

RUN uv sync --locked --extra rocm --no-dev --no-install-project

COPY src /app/src

RUN uv sync --locked --extra rocm --no-dev --no-editable

EXPOSE 8888 7860

CMD ["insanely-fast-whisper-rocm"]

FROM base AS dev

RUN uv sync --locked --extra rocm --group bench --group dev --no-install-project

COPY src /app/src
COPY tests /app/tests
COPY scripts /app/scripts
COPY .env.example /app/.env.example
COPY docker-compose.yaml /app/docker-compose.yaml
COPY docker-compose.dev.yaml /app/docker-compose.dev.yaml

RUN uv sync --locked --extra rocm --group bench --group dev --no-editable

EXPOSE 8888 7860

CMD ["insanely-fast-whisper-rocm"]
