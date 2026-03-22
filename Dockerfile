# Use AMD's validated ROCm 7.2 PyTorch runtime for Python 3.10.
FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1

LABEL org.opencontainers.image.source https://github.com/burnsco/insanely-fast-whisper-rocm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV TZ=Europe/Amsterdam
ENV ROCM_PATH=/opt/rocm
ENV PATH="/app/.venv/bin:${PATH}"

# Install OS-level dependencies required by the media pipeline.
RUN apt-get update -y && apt-get upgrade -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --no-cache-dir uv

# Set the working directory in the container
WORKDIR /app

# Copy the project metadata needed to install dependencies.
COPY pyproject.toml /app/
COPY uv.lock /app/
COPY openapi.yaml /app/

# Install only the application dependencies; torch/torchaudio come from the
# validated ROCm base image.
RUN uv sync --locked --no-dev --no-install-project

COPY ./insanely_fast_whisper_rocm /app/insanely_fast_whisper_rocm/

RUN uv sync --locked --no-dev --no-editable

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV TORCHAUDIO_USE_SOUNDFILE=1

# Expose default internal ports (API/WebUI). Actual bindings are controlled by Compose.
EXPOSE 8888
EXPOSE 7860

# Use the package entrypoint so host/port are controlled by env vars (API_HOST/API_PORT).
CMD ["insanely-fast-whisper-rocm"]
