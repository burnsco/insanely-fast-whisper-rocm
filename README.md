# Insanely Fast Whisper ROCm

ROCm-first Whisper transcription for AMD GPUs with three supported surfaces:

- FastAPI
- CLI
- optional Gradio WebUI

This repo now targets a single accelerator stack: ROCm 7.2 on Ubuntu 22.04 with
Python 3.10 (recommended baseline for PyTorch/ROCm compatibility). The project shape is simplified around a single dependency manager (`uv`) and a single GPU stack, with Docker as the primary runtime surface. The API and CLI are the main supported surfaces, with the Gradio UI as an optional companion.

## What changed

- ROCm 7.2 is the only supported GPU stack.
- `uv` is the only supported dependency manager.
- Docker is API-first by default.
- Gradio remains available as an optional companion UI.
- Generated `requirements*.txt` exports are gone to avoid dependency drift.

## ROCm baseline

The repo is aligned to the ROCm 7.2 support matrix from AMD:

- PyTorch `2.9.1`
- Triton `3.5.1`
- ROCm `7.2`

The current Radeon wheel index also provides `torchaudio 2.9.0` for ROCm 7.2,
so that is the pinned audio wheel used by the project.

## Local setup with uv

Install the project and ROCm dependencies:

```bash
uv sync --extra rocm --group dev --group bench
```

Run the standard checks:

```bash
uv run ruff check --no-cache .
uv run ruff format --check .
uv run ty check insanely_fast_whisper_rocm
uv run pytest --maxfail=1 -q
uv run pytest --cov=. --cov-report=term-missing:skip-covered --cov-report=xml
```

## Configuration

Copy [`.env.example`](/home/cburns/apps/insanely-fast-whisper-rocm/.env.example)
to `~/.config/insanely-fast-whisper-rocm/.env`.

Key settings:

- `WHISPER_MODEL`: model to load
- `WHISPER_DEVICE`: use `0` for the first ROCm GPU
- `WHISPER_BATCH_SIZE`: inference batch size
- `PYTORCH_ALLOC_CONF`: ROCm allocator tuning
- `HF_TOKEN`: optional token for gated Hugging Face models
- `SUBTITLE_SYNC_DEFAULT`: enable ALASS subtitle sync for generated SRT output
- `ALASS_BINARY`: executable name/path for the ALASS CLI (must be in `PATH`)

For subtitle synchronization, install `alass` on the host/container and ensure
`ffmpeg`/`ffprobe` are available. This project expects a host-provided `alass`
binary rather than bundling it.

PyTorch still uses `cuda:N` device strings internally on ROCm builds, so a user
value like `WHISPER_DEVICE=0` maps to the first HIP-backed GPU.

## Running the app

Start the API:

```bash
uv run insanely-fast-whisper-rocm
```

Start the WebUI:

```bash
uv run insanely-fast-whisper-webui
```

Use the CLI:

```bash
uv run insanely-fast-whisper-cli transcribe path/to/audio.mp3
uv run insanely-fast-whisper-cli translate path/to/audio.mp3
```

## Make Commands

Use `make help` to see all available shortcuts.

Common workflows:

```bash
make dev      # run API + WebUI in dev mode with hot reloading
make dev-bg   # same as above, but detached
make dev-logs # tail API + WebUI logs
make dev-down # stop and remove dev containers + volumes
```

The `dev` targets use `docker-compose.dev.yaml`, and include the `webui` profile so both services start together.

## Docker

The repo ships one multi-stage `Dockerfile` with `runtime` and `dev` targets.

Production-style API service:

```bash
docker compose up --build -d api
```

Optional Gradio UI:

```bash
docker compose --profile webui up --build -d webui
```

Development target:

```bash
docker compose -f docker-compose.dev.yaml up --build api
docker compose -f docker-compose.dev.yaml --profile webui up --build webui
```

Both compose files map ROCm devices and mount the Hugging Face cache. The API is
the default runtime surface; the UI is intentionally optional.

## Project shape

- [pyproject.toml](/home/cburns/apps/insanely-fast-whisper-rocm/pyproject.toml): single dependency source of truth
- [Dockerfile](/home/cburns/apps/insanely-fast-whisper-rocm/Dockerfile): runtime + dev images
- [docker-compose.yaml](/home/cburns/apps/insanely-fast-whisper-rocm/docker-compose.yaml): API-first runtime compose
- [docker-compose.dev.yaml](/home/cburns/apps/insanely-fast-whisper-rocm/docker-compose.dev.yaml): dev compose using the `dev` target

## Notes

- The app keeps CPU fallback paths for resilience, but the supported GPU target is ROCm.
- The Gradio UI is kept intentionally thin so the API and CLI remain the primary stable surfaces.
