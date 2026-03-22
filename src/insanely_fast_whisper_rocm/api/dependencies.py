"""Dependency injection providers for FastAPI routes.

This module implements dependency injection for backend configuration,
ASR pipeline instances, and other shared resources used by the API
endpoints.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import NoReturn, cast

from insanely_fast_whisper_rocm.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_rocm.core.backend_cache import borrow_pipeline
from insanely_fast_whisper_rocm.core.pipeline import WhisperPipeline
from insanely_fast_whisper_rocm.utils import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_LENGTH,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    FileHandler,
    constants,
)


def _normalize(value: object, default: object) -> object:
    """Normalize a FastAPI parameter to its default value if applicable.

    Args:
        value: The parameter value to normalize, potentially a FastAPI param.
        default: The default value to return if normalization fails.

    Returns:
        The normalized value, either the parameter's default or the original
        value.
    """
    if hasattr(value, "__class__") and value.__class__.__module__.startswith(
        "fastapi."
    ):
        return getattr(value, "default", default)
    return value


def _normalize_str(value: object, default: str) -> str:
    """Return a normalized string value.

    Args:
        value: Value that may be a FastAPI parameter placeholder.
        default: Fallback value to use when the parameter has no default.

    Returns:
        A string suitable for backend configuration.
    """
    return cast(str, _normalize(value, default))


def _normalize_int(value: object, default: int) -> int:
    """Return a normalized integer value.

    Args:
        value: Value that may be a FastAPI parameter placeholder.
        default: Fallback value to use when the parameter has no default.

    Returns:
        An integer suitable for backend configuration.
    """
    normalized = _normalize(value, default)
    return int(cast(int | str, normalized))


def get_backend_config(
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dtype: str = "float16",
    model_chunk_length: int = DEFAULT_CHUNK_LENGTH,
) -> HuggingFaceBackendConfig:
    """Dependency to provide backend configuration for API routes.

    This function constructs a backend configuration from dependency parameters
    without borrowing a pipeline. Routes can pass the resulting config to the
    orchestrator, which owns pipeline acquisition and release.

    Args:
        model: Name of the Whisper model to use.
        device: Device ID for processing (e.g., "0" for first GPU).
        batch_size: Number of parallel audio segments to process.
        dtype: Data type for model inference ('float16' or 'float32').
        model_chunk_length: Internal chunk length for the Whisper model
            (seconds).

    Returns:
        HuggingFaceBackendConfig: Configured backend settings for a request.
    """
    return HuggingFaceBackendConfig(
        model_name=_normalize_str(model, DEFAULT_MODEL),
        device=_normalize_str(device, DEFAULT_DEVICE),
        dtype=_normalize_str(dtype, "float16"),
        batch_size=_normalize_int(batch_size, DEFAULT_BATCH_SIZE),
        chunk_length=_normalize_int(model_chunk_length, DEFAULT_CHUNK_LENGTH),
        progress_group_size=constants.DEFAULT_PROGRESS_GROUP_SIZE,
    )


def get_asr_pipeline(
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dtype: str = "float16",
    model_chunk_length: int = DEFAULT_CHUNK_LENGTH,
) -> Generator[WhisperPipeline, None, None]:
    """Dependency to provide configured ASR pipeline.

    Args:
        model: Name of the Whisper model to use.
        device: Device ID for processing (e.g., "0" for first GPU).
        batch_size: Number of parallel audio segments to process.
        dtype: Data type for model inference ('float16' or 'float32').
        model_chunk_length: Internal chunk length for the Whisper model
            (seconds).

    Yields:
        WhisperPipeline: Configured ASR pipeline instance for the request.
    """
    backend_config = get_backend_config(
        model=model,
        device=device,
        batch_size=batch_size,
        dtype=dtype,
        model_chunk_length=model_chunk_length,
    )
    with borrow_pipeline(
        backend_config,
        save_transcriptions=True,
    ) as pipeline:
        # Generator dependency: yield the pipeline, then release in teardown
        yield pipeline


# Expose ``__wrapped__`` to allow pytest monkeypatching of dependency overrides.
# FastAPI wraps callables passed to Depends internally, but when tests import the
# original function directly they may expect this attribute for easy stubbing.
# Setting it explicitly keeps the public behaviour unchanged while improving
# testability.
def _get_asr_pipeline_unwrapped() -> NoReturn:
    """Placeholder for tests to monkeypatch. Returns WhisperPipeline when patched.

    Raises:
        RuntimeError: Always raised unless this function is monkeypatched in tests.
    """
    raise RuntimeError("This placeholder should be monkeypatched in tests.")


# Assign to avoid FastAPI/inspect wrapper loop issues while providing the attribute.
get_asr_pipeline.__wrapped__ = _get_asr_pipeline_unwrapped  # type: ignore[attr-defined]


def get_file_handler() -> FileHandler:
    """Dependency to provide file handler instance.

    Returns:
        FileHandler: File handler instance for managing uploads and cleanup
    """
    return FileHandler()


# Expose ``__wrapped__`` to allow pytest monkeypatching of dependency overrides.
# FastAPI wraps callables passed to Depends internally, but when tests import the
# original function directly they may expect this attribute for easy stubbing.
# Setting it explicitly keeps the public behaviour unchanged while improving
# testability.
def _get_file_handler_unwrapped() -> NoReturn:
    """Placeholder for tests to monkeypatch. Returns FileHandler when patched.

    Raises:
        RuntimeError: Always raised unless this function is monkeypatched in tests.
    """
    raise RuntimeError("This placeholder should be monkeypatched in tests.")


# Assign to avoid FastAPI/inspect wrapper loop issues while providing the attribute.
get_file_handler.__wrapped__ = _get_file_handler_unwrapped  # type: ignore[attr-defined]
