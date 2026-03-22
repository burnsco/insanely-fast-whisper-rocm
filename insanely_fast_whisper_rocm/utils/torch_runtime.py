"""Utilities for importing PyTorch in ROCm-aware environments.

This module keeps package imports lightweight and test-friendly when the host
does not currently provide the ROCm shared libraries required by a ROCm PyTorch
wheel. Runtime paths that truly need PyTorch can call
``ensure_torch_runtime()`` to raise a clear error.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

TorchLike = ModuleType | SimpleNamespace


class TorchRuntimeUnavailableError(RuntimeError):
    """Raised when the configured PyTorch runtime cannot be imported."""


def _build_torch_placeholder() -> SimpleNamespace:
    """Return a minimal torch-like object for import-time compatibility.

    Returns:
        A namespace that exposes the small subset of torch attributes used by
        tests and import-time package wiring.
    """
    cuda_namespace = SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        empty_cache=lambda: None,
    )
    mps_namespace = SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    return SimpleNamespace(
        cuda=cuda_namespace,
        backends=SimpleNamespace(mps=mps_namespace),
        mps=mps_namespace,
        version=SimpleNamespace(hip=None),
        float16="float16",
        float32="float32",
    )


def _load_torch() -> tuple[TorchLike, BaseException | None]:
    """Import torch or return a lightweight placeholder.

    Returns:
        A tuple of the resolved torch object and the import error, if any.
    """
    try:
        import torch as imported_torch
    except (ImportError, OSError) as exc:
        return _build_torch_placeholder(), exc
    return imported_torch, None


torch, TORCH_IMPORT_ERROR = _load_torch()


def ensure_torch_runtime() -> None:
    """Raise a clear error when the active PyTorch runtime is unavailable.

    Raises:
        TorchRuntimeUnavailableError: If torch failed to import on this host.
    """
    if TORCH_IMPORT_ERROR is None:
        return

    raise TorchRuntimeUnavailableError(
        "PyTorch could not be imported with the active ROCm dependency set. "
        "Install the ROCm 7.2 runtime libraries on the host, or run inside the "
        "supported ROCm container image before starting the app."
    ) from TORCH_IMPORT_ERROR


__all__ = [
    "TORCH_IMPORT_ERROR",
    "TorchRuntimeUnavailableError",
    "ensure_torch_runtime",
    "torch",
]
