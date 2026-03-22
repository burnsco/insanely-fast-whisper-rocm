"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest
import requests


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """Configure global warning filters for the test suite."""
    warnings.filterwarnings(
        "ignore",
        message=r"websockets\.legacy is deprecated;.*",
        category=DeprecationWarning,
    )


def _env_flag_enabled(name: str) -> bool:
    """Return whether the named environment flag is enabled.

    Args:
        name: Environment variable to interpret as a boolean toggle.

    Returns:
        ``True`` when the variable is set to ``1``, else ``False``.
    """
    return os.getenv(name, "0") == "1"


def _real_media_path() -> Path:
    """Return the configured real media path used by smoke tests.

    Returns:
        Path to the opt-in real media test file.
    """
    return Path(
        os.getenv(
            "TEST_REAL_MEDIA_PATH",
            "data/test_media/Silent Witness S14E01.mkv",
        )
    )


def _load_torch() -> types.ModuleType:
    """Import and return torch when available.

    Returns:
        The imported torch module.
    """
    import torch

    return torch


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Apply global skip rules for optional GPU and real-media tests.

    Args:
        item: The pytest test item about to run.
    """
    if "gpu" in item.keywords:
        if not _env_flag_enabled("RUN_GPU_TESTS"):
            pytest.skip("GPU tests are disabled. Set RUN_GPU_TESTS=1 to enable.")
        try:
            torch = _load_torch()
        except (ImportError, OSError):
            pytest.skip("Torch could not be imported in this test environment.")

        if not torch.cuda.is_available():
            pytest.skip("ROCm GPU runtime not available in this test environment.")

    if "real_media" in item.keywords:
        if not _env_flag_enabled("RUN_REAL_MEDIA_TESTS"):
            pytest.skip(
                "Real media tests are disabled. Set RUN_REAL_MEDIA_TESTS=1 to enable."
            )
        media_path = _real_media_path()
        if not media_path.exists():
            pytest.skip(f"Real media file not found: {media_path}")


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    """Create and return a directory for test data files.

    Returns:
        str: Absolute path to the tests/data directory, created if missing.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def temp_upload_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create and return a temporary directory for file uploads.

    Args:
        tmp_path_factory: Pytest factory to create temporary paths.

    Returns:
        Path: Newly created temporary directory path for uploads.
    """
    return tmp_path_factory.mktemp("uploads")


@pytest.fixture(scope="session")
def sample_video_path() -> Path:
    """Return the committed sample video fixture used by smoke tests.

    Returns:
        Path to ``tests/data/sample.mp4``.

    """
    video_path = Path(__file__).resolve().parent / "data" / "sample.mp4"
    if not video_path.exists():
        pytest.skip(f"Sample video fixture missing: {video_path}")
    return video_path


@pytest.fixture(scope="session")
def real_media_path() -> Path:
    """Return the configured opt-in real media file path.

    Returns:
        Path to the real media file selected by ``TEST_REAL_MEDIA_PATH``.
    """
    return _real_media_path()


@pytest.fixture(scope="session")
def webui_server(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """Spin up the WebUI once per test session and tear it down afterwards.

    Args:
        request: Pytest fixture request object.

    Yields:
        str: Base URL that can be passed to ``gradio_client.Client``.

    Raises:
        RuntimeError: If the WebUI fails to start within the timeout.

    Notes:
        Uses the lightweight ``openai/whisper-tiny`` model for faster startup.
    """
    if os.getenv("RUN_WEBUI_TESTS", "0") != "1":
        pytest.skip("WebUI server tests are disabled in this environment.")

    base_url = "http://localhost:7861"

    # Launch the WebUI as a subprocess
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "insanely_fast_whisper_rocm.webui",
            "--model",
            "openai/whisper-tiny",
            "--port",
            "7861",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait until the server responds or timeout after 60s
    timeout = 60
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(base_url, timeout=3).status_code == 200:
                break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(1)
    else:
        # Server failed to start; dump logs for debugging and abort tests
        output, _ = process.communicate(timeout=5)
        raise RuntimeError(f"WebUI failed to start within {timeout}s. Logs:\n{output}")

    yield base_url

    # Teardown: terminate the process gracefully
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
