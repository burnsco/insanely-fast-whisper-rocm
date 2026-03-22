"""Opt-in real-media smoke test for CLI transcription."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.real_media]


def _load_torch() -> types.ModuleType:
    """Import and return torch for GPU availability checks.

    Returns:
        Imported torch module.
    """
    import torch

    return torch


def test_cli_transcribe_real_media_gpu_smoke(
    real_media_path: Path,
    tmp_path: Path,
) -> None:
    """Transcribe the configured real media file on GPU without segfaulting."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary is not installed in this environment.")

    try:
        torch = _load_torch()
    except (ImportError, OSError):
        pytest.skip("Torch could not be imported in this test environment.")

    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU runtime not available in this test environment.")

    output_path = tmp_path / "real-media.txt"
    log_path = tmp_path / "real-media.log"
    command = [
        sys.executable,
        "-m",
        "insanely_fast_whisper_rocm.cli",
        "transcribe",
        str(real_media_path),
        "--model",
        "openai/whisper-tiny",
        "--device",
        "0",
        "--dtype",
        "float16",
        "--batch-size",
        "1",
        "--progress-group-size",
        "1",
        "--chunk-length",
        "15",
        "--language",
        "en",
        "--no-progress",
        "--no-stabilize",
        "--export-format",
        "txt",
        "--output",
        str(output_path),
    ]

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=5400,
    )
    log_path.write_text(
        "\n".join([
            f"returncode: {completed.returncode}",
            "stdout:",
            completed.stdout,
            "stderr:",
            completed.stderr,
        ]),
        encoding="utf-8",
    )

    assert completed.returncode == 0, (
        f"Real-media CLI transcription failed with return code "
        f"{completed.returncode}. See {log_path} for captured output."
    )
    assert output_path.exists(), f"Expected transcript output at {output_path}"
    assert output_path.read_text(encoding="utf-8").strip(), (
        f"Transcript output was empty. See {log_path} for captured output."
    )
