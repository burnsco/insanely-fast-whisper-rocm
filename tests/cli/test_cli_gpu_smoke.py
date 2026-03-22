"""GPU smoke tests for CLI transcription using a small video fixture."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _run_cli_transcription(
    *,
    input_path: Path,
    output_path: Path,
    device: str,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run the CLI transcription command for a single input file.

    Args:
        input_path: Video file to transcribe.
        output_path: Destination text file for transcription output.
        device: Device string to pass to the CLI.
        timeout_seconds: Maximum runtime for the subprocess.

    Returns:
        Completed subprocess result with captured output.
    """
    command = [
        sys.executable,
        "-m",
        "insanely_fast_whisper_rocm.cli",
        "transcribe",
        str(input_path),
        "--model",
        "openai/whisper-tiny",
        "--device",
        device,
        "--dtype",
        "float32" if device == "cpu" else "float16",
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

    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=timeout_seconds,
    )


def _assert_transcription_success(
    *,
    completed: subprocess.CompletedProcess[str],
    output_path: Path,
    log_path: Path,
) -> None:
    """Assert that the CLI run succeeded and emitted transcript text.

    Args:
        completed: Completed subprocess result.
        output_path: Transcription output file expected on success.
        log_path: File where process logs were written for inspection.
    """
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
        f"CLI transcription failed with return code {completed.returncode}. "
        f"See {log_path} for captured output."
    )
    assert output_path.exists(), f"Expected transcript output at {output_path}"
    assert output_path.read_text(encoding="utf-8").strip(), (
        f"Transcript output was empty. See {log_path} for captured output."
    )


def test_cli_transcribe_sample_video_cpu_smoke(
    sample_video_path: Path,
    tmp_path: Path,
) -> None:
    """Transcribe the sample video on CPU as a baseline smoke test."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary is not installed in this environment.")

    output_path = tmp_path / "sample-cpu.txt"
    log_path = tmp_path / "sample-cpu.log"
    completed = _run_cli_transcription(
        input_path=sample_video_path,
        output_path=output_path,
        device="cpu",
        timeout_seconds=900,
    )

    _assert_transcription_success(
        completed=completed,
        output_path=output_path,
        log_path=log_path,
    )


def test_cli_transcribe_sample_video_gpu_smoke(
    sample_video_path: Path,
    tmp_path: Path,
) -> None:
    """Transcribe the sample video on GPU without crashing."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary is not installed in this environment.")

    output_path = tmp_path / "sample-gpu.txt"
    log_path = tmp_path / "sample-gpu.log"
    completed = _run_cli_transcription(
        input_path=sample_video_path,
        output_path=output_path,
        device="0",
        timeout_seconds=900,
    )

    _assert_transcription_success(
        completed=completed,
        output_path=output_path,
        log_path=log_path,
    )
