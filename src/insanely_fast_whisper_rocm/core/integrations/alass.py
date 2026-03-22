"""ALASS subtitle synchronization integration."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from insanely_fast_whisper_rocm.utils import constant as constants

logger = logging.getLogger(__name__)


def sync_subtitle_with_alass(
    *,
    reference_media_path: str | Path,
    subtitle_content: str,
    binary: str = constants.ALASS_BINARY,
    timeout_seconds: int = constants.ALASS_TIMEOUT_SECONDS,
    split_penalty: int = constants.ALASS_SPLIT_PENALTY,
    no_splits: bool = constants.ALASS_NO_SPLITS,
) -> tuple[str, dict[str, Any]]:
    """Synchronize SRT content to media using the external ``alass`` binary.

    Args:
        reference_media_path: Path to the media file used as synchronization
            reference.
        subtitle_content: Input SRT text to synchronize.
        binary: ALASS executable name or absolute path.
        timeout_seconds: Hard timeout for the synchronization command.
        split_penalty: Split penalty passed to ``--split-penalty``.
        no_splits: Whether to pass ``--no-splits``.

    Returns:
        A tuple ``(srt_content, metadata)``. On failure, ``srt_content`` is the
        original input and metadata includes the failure reason.
    """
    started = time.perf_counter()
    metadata: dict[str, Any] = {
        "enabled": True,
        "engine": "alass",
        "applied": False,
        "reason": None,
        "runtime_ms": None,
    }
    reference_path = Path(reference_media_path).expanduser().resolve()
    if not subtitle_content.strip():
        metadata["reason"] = "empty_subtitle_content"
        metadata["runtime_ms"] = int((time.perf_counter() - started) * 1000)
        return subtitle_content, metadata
    if not reference_path.exists():
        metadata["reason"] = "reference_media_missing"
        metadata["runtime_ms"] = int((time.perf_counter() - started) * 1000)
        return subtitle_content, metadata

    binary_path = shutil.which(binary)
    if binary_path is None:
        metadata["reason"] = f"alass_binary_not_found:{binary}"
        metadata["runtime_ms"] = int((time.perf_counter() - started) * 1000)
        return subtitle_content, metadata

    with tempfile.TemporaryDirectory(prefix="ifw-alass-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = temp_dir_path / "input.srt"
        output_path = temp_dir_path / "output.srt"
        input_path.write_text(subtitle_content, encoding="utf-8")

        command = [
            binary_path,
            str(reference_path),
            str(input_path),
            str(output_path),
        ]
        if no_splits:
            command.append("--no-splits")
        else:
            command.extend(["--split-penalty", str(split_penalty)])

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            metadata["reason"] = "alass_timeout"
            metadata["runtime_ms"] = int((time.perf_counter() - started) * 1000)
            return subtitle_content, metadata
        except OSError as exc:
            metadata["reason"] = f"alass_exec_error:{exc}"
            metadata["runtime_ms"] = int((time.perf_counter() - started) * 1000)
            return subtitle_content, metadata

        if completed.returncode != 0:
            stderr_text = (completed.stderr or completed.stdout or "").strip()
            reason = stderr_text.splitlines()[0] if stderr_text else "non_zero_exit"
            metadata["reason"] = f"alass_failed:{reason}"
            metadata["runtime_ms"] = int((time.perf_counter() - started) * 1000)
            return subtitle_content, metadata

        if not output_path.exists():
            metadata["reason"] = "alass_no_output"
            metadata["runtime_ms"] = int((time.perf_counter() - started) * 1000)
            return subtitle_content, metadata

        synced_content = output_path.read_text(encoding="utf-8")
        metadata["applied"] = True
        metadata["runtime_ms"] = int((time.perf_counter() - started) * 1000)
        logger.debug(
            "ALASS synchronization completed for %s in %dms",
            reference_path,
            metadata["runtime_ms"],
        )
        return synced_content, metadata
