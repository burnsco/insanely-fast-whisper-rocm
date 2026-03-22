"""Tests for the ALASS integration wrapper."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import NoReturn

import pytest

from insanely_fast_whisper_rocm.core.integrations.alass import sync_subtitle_with_alass


def test_sync_subtitle_with_alass__binary_missing(tmp_path: Path) -> None:
    """Return original subtitle text when ALASS binary is unavailable."""
    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"fake")

    output_text, metadata = sync_subtitle_with_alass(
        reference_media_path=media_path,
        subtitle_content="1\n00:00:00,000 --> 00:00:01,000\nHi\n",
        binary="missing-alass-binary",
    )

    assert "Hi" in output_text
    assert metadata["applied"] is False
    assert "alass_binary_not_found" in str(metadata["reason"])


def test_sync_subtitle_with_alass__timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return original subtitle text when ALASS command times out."""
    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"fake")

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.integrations.alass.shutil.which",
        lambda _: "/usr/bin/alass",
    )

    def _raise_timeout(*args: object, **kwargs: object) -> NoReturn:
        raise subprocess.TimeoutExpired(cmd=["alass"], timeout=1)

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.integrations.alass.subprocess.run",
        _raise_timeout,
    )

    output_text, metadata = sync_subtitle_with_alass(
        reference_media_path=media_path,
        subtitle_content="1\n00:00:00,000 --> 00:00:01,000\nHi\n",
    )
    assert "Hi" in output_text
    assert metadata["applied"] is False
    assert metadata["reason"] == "alass_timeout"


def test_sync_subtitle_with_alass__success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return synchronized subtitle text when ALASS succeeds."""
    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"fake")
    expected = "1\n00:00:01,000 --> 00:00:02,000\nHello\n"

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.integrations.alass.shutil.which",
        lambda _: "/usr/bin/alass",
    )

    def _fake_run(
        command: list[str],
        *,
        capture_output: bool,
        check: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[3])
        output_path.write_text(expected, encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(
        "insanely_fast_whisper_rocm.core.integrations.alass.subprocess.run",
        _fake_run,
    )

    output_text, metadata = sync_subtitle_with_alass(
        reference_media_path=media_path,
        subtitle_content="1\n00:00:00,000 --> 00:00:01,000\nHello\n",
    )
    assert output_text == expected
    assert metadata["applied"] is True
    assert metadata["engine"] == "alass"
