"""Tests for ROCm runtime diagnostics helpers."""

from __future__ import annotations

import pytest

from insanely_fast_whisper_rocm.utils import rocm_report


def test_generate_report_includes_runtime_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate a report with stable test doubles for external probes."""
    monkeypatch.setattr(rocm_report.constant, "ROCM_PATH", "/opt/rocm")
    monkeypatch.setattr(
        rocm_report.constant,
        "PYTORCH_ALLOC_CONF",
        "max_split_size_mb:128",
    )

    monkeypatch.setattr(
        rocm_report,
        "_read_os_release",
        lambda: "Test Linux",
    )
    monkeypatch.setattr(
        rocm_report,
        "_command_output",
        lambda command: "ffmpeg version test" if command[0] == "ffmpeg" else None,
    )
    monkeypatch.setattr(
        rocm_report.shutil,
        "which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    monkeypatch.setattr(
        rocm_report,
        "_torch_report",
        lambda: {
            "torch_import_ok": True,
            "torch_version": "2.9.1+rocm7.2.0",
            "hip_version": "7.2.0",
            "gpu_available": True,
            "gpu_count": 1,
            "gpu_name": "AMD Radeon RX 9070",
            "gpu_probe_ok": True,
            "gpu_probe_error": None,
            "torch_import_error": None,
        },
    )

    report = rocm_report.generate_report()

    assert report["os"] == "Test Linux"
    assert report["rocm_path"] == "/opt/rocm"
    assert report["pytorch_alloc_conf"] == "max_split_size_mb:128"
    assert report["ffmpeg_path"] == "/usr/bin/ffmpeg"
    assert report["ffmpeg_version"] == "ffmpeg version test"
    assert report["torch_version"] == "2.9.1+rocm7.2.0"
    assert report["gpu_available"] is True
