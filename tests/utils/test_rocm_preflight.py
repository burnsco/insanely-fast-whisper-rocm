"""Tests for ROCm preflight diagnostics."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from insanely_fast_whisper_rocm.utils import rocm_preflight


def test_run_preflight_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return a passing payload when required GPU checks succeed."""
    monkeypatch.setattr(
        rocm_preflight,
        "generate_report",
        lambda: {
            "torch_import_ok": True,
            "hip_version": "7.2.0",
            "gpu_available": True,
            "gpu_count": 1,
            "gpu_probe_ok": True,
        },
    )
    monkeypatch.setattr(
        rocm_preflight,
        "_command_output",
        lambda command: {"command": command, "available": True, "returncode": 0},
    )

    payload, passed = rocm_preflight.run_preflight()

    assert passed is True
    assert payload["passed"] is True
    assert isinstance(payload["linked_local_rocm_libraries"], list)


def test_run_preflight_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return a failing payload when required GPU checks are missing."""
    monkeypatch.setattr(
        rocm_preflight,
        "generate_report",
        lambda: {
            "torch_import_ok": True,
            "hip_version": None,
            "gpu_available": False,
            "gpu_count": 0,
            "gpu_probe_ok": False,
        },
    )
    monkeypatch.setattr(
        rocm_preflight,
        "_command_output",
        lambda command: {"command": command, "available": False, "returncode": None},
    )

    payload, passed = rocm_preflight.run_preflight()

    assert passed is False
    assert payload["passed"] is False


def test_rocm_bootstrap_links_helper_libraries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Link helper libraries into the torch library directory."""
    module = importlib.import_module("insanely_fast_whisper_rocm.rocm_bootstrap")
    torch_lib_dir = tmp_path / "torch-lib"
    helper_dir = tmp_path / "helper"
    source = helper_dir / "libhipsparselt.so.0"
    helper_dir.mkdir(parents=True)
    torch_lib_dir.mkdir(parents=True)
    source.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(module, "_torch_library_dir", lambda: torch_lib_dir)
    monkeypatch.setattr(module, "_candidate_helper_library_dirs", lambda: [helper_dir])

    created = module.link_local_rocm_shared_libraries()

    destination = torch_lib_dir / source.name
    assert created == [str(destination)]
    assert destination.is_symlink()
    assert destination.resolve() == source.resolve()


def test_rocm_bootstrap_removes_stale_helper_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remove stale helper-library symlinks before linking new libraries."""
    module = importlib.import_module("insanely_fast_whisper_rocm.rocm_bootstrap")
    torch_lib_dir = tmp_path / "torch-lib"
    helper_root = tmp_path / ".local-rocm-libs"
    stale_target = (
        helper_root / "hipsparselt" / "opt" / "rocm" / "lib" / "libmissing.so"
    )
    stale_link = torch_lib_dir / "libmissing.so"
    torch_lib_dir.mkdir(parents=True)
    stale_link.symlink_to(stale_target)

    monkeypatch.setattr(module, "_torch_library_dir", lambda: torch_lib_dir)
    monkeypatch.setattr(module, "_helper_root", lambda: helper_root)
    monkeypatch.setattr(module, "_candidate_helper_library_dirs", lambda: [])

    created = module.link_local_rocm_shared_libraries()

    assert created == []
    assert stale_link.is_symlink() is False
