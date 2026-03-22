"""ROCm runtime diagnostics helpers.

This module provides a lightweight report for the current Python and ROCm
runtime. The report is intentionally safe to run in local development
environments and inside Docker containers.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from insanely_fast_whisper_rocm.utils import constant


def _read_os_release() -> str:
    """Return a compact operating system description.

    Returns:
        A human-readable operating system string.
    """
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return platform.platform()

    fields: dict[str, str] = {}
    for line in os_release.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key] = value.strip().strip('"')

    return fields.get("PRETTY_NAME", platform.platform())


def _command_output(command: list[str]) -> str | None:
    """Return trimmed output for ``command`` when available.

    Args:
        command: Command and arguments to execute.

    Returns:
        Trimmed stdout on success, else ``None``.
    """
    executable = shutil.which(command[0])
    if executable is None:
        return None

    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return completed.stdout.strip() or completed.stderr.strip() or None


def _torch_report() -> dict[str, Any]:
    """Build a report for the current torch runtime.

    Returns:
        Mapping of torch and GPU diagnostic values.
    """
    report: dict[str, Any] = {
        "torch_import_ok": False,
        "torch_version": None,
        "hip_version": None,
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_probe_ok": False,
        "gpu_probe_error": None,
        "torch_import_error": None,
    }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised on broken runtimes
        report["torch_import_error"] = repr(exc)
        return report

    report["torch_import_ok"] = True
    report["torch_version"] = torch.__version__
    report["hip_version"] = getattr(torch.version, "hip", None)
    report["gpu_available"] = torch.cuda.is_available()
    report["gpu_count"] = torch.cuda.device_count()

    if report["gpu_available"] and report["gpu_count"] > 0:
        try:
            report["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover - defensive
            report["gpu_name"] = f"<error: {exc}>"

        try:
            tensor = torch.ones((4, 4), device="cuda:0", dtype=torch.float16)
            result = (tensor @ tensor).sum().item()
            report["gpu_probe_ok"] = True
            report["gpu_probe_result"] = result
        except Exception as exc:  # pragma: no cover - depends on local runtime
            report["gpu_probe_error"] = repr(exc)

    return report


def generate_report() -> dict[str, Any]:
    """Generate a ROCm runtime report for the current process.

    Returns:
        Mapping containing runtime diagnostics and app defaults.
    """
    ffmpeg_version = _command_output(["ffmpeg", "-version"])
    report: dict[str, Any] = {
        "os": _read_os_release(),
        "kernel": platform.release(),
        "python": sys.version.splitlines()[0],
        "rocm_path": constant.ROCM_PATH,
        "hsa_override_gfx_version": constant.HSA_OVERRIDE_GFX_VERSION,
        "pytorch_alloc_conf": constant.PYTORCH_ALLOC_CONF,
        "pytorch_hip_alloc_conf": constant.PYTORCH_HIP_ALLOC_CONF,
        "torchaudio_use_soundfile": constant.TORCHAUDIO_USE_SOUNDFILE,
        "ffmpeg_path": shutil.which("ffmpeg"),
        "ffmpeg_version": ffmpeg_version.splitlines()[0] if ffmpeg_version else None,
        "default_model": constant.DEFAULT_MODEL,
        "default_device": constant.DEFAULT_DEVICE,
    }
    report.update(_torch_report())
    return report


def main() -> None:
    """Print the ROCm runtime report in a readable format."""
    report = generate_report()
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
