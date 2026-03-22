"""ROCm preflight checks for local troubleshooting."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from insanely_fast_whisper_rocm.rocm_bootstrap import LINKED_LOCAL_ROCM_LIBRARIES
from insanely_fast_whisper_rocm.utils.rocm_report import generate_report


def _command_output(command: list[str]) -> dict[str, Any]:
    """Return the outcome of running a diagnostic command.

    Args:
        command: Command to execute.

    Returns:
        A small structured result describing the command outcome.
    """
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": command,
            "available": False,
            "returncode": None,
            "output": None,
        }

    completed = subprocess.run(
        [executable, *command[1:]],
        capture_output=True,
        check=False,
        text=True,
    )
    output = completed.stdout.strip() or completed.stderr.strip() or None
    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "output": output,
    }


def run_preflight() -> tuple[dict[str, Any], bool]:
    """Run ROCm preflight diagnostics for the current environment.

    Returns:
        A tuple of the diagnostic payload and whether required checks passed.
    """
    report = generate_report()
    commands = {
        "rocminfo": _command_output(["rocminfo"]),
        "rocm_smi": _command_output(["rocm-smi"]),
        "rocm_agent_enumerator": _command_output(["rocm_agent_enumerator", "-name"]),
    }
    passed = bool(
        report.get("torch_import_ok")
        and report.get("hip_version")
        and report.get("gpu_available")
        and (report.get("gpu_count", 0) >= 1)
        and report.get("gpu_probe_ok")
    )
    payload = {
        "passed": passed,
        "linked_local_rocm_libraries": LINKED_LOCAL_ROCM_LIBRARIES,
        "report": report,
        "commands": commands,
    }
    return payload, passed


def main() -> None:
    """Print ROCm preflight diagnostics and exit non-zero on failure.

    Raises:
        SystemExit: If the required ROCm checks fail.
    """
    payload, passed = run_preflight()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
