"""Bootstrap local ROCm helper libraries for the active Python environment."""

from __future__ import annotations

import sysconfig
from pathlib import Path


def _candidate_helper_library_dirs() -> list[Path]:
    """Return helper ROCm library directories bundled with the repo.

    Returns:
        Candidate directories that contain extra ROCm shared libraries.
    """
    repo_root = Path(__file__).resolve().parent.parent
    helper_root = repo_root / ".local-rocm-libs"
    if not helper_root.exists():
        return []

    candidates: list[Path] = []
    for child in sorted(helper_root.iterdir()):
        lib_dir = child / "opt" / "rocm" / "lib"
        if lib_dir.is_dir():
            candidates.append(lib_dir)
    return candidates


def _torch_library_dir() -> Path:
    """Return the expected ``torch/lib`` directory for this interpreter.

    Returns:
        The torch shared-library directory in the active environment.
    """
    return Path(sysconfig.get_path("purelib")) / "torch" / "lib"


def link_local_rocm_shared_libraries() -> list[str]:
    """Link bundled helper libraries into ``torch/lib`` when missing.

    Returns:
        The symlink paths created during the current bootstrap run.
    """
    torch_lib_dir = _torch_library_dir()
    if not torch_lib_dir.is_dir():
        return []

    created: list[str] = []
    for helper_dir in _candidate_helper_library_dirs():
        for source in sorted(helper_dir.glob("lib*.so*")):
            destination = torch_lib_dir / source.name
            if destination.exists():
                continue
            destination.symlink_to(source)
            created.append(str(destination))
    return created


LINKED_LOCAL_ROCM_LIBRARIES = link_local_rocm_shared_libraries()
