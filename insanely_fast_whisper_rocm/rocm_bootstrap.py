"""Bootstrap local ROCm helper libraries for the active Python environment."""

from __future__ import annotations

import sysconfig
from pathlib import Path


def _helper_root() -> Path:
    """Return the repository-local helper library root.

    Returns:
        Path to the optional helper library directory.
    """
    return Path(__file__).resolve().parent.parent / ".local-rocm-libs"


def _candidate_helper_library_dirs() -> list[Path]:
    """Return helper ROCm library directories bundled with the repo.

    Returns:
        Candidate directories that contain extra ROCm shared libraries.
    """
    helper_root = _helper_root()
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


def _remove_stale_helper_symlinks(torch_lib_dir: Path) -> list[str]:
    """Remove broken helper-library symlinks from ``torch/lib``.

    Args:
        torch_lib_dir: The torch shared-library directory for the environment.

    Returns:
        Symlink paths removed during the cleanup pass.
    """
    helper_root = _helper_root()
    removed: list[str] = []
    for candidate in sorted(torch_lib_dir.glob("lib*.so*")):
        if not candidate.is_symlink():
            continue
        target = candidate.readlink()
        if not target.is_absolute():
            target = (candidate.parent / target).resolve(strict=False)
        try:
            target.relative_to(helper_root)
        except ValueError:
            continue
        if target.exists():
            continue
        candidate.unlink()
        removed.append(str(candidate))
    return removed


def link_local_rocm_shared_libraries() -> list[str]:
    """Link bundled helper libraries into ``torch/lib`` when missing.

    Returns:
        The symlink paths created during the current bootstrap run.
    """
    torch_lib_dir = _torch_library_dir()
    if not torch_lib_dir.is_dir():
        return []

    _remove_stale_helper_symlinks(torch_lib_dir)
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
