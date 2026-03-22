"""Repository-level contract tests for the cleaned ROCm 7.2 app surface."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_uses_single_rocm_extra() -> None:
    """Keep one ROCm dependency surface in ``pyproject.toml``."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "rocm = [" in pyproject
    assert "rocm-6-4-1" not in pyproject
    assert "rocm-7-0" not in pyproject
    assert "rocm-7-2 = [" not in pyproject


def test_generated_requirements_exports_are_removed() -> None:
    """Avoid maintaining duplicate requirements exports beside uv metadata."""
    removed_files = [
        "constraints-no-heavy.txt",
        "requirements-all.txt",
        "requirements-dev.txt",
        "requirements-reviewer.txt",
        "requirements-rocm-v6-4-1.txt",
        "requirements-rocm-v7-0.txt",
        "requirements-rocm-v7-2.txt",
        "requirements.txt",
        "project-overview.md",
    ]

    for relative_path in removed_files:
        assert not (PROJECT_ROOT / relative_path).exists(), relative_path


def test_compose_is_api_first_with_optional_webui() -> None:
    """Keep both compose files aligned to the API-first deployment model."""
    compose_paths = [
        PROJECT_ROOT / "docker-compose.yaml",
        PROJECT_ROOT / "docker-compose.dev.yaml",
    ]

    for compose_path in compose_paths:
        payload = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        services = payload["services"]

        assert "api" in services
        assert "webui" in services
        assert services["webui"]["profiles"] == ["webui"]


def test_environment_reads_are_centralized() -> None:
    """Disallow new direct environment reads outside config/bootstrap modules."""
    allowed_files = {
        PROJECT_ROOT / "insanely_fast_whisper_rocm" / "utils" / "constant.py",
        PROJECT_ROOT / "insanely_fast_whisper_rocm" / "utils" / "constants.py",
        PROJECT_ROOT / "insanely_fast_whisper_rocm" / "utils" / "env_loader.py",
        PROJECT_ROOT / "insanely_fast_whisper_rocm" / "cli" / "cli.py",
    }

    for path in (PROJECT_ROOT / "insanely_fast_whisper_rocm").rglob("*.py"):
        if path in allowed_files:
            continue

        content = path.read_text(encoding="utf-8")
        assert "os.getenv(" not in content, str(path)
        assert "os.environ[" not in content, str(path)
