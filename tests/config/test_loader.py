"""Tests for pi_mono.config.loader — YAML config loading with priority merging."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pi_mono.config.loader import load_settings
from pi_mono.config.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadFromProjectFile:
    async def test_load_project_settings(self, tmp_path: Path) -> None:
        project_file = tmp_path / ".pi" / "settings.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("models:\n  default: gpt-4o\nverbose: true\n")

        s = await load_settings(project_dir=tmp_path, user_dir=tmp_path / "nonexistent")
        assert s.models.default == "gpt-4o"
        assert s.verbose is True

    async def test_missing_project_file_uses_defaults(self, tmp_path: Path) -> None:
        s = await load_settings(project_dir=tmp_path, user_dir=tmp_path / "nonexistent")
        assert s.models.default == "claude-sonnet-4-20250514"


class TestLoadFromUserFile:
    async def test_user_level_fallback(self, tmp_path: Path) -> None:
        user_file = tmp_path / "user" / ".pi" / "settings.yaml"
        user_file.parent.mkdir(parents=True)
        user_file.write_text("models:\n  default: gpt-4o\n")

        s = await load_settings(project_dir=tmp_path / "project", user_dir=tmp_path / "user")
        assert s.models.default == "gpt-4o"

    async def test_project_overrides_user(self, tmp_path: Path) -> None:
        user_dir = tmp_path / "user"
        user_file = user_dir / ".pi" / "settings.yaml"
        user_file.parent.mkdir(parents=True)
        user_file.write_text("models:\n  default: user-model\nverbose: true\n")

        project_dir = tmp_path / "project"
        project_file = project_dir / ".pi" / "settings.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("models:\n  default: project-model\n")

        s = await load_settings(project_dir=project_dir, user_dir=user_dir)
        assert s.models.default == "project-model"
        # User-level verbose should be preserved since project doesn't override it
        assert s.verbose is True


class TestEnvironmentVariableOverride:
    async def test_env_overrides_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        project_file = tmp_path / ".pi" / "settings.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("models:\n  default: file-model\n")

        monkeypatch.setenv("PI_DEFAULT_MODEL", "env-model")

        s = await load_settings(project_dir=tmp_path, user_dir=tmp_path / "nonexistent")
        assert s.models.default == "env-model"

    async def test_env_verbose(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PI_VERBOSE", "true")
        s = await load_settings(project_dir=tmp_path, user_dir=tmp_path / "nonexistent")
        assert s.verbose is True

    async def test_runtime_model_env_override_wins_over_dotenv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "PI_RUNTIME_MODEL_PROVIDER=dotenv-provider",
                    "PI_RUNTIME_MODEL_BASE_URL=http://127.0.0.1:8317/v1",
                    "PI_RUNTIME_MODEL_ID=dotenv-model",
                    "PI_RUNTIME_MODEL_API_KEY_ENV=DOTENV_PROVIDER_API_KEY",
                    "DOTENV_PROVIDER_API_KEY=sk-dotenv",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("PI_RUNTIME_MODEL_ID", "env-model")

        s = await load_settings(project_dir=tmp_path, user_dir=tmp_path / "nonexistent")
        assert s.runtime_model_id == "env-model"
        assert any(model.id == "env-model" for model in s.models.custom)


class TestDotenvRuntimeModel:
    async def test_loads_runtime_model_from_dotenv(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "PI_RUNTIME_MODEL_PROVIDER=cliproxyapi",
                    "PI_RUNTIME_MODEL_BASE_URL=http://127.0.0.1:8317/v1",
                    "PI_RUNTIME_MODEL_ID=gpt-5.2",
                    "PI_RUNTIME_MODEL_API_KEY_ENV=CLIPROXYAPI_API_KEY",
                    "CLIPROXYAPI_API_KEY=sk-test",
                ]
            ),
            encoding="utf-8",
        )

        s = await load_settings(project_dir=tmp_path, user_dir=tmp_path / "nonexistent")
        assert s.runtime_model_id == "gpt-5.2"
        assert any(model.id == "gpt-5.2" and model.provider == "cliproxyapi" for model in s.models.custom)

    async def test_invalid_runtime_model_config_raises(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "PI_RUNTIME_MODEL_PROVIDER=cliproxyapi",
                    "PI_RUNTIME_MODEL_BASE_URL=http://127.0.0.1:8317/v1",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="PI_RUNTIME_MODEL_ID"):
            await load_settings(project_dir=tmp_path, user_dir=tmp_path / "nonexistent")


class TestMissingFiles:
    async def test_both_missing_returns_defaults(self, tmp_path: Path) -> None:
        s = await load_settings(
            project_dir=tmp_path / "no_project",
            user_dir=tmp_path / "no_user",
        )
        assert isinstance(s, Settings)
        assert s.models.default == "claude-sonnet-4-20250514"


class TestInvalidYaml:
    async def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        project_file = tmp_path / ".pi" / "settings.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("invalid: yaml: [broken: {")

        with pytest.raises(ValueError, match=r"settings\.yaml"):
            await load_settings(project_dir=tmp_path, user_dir=tmp_path / "nonexistent")

    async def test_invalid_field_type_raises(self, tmp_path: Path) -> None:
        project_file = tmp_path / ".pi" / "settings.yaml"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("models:\n  max_turns: fifty\n")

        with pytest.raises(ValueError):
            await load_settings(project_dir=tmp_path, user_dir=tmp_path / "nonexistent")
