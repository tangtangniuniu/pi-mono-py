"""Tests for pi_mono.coding_agent.core.settings_manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from pi_mono.ai.api_registry import clear_api_providers, get_api_provider
from pi_mono.coding_agent.core.settings_manager import Settings, SettingsManager

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def settings_mgr(tmp_path: Path) -> SettingsManager:
    return SettingsManager(settings_file=tmp_path / "settings.yaml")


class TestSettings:
    def test_default_values(self) -> None:
        s = Settings()
        assert s.default_model == "claude-sonnet-4-20250514"
        assert s.thinking_level == "off"
        assert s.max_turns == 100
        assert s.auto_compact is True
        assert s.theme == "default"
        assert s.verbose is False
        assert s.custom_api_urls == {}
        assert s.custom_headers == {}

    def test_custom_values(self) -> None:
        s = Settings(default_model="gpt-4", thinking_level="high", verbose=True)
        assert s.default_model == "gpt-4"
        assert s.thinking_level == "high"
        assert s.verbose is True

    def test_model_copy_immutability(self) -> None:
        s1 = Settings()
        s2 = s1.model_copy(update={"verbose": True})
        assert s1.verbose is False
        assert s2.verbose is True

    def test_invalid_field_type_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            Settings(max_turns="not_an_int")  # type: ignore[arg-type]


class TestLoadSettings:
    @pytest.fixture(autouse=True)
    def _clear_provider_registry(self) -> None:
        clear_api_providers()
        yield  # type: ignore[misc]
        clear_api_providers()

    async def test_load_returns_default_when_no_file(self, settings_mgr: SettingsManager) -> None:
        s = await settings_mgr.load()
        assert isinstance(s, Settings)
        assert s.default_model == "claude-sonnet-4-20250514"
        assert get_api_provider("openai-completions") is not None

    async def test_load_reads_existing_yaml_file(self, tmp_path: Path) -> None:
        f = tmp_path / "settings.yaml"
        f.write_text(yaml.dump({"verbose": True, "theme": "dark"}))
        mgr = SettingsManager(settings_file=f)
        s = await mgr.load()
        assert s.verbose is True
        assert s.theme == "dark"

    async def test_load_caches_result(self, settings_mgr: SettingsManager) -> None:
        s1 = await settings_mgr.load()
        s2 = await settings_mgr.load()
        assert s1 is s2  # same object

    async def test_load_returns_default_on_corrupt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "settings.yaml"
        f.write_text("invalid: yaml: [broken: {")
        mgr = SettingsManager(settings_file=f)
        s = await mgr.load()
        assert isinstance(s, Settings)
        assert s.verbose is False  # default

    async def test_load_sets_runtime_model_id_from_dotenv(self, tmp_path: Path) -> None:
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
        mgr = SettingsManager(settings_file=tmp_path / "settings.yaml", project_dir=tmp_path)
        s = await mgr.load()
        assert s.runtime_model_id == "gpt-5.2"

    async def test_bootstrap_runtime_model_registers_model(self, tmp_path: Path) -> None:
        from pi_mono.coding_agent.core.model_registry import ModelRegistry

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
        mgr = SettingsManager(settings_file=tmp_path / "settings.yaml", project_dir=tmp_path)
        registry = ModelRegistry()

        model = mgr.bootstrap_runtime_model(registry)
        assert model is not None
        assert registry.get_model("gpt-5.2") is not None
        assert get_api_provider("openai-completions") is not None


class TestSaveSettings:
    async def test_save_creates_yaml_file(self, tmp_path: Path) -> None:
        f = tmp_path / "subdir" / "settings.yaml"
        mgr = SettingsManager(settings_file=f)
        s = Settings(verbose=True)
        await mgr.save(s)
        assert f.exists()
        data = yaml.safe_load(f.read_text())
        assert data["verbose"] is True

    async def test_save_updates_cache(self, settings_mgr: SettingsManager) -> None:
        s1 = Settings(theme="dark")
        await settings_mgr.save(s1)
        s2 = await settings_mgr.load()
        assert s2.theme == "dark"
        assert s2 is s1  # cached


class TestUpdateSettings:
    async def test_update_changes_field(self, settings_mgr: SettingsManager) -> None:
        updated = await settings_mgr.update(verbose=True, theme="monokai")
        assert updated.verbose is True
        assert updated.theme == "monokai"

    async def test_update_preserves_other_fields(self, settings_mgr: SettingsManager) -> None:
        await settings_mgr.update(verbose=True)
        updated = await settings_mgr.update(theme="dark")
        assert updated.verbose is True
        assert updated.theme == "dark"


class TestResetSettings:
    async def test_reset_returns_defaults(self, settings_mgr: SettingsManager) -> None:
        await settings_mgr.update(verbose=True, theme="dark")
        reset = await settings_mgr.reset()
        assert reset.verbose is False
        assert reset.theme == "default"

    async def test_reset_persists_to_file(self, tmp_path: Path) -> None:
        f = tmp_path / "settings.yaml"
        mgr = SettingsManager(settings_file=f)
        await mgr.update(verbose=True)
        await mgr.reset()
        # Re-create manager to verify persistence
        mgr2 = SettingsManager(settings_file=f)
        s = await mgr2.load()
        assert s.verbose is False
