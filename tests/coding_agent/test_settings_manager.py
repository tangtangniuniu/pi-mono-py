"""Tests for pi_mono.coding_agent.core.settings_manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

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
    async def test_load_returns_default_when_no_file(self, settings_mgr: SettingsManager) -> None:
        s = await settings_mgr.load()
        assert isinstance(s, Settings)
        assert s.default_model == "claude-sonnet-4-20250514"

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
