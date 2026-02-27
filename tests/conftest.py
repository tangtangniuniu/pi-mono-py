"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def tmp_config_dir(tmp_path: object) -> object:
    """Provide a temporary config directory for tests."""
    return tmp_path
