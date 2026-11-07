import importlib
import os

import pytest

from sentinel_rosetta.infrastructure import _offline_env

_VARIABLE_NAMES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _forget_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv first so monkeypatch restores the original state at teardown,
    # even when the variable was absent before the test.
    for name in _VARIABLE_NAMES:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_offline_env_import_sets_both_variables_to_one_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _forget_variables(monkeypatch)

    # Act
    importlib.reload(_offline_env)

    # Assert
    assert [os.environ[name] for name in _VARIABLE_NAMES] == ["1", "1"]


def test_offline_env_import_keeps_existing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _forget_variables(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    # Act
    importlib.reload(_offline_env)

    # Assert
    assert os.environ["HF_HUB_OFFLINE"] == "0"
