"""Tests for ``utils.config``."""
import pytest

from utils.config import load_config


def test_load_default():
    cfg = load_config()
    assert cfg.data.symbol
    assert cfg.lstm.sequence_length > 0
    assert cfg.ppo.total_timesteps > 0


def test_attribute_access_nested():
    cfg = load_config()
    # Net arch should be a list
    assert isinstance(cfg.ppo.policy_kwargs.net_arch, list)


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_raw_preserved():
    cfg = load_config()
    assert "data" in cfg._raw
    assert cfg._raw["data"]["symbol"] == cfg.data.symbol
