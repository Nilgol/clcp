# test_config.py
import pytest
import os
import argparse
from config import Config

# Sample config files for testing
PY_CONFIG_PATH = "test/config_sample.py"
YAML_CONFIG_PATH = "test/config_sample.yaml"
UNSUPPORTED_CONFIG_PATH = "test/config_sample.json"


def test_load_py_config():
    cfg = Config(PY_CONFIG_PATH)
    assert cfg.learning_rate == 0.001, "Failed to load Python config correctly"
    assert cfg.batch_size == 32, "Failed to load Python config correctly"


def test_load_yaml_config():
    cfg = Config(YAML_CONFIG_PATH)
    assert cfg.learning_rate == 0.001, "Failed to load YAML config correctly"
    assert cfg.batch_size == 32, "Failed to load YAML config correctly"


def test_unsupported_config_format():
    with pytest.raises(ValueError):
        cfg = Config(UNSUPPORTED_CONFIG_PATH)


def test_attribute_access():
    cfg = Config(PY_CONFIG_PATH)
    assert hasattr(cfg, "learning_rate"), "Attribute 'learning_rate' should exist"
    assert hasattr(cfg, "batch_size"), "Attribute 'batch_size' should exist"


def test_update_from_args():
    cfg = Config(PY_CONFIG_PATH)
    args = argparse.Namespace(learning_rate=0.01, batch_size=None)
    cfg.update_from_args(args)
    assert cfg.learning_rate == 0.01, "Command-line argument override failed"
    assert cfg.batch_size == 32, "Default config should remain if not overridden"


def test_get_with_default():
    cfg = Config(PY_CONFIG_PATH)
    assert cfg.get("non_existing_key", 42) == 42, "Default value fallback failed"
