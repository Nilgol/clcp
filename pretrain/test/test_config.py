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


def test_config_dict_and_attributes_consistency():
    """
    Ensure that class attributes and the `config` dictionary are consistent after loading the config.
    """
    cfg = Config(PY_CONFIG_PATH)

    # Check for consistency between class attributes and config dict
    for key in cfg.config:
        assert hasattr(
            cfg, key
        ), f"Class attribute '{key}' is missing but present in config dict"
        assert (
            getattr(cfg, key) == cfg.config[key]
        ), f"Mismatch between class attribute '{key}' and config dict value"


def test_config_dict_and_attributes_consistency_after_update():
    """
    Ensure consistency between class attributes and the `config` dictionary after updating with command-line args.
    """
    cfg = Config(PY_CONFIG_PATH)
    args = argparse.Namespace(learning_rate=0.02, batch_size=16)
    cfg.update_from_args(args)

    # Check for consistency between class attributes and config dict after update
    for key in cfg.config:
        assert hasattr(
            cfg, key
        ), f"Class attribute '{key}' is missing but present in config dict"
        assert (
            getattr(cfg, key) == cfg.config[key]
        ), f"Mismatch between class attribute '{key}' and config dict value after update"


def test_exp_name_generation():
    """
    Ensure that the experiment name is generated correctly if not provided
    in the config file or command-line arguments.
    """
    cfg = Config(PY_CONFIG_PATH)

    # Simulate no exp_name in config or args
    args = argparse.Namespace(exp_name=None)
    cfg.update_from_args(args)

    # Check that exp_name is automatically set
    assert hasattr(cfg, "exp_name"), "exp_name should be generated if missing"
    assert "exp_name" in cfg.config, "exp_name should be present in config dictionary"
    assert (
        cfg.exp_name == cfg.config["exp_name"]
    ), "exp_name should be consistent between attribute and config dict"
