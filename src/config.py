"""
Module for handling configurations.

This module provides functions to load configuration files (from .py or .yaml),
update configuration dictionaries with command line arguments, and return the final
configurations as dictionaries.
"""

import argparse
import importlib.util
import os
from typing import Dict, Any

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a .py or .yaml configuration file and return it as a dictionary.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    ext = os.path.splitext(config_path)[1]
    if ext == ".py":
        return _load_py_config(config_path)
    if ext == ".yaml":
        return _load_yaml_config(config_path)
    raise ValueError(f"Unsupported config file format: {ext}")


def _load_py_config(config_path: str) -> Dict[str, Any]:
    """Load a Python configuration file and return it as a dictionary.

    Args:
        config_path (str): The path to the .py config file.

    Returns:
        dict: A dictionary of the configuration values.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("__")}


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path (str): The path to the .yaml config file.

    Returns:
        dict: A dictionary of the configuration values.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update a configuration dictionary with values from command line arguments.

    Args:
        config (dict): The configuration dictionary to update.
        args (argparse.Namespace): Command-line arguments that override config values.

    Returns:
        dict: The updated configuration dictionary.
    """
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config
