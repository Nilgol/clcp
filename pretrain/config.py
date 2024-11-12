"""Configuration management module for loading and handling experiment configurations.

This module provides the `Config` class, which supports loading configuration parameters 
from both Python (.py) and YAML (.yaml) files. It allows parameters to be accessed 
as class attributes and provides methods for overriding these parameters from command-line
arguments.
"""

import importlib.util
import os
from datetime import datetime
from typing import Any, Optional

import yaml


class Config:
    """A class to load, store, and manage configuration parameters.

    The `Config` class allows configuration parameters to be loaded from `.py` or `.yaml`
    files, and makes the parameters available as class attributes. It also supports updating
    configuration values based on command-line arguments.

    Attributes:
        config (dict): A dictionary holding the configuration parameters.
    """

    def __init__(self, config_path: str = None) -> None:
        """Initialize the Config object.

        Args:
            config_path (str, optional): The path to the configuration file to be loaded.
                If no path is provided, an empty configuration is initialized.
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)
        self._set_exp_name_during_init(config_path)

    def __setattr__(self, key: str, value: any) -> None:
        """Override setattr to ensure the config dictionary is always in sync with attributes.

        Args:
            key (str): The attribute key.
            value (any): The attribute value.
        """
        if key != "config":  # Avoid recursion for 'config' itself
            self.config[key] = value
        super().__setattr__(key, value)

    def _set_exp_name_during_init(self, config_path: str = None) -> None:
        """Set the default experiment name during initialization. If no name is provided,
        it generates a name based on the current timestamp.

        Args:
            config_path (str, optional): The path to the configuration file to be loaded.
        """
        if not hasattr(self, "exp_name"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if config_path:
                config_name = os.path.splitext(os.path.basename(config_path))[0]
                self.exp_name = f"{config_name}_{timestamp}"
            else:
                self.exp_name = f"exp_{timestamp}"

    def __repr__(self):
        """Formal representation of the Config object for debugging."""
        return f"Config(config={self.config})"

    def __str__(self):
        """User-friendly string representation of the Config object."""
        return f"Config with {len(self.config)} parameters: {list(self.config.keys())}"

    def load_config(self, config_path: str) -> None:
        """Load a configuration file based on its extension.

        Supported formats:
            - .py: Python script containing configuration variables.
            - .yaml: YAML file.

        Args:
            config_path (str): The path to the configuration file to be loaded.

        Raises:
            ValueError: If the file extension is unsupported.
        """
        ext = os.path.splitext(config_path)[1]
        if ext == ".py":
            self._load_py_config(config_path)
        elif ext == ".yaml":
            self._load_yaml_config(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")

    def _load_py_config(self, config_path: str) -> None:
        """Load a Python configuration file and store the parameters as class attributes.

        Args:
            config_path (str): The path to the Python configuration file.
        """
        spec = importlib.util.spec_from_file_location("config", config_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        self.config.update({k: v for k, v in vars(cfg).items() if not k.startswith("__")})
        self._set_attrs_from_dict(self.config)

    def _load_yaml_config(self, config_path: str) -> None:
        """Load a YAML configuration file and store the parameters as class attributes.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            self.config.update(yaml.safe_load(file))
        self._set_attrs_from_dict(self.config)

    def _set_attrs_from_dict(self, config_dict: dict) -> None:
        """Set configuration parameters as class attributes.

        Args:
            config_dict (dict): The dictionary of configuration parameters.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_args(self, args: object) -> None:
        """Update configuration based on command-line arguments.

        This function overrides configuration parameters with values provided via
        command-line arguments, if they are not `None`.

        Args:
            args: An object (typically from `argparse.Namespace`) containing command-line
                arguments and their values.
        """
        for key, value in vars(args).items():
            if value is not None:
                self.config[key] = value
                setattr(self, key, value)

        # Handle exp_name override via command-line args
        if hasattr(args, "exp_name") and args.exp_name:
            setattr(self, "exp_name", args.exp_name)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value with a default fallback.

        This function retrieves a configuration value by key. If the key is not found,
        it returns the provided default value.

        Args:
            key (str): The key of the configuration parameter to retrieve.
            default (optional): The value to return if the key is not found. Defaults to None.

        Returns:
            The value associated with the key, or the default value if the key is not found.
        """
        return self.config.get(key, default)
