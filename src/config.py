import yaml
import importlib.util
import os

class Config:
    def __init__(self, config_path=None):
        self.config = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        """Load a config file based on the extension."""
        ext = os.path.splitext(config_path)[1]
        if ext == ".py":
            self._load_py_config(config_path)
        elif ext == ".yaml":
            self._load_yaml_config(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")

    def _load_py_config(self, config_path):
        """Load a Python config file."""
        spec = importlib.util.spec_from_file_location("config", config_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        # Update the internal dictionary with all non-private variables
        self.config.update({k: v for k, v in vars(cfg).items() if not k.startswith("__")})
        # Set these variables as attributes
        self._set_attrs_from_dict(self.config)

    def _load_yaml_config(self, config_path):
        """Load a YAML config file."""
        with open(config_path, 'r') as file:
            self.config.update(yaml.safe_load(file))
        # Set these variables as attributes
        self._set_attrs_from_dict(self.config)

    def _set_attrs_from_dict(self, config_dict):
        """Set config variables as class attributes for easy access."""
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_args(self, args):
        """Update config based on command-line arguments."""
        for key, value in vars(args).items():
            if value is not None:
                self.config[key] = value
                setattr(self, key, value)

    def get(self, key, default=None):
        """Get a configuration value with a default fallback."""
        return self.config.get(key, default)
