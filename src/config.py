import yaml
import importlib.util
import os

import argparse
from icecream import ic


def load_config(config_path):
    """Load a configuration from a .py or .yaml file."""
    ext = os.path.splitext(config_path)[1]
    if ext == ".py":
        return _load_py_config(config_path)
    elif ext == ".yaml":
        return _load_yaml_config(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {ext}")


def _load_py_config(config_path):
    """Load a Python config file and return a dictionary."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("__")}


def _load_yaml_config(config_path):
    """Load a YAML config file and return a dictionary."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def update_config_from_args(config, args):
    """Update a configuration dictionary with command-line arguments."""
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


# Argument parser to fetch learning_rate and batch_size, for testing purposes
def parse_args():
    parser = argparse.ArgumentParser(description="Override config parameters")
    parser.add_argument("-ls", "--learning_rate", type=float, help="Override the learning rate")
    parser.add_argument("-bs", "--batch_size", type=int, help="Override the batch size")
    return parser.parse_args()


if __name__ == "__main__":
    PY_CONFIG_PATH = "/homes/math/golombiewski/workspace/clcl/test/config_sample.py"
    cfg = load_config(PY_CONFIG_PATH)
    ic(cfg["learning_rate"])
    ic(cfg["batch_size"])
    args = parse_args()
    ic(args)
    ic(vars(args).items())
    update_config_from_args(cfg, args)
    ic(cfg["learning_rate"])
    ic(cfg["batch_size"])
