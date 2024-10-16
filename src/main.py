"""Main entry point for training a model.

This module loads configuration files, updates them with any command-line argument overrides,
and starts the training.
"""

import argparse
import os
from datetime import datetime

from config import Config
from train import train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for overriding configuration parameters.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Override config parameters")

    parser.add_argument("-n", "--exp-name", type=str, help="Name of the experiment")
    parser.add_argument("-cfg", "--config_path", type=str, help="Path to the config file")
    parser.add_argument("--checkpoint-save-dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-path", type=str, help="Path to the checkpoint to load")
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size for training")
    parser.add_argument("-lr", "--learning-rate", type=float, help="Learning rate for training")
    parser.add_argument("-wd", "--weight-decay", type=float, help="Weight decay for training")
    parser.add_argument("-w", "--workers", type=int, help="Number of workers for data loading")
    parser.add_argument("-e", "--max-epochs", type=int, help="Number of epochs to train")
    parser.add_argument("-vr", "--val-ratio", type=float, help="Validation set ratio")

    parser.add_argument(
        "-fl", "--freeze-lidar-encoder", type=bool, help="Freeze the lidar encoder"
    )
    parser.add_argument(
        "-lom",
        "--load-only-model",
        type=bool,
        help="When loading checkpoint, load only model without training state",
    )
    parser.add_argument(
        "-pt",
        "--projection-type",
        type=str,
        choices=["linear", "mlp"],
        help="Type of projection head",
    )
    parser.add_argument(
        "-a", "--augment", type=bool, help="Apply additional image augmentations"
    )

    return parser.parse_args()


def main() -> None:
    """Load a configuration file if provided, update it with any
    command-line arguments, and then call the `train` function with the configuration.
    """
    args = parse_args()

    # Create config dict from config file path or empty dict
    if args.config_path:
        cfg = Config(args.config_path)
    else:
        cfg = Config()

    # Update config dict with command-line arguments
    cfg.update_from_args(args)

    # Start training
    train(cfg)


if __name__ == "__main__":
    main()
