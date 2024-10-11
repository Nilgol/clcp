"""
Main entry point for training a model.

This module loads configuration files, updates them with any command line argument overrides,
and starts the training.
"""

import argparse

from config import load_config, update_config_from_args
from train import train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for overriding configuration parameters.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Override config parameters")

    parser.add_argument("-n", "--exp-name", required=True, help="Name of the experiment")
    parser.add_argument("-cp", "--config_path", type=str, help="Path to the config file")
    parser.add_argument(
        "--checkpoint-save-dir",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--checkpoint-path", default="", help="Path to the checkpoint to load")
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size for training")
    parser.add_argument("-lr", "--learning-rate", type=float, help="Learning rate for training")
    parser.add_argument("-wd", "--weight-decay", type=float, help="Weight decay for training")
    parser.add_argument("-w", "--workers", type=int, help="Number of workers for data loading")
    parser.add_argument("-e", "--max-epochs", type=int, help="Number of epochs to train")
    parser.add_argument("-vr", "--val-ratio", type=float, help="Validation set ratio")

    parser.add_argument(
        "-fl",
        "--freeze-lidar-encoder",
        action="store_true",
        help="Freeze the lidar encoder",
    )
    parser.add_argument(
        "-lom",
        "--load-only-model",
        action="store_true",
        help="Load only the model without training state",
    )
    parser.add_argument(
        "-pt",
        "--projection-type",
        type=str,
        choices=["linear", "mlp"],
        help="Type of projection head",
    )
    parser.add_argument(
        "-a", "--augment", action="store_true", help="Apply image augmentations"
    )

    return parser.parse_args()


def main() -> None:
    """Load a configuration file if provided, update it with any
    command line arguments, and then call the `train` function with the configuration.
    """
    args = parse_args()

    # Create config dict from config file path or empty dict
    if args.config_path:
        config = load_config(args.config_path)
    else:
        config = {}

    # Update config dict with command line arguments
    config = update_config_from_args(config, args)

    # Start training
    train(**config)


if __name__ == "__main__":
    main()
