import argparse
from train import train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of the experiment")
    parser.add_argument(
        "--checkpoint-save-dir",
        default="/homes/math/golombiewski/workspace/fast/clcl/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint", default="", help="Path to the checkpoint to load"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay for training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio"
    )
    parser.add_argument(
        "--freeze-lidar-encoder", action="store_true", help="Freeze the lidar encoder"
    )
    parser.add_argument(
        "--load-only-model",
        action="store_true",
        help="Load only the model without training state",
    )
    parser.add_argument(
        "--projection-type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Type of projection head",
    )
    parser.add_argument(
        "--augment",
        action="store_true",  # Boolean flag
        help="Apply image augmentations",
    )
    args = parser.parse_args()
    assert args.name, "Empty name is not allowed"
    return args

def main():
    args = parse_args()
    train(
        checkpoint_path=args.checkpoint,
        exp_name=args.name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        max_epochs=args.max_epochs,
        freeze_lidar_encoder=args.freeze_lidar_encoder,
        load_only_model=args.load_only_model,
        checkpoint_save_dir=args.checkpoint_save_dir,
        projection_type=args.projection_type,
        augment=args.augment,
    )


if __name__ == "__main__":
    main()