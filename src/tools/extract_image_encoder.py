import torch

import os
import glob


def find_checkpoint_file(exp, epoch, base_checkpoint_dir):
    # Construct the pattern to search for files containing the epoch number
    exp_pattern = f"exp{exp}*"
    epoch_pattern = f"*epoch={epoch}_*.ckpt"
    search_path = os.path.join(base_checkpoint_dir, exp_pattern, epoch_pattern)

    # Use glob to find files matching the pattern
    matching_files = glob.glob(search_path)

    if len(matching_files) == 0:
        raise FileNotFoundError(f"No checkpoint file found for epoch {epoch} in {search_path}")
    elif len(matching_files) > 1:
        print(
            f"Multiple checkpoint files found for epoch {epoch} in {search_path}. Using the first one."
        )

    # Return the first matching file (assuming you want to use the first match)
    return matching_files[0]


def extract_weights_for_experiments(experiments, epochs):
    base_checkpoint_dir = "/homes/math/golombiewski/workspace/fast/clcl/checkpoints/"
    output_dir = "/homes/math/golombiewski/workspace/data/models/"

    for exp in experiments:
        for epoch in epochs:
            # Find the checkpoint file for the given experiment and epoch
            checkpoint_path = find_checkpoint_file(str(exp), epoch, base_checkpoint_dir)

            # Construct the output filename
            exp_name = f"exp{exp}_{checkpoint_path.split('_')[1]}_epoch{epoch}"
            target_path = os.path.join(output_dir, f"{exp_name}.pth")

            # Extract the weights
            extract_timm_weights_from_checkpoint(checkpoint_path, target_path)
            # print(f"Extracted weights for {exp_name} to {target_path}")


def print_state_dict(state_dict):
    max_key_length = max(len(key) for key in state_dict.keys())
    for key, tensor in state_dict.items():
        padding = " " * (max_key_length - len(key))
        print(f"{key}{padding} | Shape: {tensor.shape}")


def extract_timm_weights_from_checkpoint(checkpoint_path, output_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]

    # Filter out image_encoder keys and remove projection weights
    timm_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("image_encoder.model"):
            new_key = key.replace("image_encoder.model.", "")
            timm_state_dict[new_key] = value

    torch.save(timm_state_dict, output_path)
    print("Checkpoint path:\n", checkpoint_path)
    print(f"Image encoder weights from saved to\n {output_path}")


if __name__ == "__main__":
    source_path = "/homes/math/golombiewski/workspace/fast/clcl/checkpoints/exp46_b256_lr1e-3_wd1e-2_wu5_etamin1e-5_linear_unfreeze/exp46_b256_lr1e-3_wd1e-2_wu5_etamin1e-5_linear_unfreeze_epoch=05_val_loss=4.73.ckpt"
    exp_name = "exp46_b256_epoch05"
    target_path = f"/homes/math/golombiewski/workspace/data/models/{exp_name}.pth"
    extract_timm_weights_from_checkpoint(source_path, target_path)

    # experiments = [46]
    # epochs = [24]

    # extract_weights_for_experiments(experiments, epochs)
