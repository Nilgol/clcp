import torch


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
    print("Checkpoint path:", checkpoint_path)
    print(f"Image encoder weights from saved to {output_path}")


if __name__ == "__main__":
    source_path = (
        "/homes/math/golombiewski/workspace/fast/clcl/checkpoints/clcl_exp6_b768/last.ckpt"
    )
    exp_name = "clcl_vit_b768"
    target_path = f"/homes/math/golombiewski/workspace/data/models/{exp_name}.pth"
    extract_timm_weights_from_checkpoint(source_path, target_path)
