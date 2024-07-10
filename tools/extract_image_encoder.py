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

    # Filter out only the image_encoder keys and remove projection weights
    timm_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("image_encoder.model"):
            # Strip the 'image_encoder.model' prefix
            new_key = key.replace("image_encoder.model.", "")
            timm_state_dict[new_key] = value

    # Save the TIMM model weights to a new file
    torch.save(timm_state_dict, output_path)
    print(f"TIMM image encoder weights saved to {output_path}")


if __name__ == "__main__":
    clcl_ckpt = "/homes/math/golombiewski/workspace/fast/clcl/checkpoints/last.ckpt"
    timm_vit_pth = (
        "/homes/math/golombiewski/workspace/data/timm_vit_small_patch16_224.pth"
    )
    liploc_pth = (
        "/homes/math/golombiewski/workspace/data/liploc_vit_small_patch16_224.pth"
    )
    clcl_epoch30 = "/homes/math/golombiewski/workspace/fast/clcl/checkpoints/epoch=30-step=27714.ckpt"
    # state_dict = checkpoint['state_dict']
    target_path = "/homes/math/golombiewski/workspace/data/clcl_vit_epoch30.pth"
    extract_timm_weights_from_checkpoint(clcl_epoch30, target_path)
    # checkpoint = torch.load(target_path)
    # print_state_dict(checkpoint)
