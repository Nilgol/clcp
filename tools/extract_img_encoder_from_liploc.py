"""A script to extract the pretrained image encoder weights from the LIP-Loc model."""
import torch
import timm

# Path saved liploc weights
weights_path = 'liploc_camera_encoder.pth'

# Load the weights
state_dict = torch.load(weights_path)

# Filter keys of the state dictionary
adjusted_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
filtered_state_dict = {k: v for k, v in adjusted_state_dict.items() if not k.startswith('fc.')}

# Initialize the timm model
model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)

# Load the weights into the model
missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

# Check for any missing or unexpected keys
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

if len(missing_keys) + len(unexpected_keys) == 0:
    torch.save(filtered_state_dict, 'liploc_vit_weights.pth')
    print('Saved LIP-Loc weights as liploc_vit_weights.pth')