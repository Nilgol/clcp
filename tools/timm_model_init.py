import timm
import torch

# Load model
model = timm.create_model(
    'vit_small_patch16_224',
    pretrained=True,
    num_classes=0,
    )
model.eval()  # Set the model to evaluation mode

# Sanity check
dummy_input = torch.randn(12, 3, 224, 224)  # [batch_size, channels, height, width]
with torch.no_grad():
    output = model(dummy_input)

# print(model)
print(output.shape)
print(model.num_features)
#print(output)
