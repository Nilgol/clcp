import timm
import torch

# Load the pretrained model
model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)

# output = model.forward_features(torch.randn(2, 3, 224, 224))
# model.reset_classifier(0, '')
output = model(torch.randn(2, 3, 224, 224))

print(output.shape)

# Save the state dictionary
#torch.save(model.state_dict(), 'vit_small_patch16_224.pth')
