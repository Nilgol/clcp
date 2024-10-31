import torch
from liploc.models.CLIPModelV1_vit import Model as LIP_Loc

# Load model config
from liploc.config.exp_largest_vit import CFG

# Update config (if necessary)
CFG.trained_image_model_name = 'vit_small_patch16_224'
CFG.pretrained = False
CFG.trainable = False

# Initialize  LIP-Loc model
lip_model = LIP_Loc(CFG)

# Path to pretrained weights
model_path = '/work/golombiewski/liploc/data/exp_largest_vit/best.pth'

# Load pretrained weights
lip_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Sanity check
dummy_input = torch.randn(1, 3, 224, 224).to('cpu')
with torch.no_grad():
    embeddings = lip_model.encoder_camera(dummy_input)
    print(embeddings.shape)
