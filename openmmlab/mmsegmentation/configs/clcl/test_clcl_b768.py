_base_ = ['./segmenter_mask_acdc.py']

backbone_checkpoint_path = '/homes/math/golombiewski/workspace/data/models/clcl_vit_b768.pth'
frozen_weights = False

backbone=dict(
    type='mmpretrain.TIMMBackbone',
    model_name='vit_small_patch16_224',
    features_only=False,
    pretrained=False,
    checkpoint_path=backbone_checkpoint_path,
    frozen_weights=frozen_weights
    )

model = dict(backbone=backbone)