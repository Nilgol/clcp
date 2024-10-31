_base_ = ['./segmenter_mask_cityscapes_b64.py']

backbone_checkpoint_path = '/homes/math/golombiewski/workspace/data/timm_vit_small_patch16_224.pth'
frozen_weights = False

model = dict(
    backbone=dict(
    type='mmpretrain.TIMMBackbone',
    model_name='vit_small_patch16_224',
    features_only=False,
    pretrained=False,
    checkpoint_path=backbone_checkpoint_path,
    frozen_weights=frozen_weights,
    num_classes=1000 ## So backbone weights (including classifier) can be loaded
    )
)