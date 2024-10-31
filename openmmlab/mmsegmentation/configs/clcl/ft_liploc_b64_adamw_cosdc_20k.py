_base_ = ['./segmenter_mask_cityscapes_adam_cosdc_b64_20k.py']

backbone_checkpoint_path = '/homes/math/golombiewski/workspace/data/models/liploc_vit_small_patch16_224.pth'
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

visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend',
                       init_kwargs=dict(
                           project='segmenter_mask_cityscapes_adamw_cosdc_b64_10k',
                            name='ft_liploc_b64_adamw_cosdc_20k'
                            )
                        )
                    ]
)