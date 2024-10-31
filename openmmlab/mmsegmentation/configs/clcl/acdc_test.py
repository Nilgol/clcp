_base_ = [
    '../_base_/datasets/acdc_224x224.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py']


# MODEL

crop_size = (224, 224)

data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')

custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmpretrain.models',
    ])
backbone=dict(
    type='mmpretrain.TIMMBackbone',
    model_name='vit_small_patch16_224',
    features_only=False,
    pretrained=False,
    # checkpoint_path=backbone_checkpoint_path,
    # frozen_weights=frozen_weights
    )

model = dict(
    backbone=backbone,
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=384,
        channels=384,
        num_classes=19,
        num_layers=2,
        num_heads=6,
        embed_dims=384,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(150, 150)),
    train_cfg=dict(),
    type='EncoderDecoder')


# RUNTIME

visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
)