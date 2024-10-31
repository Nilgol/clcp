### Base config for all finetuning on Cityscapes.

## KEY PARAMETERS

# model
crop_size = (224, 224)
backbone_checkpoint_path = '/homes/math/golombiewski/workspace/data/liploc_vit_small_patch16_224.pth'
frozen_weights = False

# dataset
train_batch_size = 8
train_workers = 4
val_batch_size = 1
val_workers = 4

# runtime
checkpoint_path = None

# schedule
train_iterations = 80000
learning_rate = 0.01
val_interval = train_iterations // 10
checkpoint_interval = val_interval
log_interval = 100
visualization_interval = 50


## MODEL

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
    checkpoint_path=backbone_checkpoint_path,
    frozen_weights=frozen_weights
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
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(168, 168)),
    train_cfg=dict(),
    type='EncoderDecoder')
#norm_cfg = dict(requires_grad=True, type='SyncBN')


## DATASET

train_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict( type='RandomResize',
         ratio_range=(0.5, 2.0),
         scale=(2048, 1024),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
val_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
test_pipeline = val_pipeline

tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ]),
]

train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root='data/cityscapes/',
        pipeline=train_pipeline,
        type='CityscapesDataset'),
    num_workers=train_workers,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))

val_dataloader = dict(
    batch_size=val_batch_size,
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root='data/cityscapes/',
        pipeline=val_pipeline,
        type='CityscapesDataset'),
    num_workers=val_workers,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_dataloader = dict(val_dataloader, batch_size=1, num_workers=4)

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'])

test_evaluator = val_evaluator


## DEFAULT RUNTIME

default_scope = 'mmseg'
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend',
                       init_kwargs=dict(
                           project='segmenter_mask_cityscapes'
                            )
                        )
                    ]
)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = checkpoint_path
resume = False

tta_model = dict(type='SegTTAModel')


## SCHEDULE

optimizer = dict(
    lr=learning_rate,
    momentum=0.9,
    type='SGD',
    weight_decay=0.0)

optim_wrapper = dict(
    clip_grad=None,
    optimizer=optimizer,
    type='OptimWrapper')

param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=train_iterations,
        eta_min=1e-4,
        power=0.9,
        type='PolyLR')]

train_cfg = dict(
    max_iters=train_iterations,
    type='IterBasedTrainLoop',
    val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',
                    by_epoch=False,
                    interval=checkpoint_interval),
    logger=dict(type='LoggerHook',
                interval=log_interval,
                log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook',
                       draw=True,
                       interval=visualization_interval))