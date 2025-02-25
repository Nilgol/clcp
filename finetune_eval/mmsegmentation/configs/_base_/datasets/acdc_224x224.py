# dataset settings
dataset_type = 'ACDCDataset'
data_root = 'data/ACDC/'
crop_size = (224, 224)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

# Define separate dataloaders for each condition
conditions = ['fog', 'night', 'rain', 'snow']

train_datasets = []
for cond in conditions:
    train_datasets.append(dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=f'rgb_anon/{cond}/train',
            seg_map_path=f'gt/{cond}/train'),
        pipeline=train_pipeline
    ))

val_datasets = []
for cond in conditions:
    val_datasets.append(dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=f'rgb_anon/{cond}/val',
            seg_map_path=f'gt/{cond}/val'),
        pipeline=val_pipeline
    ))

# Concatenate datasets for training
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_datasets,
        separate_eval=True  # set to False for combined evaluation
    )
)

# Concatenate datasets for validation/testing
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=val_datasets,
        separate_eval=True  # set to False for combined evaluation
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
