_base_ = [
    '../_base_/default_runtime.py',
    #'./_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)

num_classes = 19
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=(512,512),
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    #token_embed_dim=768,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='EVA2',
        img_size=512, 
        patch_size=14, 
        num_classes=1000,
        
        in_chans=3,
        embed_dim=1024, 
        depth=24,
        num_heads=16, 
        mlp_ratio=4*2/3,      # GLU default
        out_indices=[9, 14, 19, 23],
        output_dim = 768,
        
        qkv_bias=True, 
        drop_path_rate=0.2, 
        
        init_values=None, 
        use_checkpoint=False, 

        use_abs_pos_emb=True, 
        use_rel_pos_bias=False, 
        use_shared_rel_pos_bias=False,

        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,
        get_embeddings=True,

        subln=True,
        xattn=True,
        naiveswiglu=True,

        pretrained="/home/liangwei/all_models/evaclip_c/pretrained/EVA02_CLIP_L_psz14_s4B.pt",
    ),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[1024, 1024, 1024, 1024], 
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)), #(640, 640) 426
    #text_head=False,
    #text_dim=768, #lw: 512 to 768
    #score_concat_index=2
    )



'''
dataset_type = 'ADE20KDataset'
data_root = '/home/liangwei/all_models/DenseCLIP/ade20k/ADEChallengeData2016'
IMG_MEAN = [122.7709383, 116.7460125, 104.09373615000001]
IMG_VAR = [68.5005327, 66.6321579, 70.32316304999999]
img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615000001],
    std=[68.5005327, 66.6321579, 70.32316304999999],
    to_rgb=True)
'''
crop_size = (512,512)#(640, 640) #400
'''
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[122.7709383, 116.7460125, 104.09373615000001],
        std=[68.5005327, 66.6321579, 70.32316304999999],
        to_rgb=True),
    dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[122.7709383, 116.7460125, 104.09373615000001],
                std=[68.5005327, 66.6321579, 70.32316304999999],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
'''







# dataset settings
dataset_type_train = 'GTADataset'
dataset_type_test = "CityscapesDataset"
data_root_train = '/home/liangwei/data/synthia'
data_root_test = '/home/liangwei/data/cityscapes'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
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
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type_train,
        data_root=data_root_train,
        data_prefix=dict(
            img_path='RGB', seg_map_path='GT/LABELS/LABELS'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_test,
        data_root=data_root_test,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True # lw

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'text_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'backbone.norm': dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.pos_embed': dict(lr_mult=0.1, decay_mult=0.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        }

custom_keys.update({
    f'backbone.blocks.{block_id}.norm1': dict(lr_mult=0.1, decay_mult=0.0)
    for block_id in range(24)
})

custom_keys.update({
    f'backbone.blocks.{block_id}.norm2': dict(lr_mult=0.1, decay_mult=0.0)
    for block_id in range(24)
})

optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=custom_keys,
        norm_decay_mult=0.0))


# learning policy
max_iters = 20000
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.001, begin=0, end=500),
     dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=500,
        end=max_iters,
        by_epoch=False)
]

# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))


#runner = dict(type='IterBasedRunner', max_iters=40000)
#checkpoint_config = dict(by_epoch=False, interval=4000)
#evaluation = dict(interval=4000, metric='mIoU')
#resume_from = "/data2/DenseCLIP_dg-master/mmsegmentation/tools/work_dirs/mask2former_EVA_s_8xb2-90k_gta5-512/iter_5000.pth"
gpu_ids = range(0)
