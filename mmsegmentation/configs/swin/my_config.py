#from mmseg.datasets.forest_dataset import ForestDataset
#from configs.swin.to_float import ToFloat

img_norm_cfg = dict(
    mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True)

_base_ = [
    '../_base_/default_runtime.py'
]


DATA_DIR = '/data/data1/users/lefterislymp/coast_dir/resuneta/'

norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(type='Adam', lr=1e-4, betas=(0.9, 0.999), weight_decay=0,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

optimizer_config = dict()
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        transforms=[
            dict(type='RandomFlip', flip_ratio=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        transforms=[
            dict(type='RandomFlip', flip_ratio=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(  # Train dataset config
        type='CustomDataset',  # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=DATA_DIR,  # The root of dataset.
        img_dir='data/train',  # The image directory of dataset.
        ann_dir='annotations/train',  # The annotation directory of dataset.
        img_suffix='.png',
        seg_map_suffix='_labelTrainIds.png',
        classes=('unlabeled', 'forest'),
        palette=[[192, 0, 64],[0, 192, 64]],
        pipeline=[  # pipeline, this is passed by the train_pipeline created before.
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomFlip', flip_ratio=0),
            dict(type='Transpose', order=(0, 1, 2), keys=['img']),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(  # Validation dataset config
        type='CustomDataset',
        data_root=DATA_DIR,
        img_dir='data/val',
        ann_dir='annotations/val',
        img_suffix='.png',
        seg_map_suffix='_labelTrainIds.png',
        classes=('unlabeled', 'forest'),
        palette=[[192, 0, 64],[0, 192, 64]],
        pipeline=[  # Pipeline is passed by test_pipeline created before
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
                flip=False,
                transforms=[
                dict(type='RandomFlip', flip_ratio=0),
                #dict(type='Transpose', order=(0, 1, 2), keys=['img']),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                #dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img'])])
        ]),
    test=dict(
        type='CustomDataset',
        data_root=DATA_DIR,
        img_dir='data_mix',
        ann_dir='annotations_mix',
        img_suffix='.png',
        seg_map_suffix='_labelTrainIds.png',
        classes=('unlabeled', 'forest'),
        palette=[[192, 0, 64],[0, 192, 64]],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
                flip=False,
                transforms=[
                    dict(type='RandomFlip', flip_ratio=0),
                    # dict(type='Transpose', order=(0, 1, 2), keys=['img']),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    # dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])])
        ])
)

log_level = 'INFO'

load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None

checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=True,  # Whether count by epoch or not.
    interval=1)  # The save interval.

dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'

runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=70,

    max_iters=None)

evaluation = dict(  # The config to build the evaluation hook. Please refer to mmseg/core/evaluation/eval_hook.py for details.
    by_epoch=True,
    interval=1,  # The interval of evaluation.
    metric='mIoU')  # The evaluation metric.
