_base_ = './houghnet_resnet18_dcnv2_140e_coco.py'

model = dict(
    backbone=dict(type='ResNet',
                  depth=101,
                  init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(in_channel=2048),
)

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=12,
)
