# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .centernet_head import CenterNetHead


@HEADS.register_module()
class HoughNetHead(CenterNetHead):
    """ HoughNet Head. HoughNetHead use center_point to indicate object's
    position. Paper link <https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700409.pdf>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 region_num,
                 vote_field_size,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(HoughNetHead, self).__init__(in_channel,
                                           feat_channel,
                                           num_classes,
                                           loss_center_heatmap=loss_center_heatmap,
                                           loss_wh=loss_wh,
                                           loss_offset=loss_offset,
                                           train_cfg=train_cfg,
                                           test_cfg=test_cfg,
                                           init_cfg=init_cfg)
        self.heatmap_head = self._build_head_hough(in_channel, feat_channel,
                                                    num_classes, region_num, vote_field_size)

    def _build_head_hough(self, in_channel, feat_channel, num_classes, region_num, vote_field_size):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, num_classes * region_num, kernel_size=1),
            Hough(region_num=region_num,
                  vote_field_size=vote_field_size,
                  num_classes=num_classes)
        )

        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-2].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)


class Hough(nn.Module):

    def __init__(self, angle=90, R2_list=[4, 64, 256, 1024],
                 num_classes=80, region_num=9, vote_field_size=17,
                 voting_map_size_w=128, voting_map_size_h=128):
        super(Hough, self).__init__()
        self.angle = angle
        self.R2_list = R2_list
        self.region_num = region_num
        self.num_classes = num_classes
        self.vote_field_size = vote_field_size
        self.deconv_filter_padding = int(self.vote_field_size / 2)
        self.voting_map_size_w = voting_map_size_w
        self.voting_map_size_h = voting_map_size_h
        self.deconv_filters = self._prepare_deconv_filters()

    def _prepare_deconv_filters(self):

        half_w = int(self.voting_map_size_w / 2)
        half_h = int(self.voting_map_size_h / 2)

        vote_center = torch.tensor([half_h, half_w]).cuda()

        logmap_onehot = self.calculate_logmap((self.voting_map_size_h, self.voting_map_size_w), vote_center)

        weights = logmap_onehot / \
                        torch.clamp(torch.sum(torch.sum(logmap_onehot, dim=0), dim=0).float(), min=1.0)

        start_x = half_h - int(self.vote_field_size/2)
        stop_x  = half_h + int(self.vote_field_size/2) + 1

        start_y = half_w - int(self.vote_field_size/2)
        stop_y  = half_w + int(self.vote_field_size/2) + 1

        deconv_filters = weights[start_x:stop_x, start_y:stop_y,:].permute(2,0,1).view(self.region_num, 1,
                                                                     self.vote_field_size, self.vote_field_size)

        W = nn.Parameter(deconv_filters.repeat(self.num_classes, 1, 1, 1))
        W.requires_grad = False

        layers = []
        deconv_kernel = nn.ConvTranspose2d(
            in_channels=self.region_num*self.num_classes,
            out_channels=1*self.num_classes,
            kernel_size=self.vote_field_size,
            padding=self.deconv_filter_padding,
            groups=self.num_classes,
            bias=False)

        with torch.no_grad():
            deconv_kernel.weight = W

        layers.append(deconv_kernel)

        return nn.Sequential(*layers)

    def generate_grid(self, h, w):
        x = torch.arange(0, w).float().cuda()
        y = torch.arange(0, h).float().cuda()
        grid = torch.stack([x.repeat(h), y.repeat(w, 1).t().contiguous().view(-1)], 1)
        return grid.repeat(1, 1).view(-1, 2)

    def calculate_logmap(self, im_size, center, angle=90, R2_list=[4, 64, 256, 1024]):
        PI = np.pi
        points = self.generate_grid(im_size[0], im_size[1])  # [x,y]
        total_angles = 360 / angle

        # check inside which circle
        y_dif = points[:, 1].cuda() - center[0].float()
        x_dif = points[:, 0].cuda() - center[1].float()

        xdif_2 = x_dif * x_dif
        ydif_2 = y_dif * y_dif
        sum_of_squares = xdif_2 + ydif_2

        # find angle
        arc_angle = (torch.atan2(y_dif, x_dif) * 180 / PI).long()

        arc_angle[arc_angle < 0] += 360

        angle_id = (arc_angle / angle).long() + 1

        c_region = torch.ones(xdif_2.shape, dtype=torch.long).cuda() * len(R2_list)

        for i in range(len(R2_list) - 1, -1, -1):
            region = R2_list[i]
            c_region[(sum_of_squares) <= region] = i

        results = angle_id + (c_region - 1) * total_angles
        results[results < 0] = 0

        results.view(im_size[0], im_size[1])

        logmap = results.view(im_size[0], im_size[1])
        logmap_onehot = torch.nn.functional.one_hot(logmap.long(), num_classes=17).float()
        logmap_onehot = logmap_onehot[:, :, :self.region_num]

        return logmap_onehot

    def forward(self, voting_map):
        return self.deconv_filters(voting_map)
