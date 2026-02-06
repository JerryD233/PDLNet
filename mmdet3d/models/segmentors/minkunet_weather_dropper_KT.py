# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from .encoder_decoder import EncoderDecoder3D

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.utils import rename_loss_dict
import random
import os
import os.path as osp
import numpy as np

import math
from collections import namedtuple, deque
from mmdet3d.structures import PointData
from typing import Dict
from mmdet3d.registry import MODELS

from mmdet3d.utils import ConfigType
from ..utils import add_prefix

@MODELS.register_module()
class MinkUNetWeatherDropper_KT(EncoderDecoder3D):
    r"""MinkUNetWeatherDropper is the implementation of `4D Spatio-Temporal ConvNets.
    <https://arxiv.org/abs/1904.08755>`_ with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`EncoderDecoder3D`.
    """

    def __init__(self,
                # discriminator1: ConfigType,
                teacher_backbone: ConfigType,
                teacher_decode_head: ConfigType,
                 **kwargs) -> None:
        super(MinkUNetWeatherDropper_KT, self).__init__(**kwargs)
        # self.discriminator1 = MODELS.build(discriminator1)
        self.teacher_backbone = MODELS.build(teacher_backbone)
        self.teacher_decode_head = MODELS.build(teacher_decode_head)
        self.epoch = 1

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from voxels.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.

        Returns:
            dict: The dict containing features.
        """
        voxel_dict = batch_inputs_dict['voxels'].copy()
        x = self.backbone(voxel_dict['voxels'], voxel_dict['coors'])
        if self.with_neck:
            x = self.neck(x)
        voxel_dict['voxel_feats'] = x
        return voxel_dict
    
    def teacher_extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from voxels.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.

        Returns:
            dict: The dict containing features.
        """
        voxel_dict = batch_inputs_dict['voxels'].copy()
        x, Qs, coors, rain_data, input_t = self.teacher_backbone(voxel_dict['voxels'], voxel_dict['coors'])
        if self.with_neck:
            x = self.neck(x)
        voxel_dict['voxel_feats'] = x
        return voxel_dict, Qs, coors, rain_data, input_t
    
    def teacher_decode_head_forward_train(
            self, batch_inputs_dict: dict,
            batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Run forward function and calculate loss for decode head in training.

        Args:
            batch_input (Tensor): Input point cloud sample
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components for decode head.
        """
        losses = dict()
        loss_decode = self.teacher_decode_head.loss(batch_inputs_dict,
                                            batch_data_samples, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'teacher_decode'))
        return losses

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        losses = dict()

        # if self.epoch <= 5:
        #     lam = 0.1
        # elif self.epoch <=10:
        #     lam = 1
        # else:
        #     lam = 10

        bt = batch_data_samples.copy()
        voxel_dict, Qs, coors, rain_data, input_t = self.teacher_extract_feat(batch_inputs_dict)
        teacher_loss_decode = self.teacher_decode_head_forward_train(voxel_dict,
                                                    batch_data_samples)
        losses.update(teacher_loss_decode)

        # loss_adv = compute_adv_shallow_loss(Qs, coors, self.discriminator1)
        # losses.update({'loss_adv': loss_adv})


        result = self.data_preprocessor.all_voxelize(rain_data, None)
        results = {'voxels': result}
        self.teacher_backbone.use_target = False
        pred_t = self.teacher_predict(results, bt)
        self.teacher_backbone.use_target = True

        for i in range(len(pred_t)):
            bt[i].gt_pts_seg.pts_semantic_mask = pred_t[i].pred_pts_seg.pts_semantic_mask

        inputs_t = {'voxels': input_t}
        voxel_dict_t = self.extract_feat(inputs_t)
        loss_decode = self._decode_head_forward_train(voxel_dict_t,
                                                    bt)
        
        # loss_decode['decode.loss_ce'] *= lam
        losses.update(loss_decode)

        return losses
    
    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        voxel_dict = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(voxel_dict,
                                                   batch_data_samples)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)
        return self.postprocess_result(seg_logits_list, batch_data_samples)
    
    def teacher_predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        voxel_dict, _, _, _, _ = self.teacher_extract_feat(batch_inputs_dict)
        seg_logits_list = self.teacher_decode_head.predict(voxel_dict,
                                                   batch_data_samples)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)
        return self.postprocess_result(seg_logits_list, batch_data_samples)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)
    
def compute_discriminator1_loss(Qs, coors, Qt, rain_coors, D1):
    """
    Qs: Generated source domain voxel features with target style.
    Qt: Target domain voxel features.
    D1: Discriminator instance for aligning Qs and Qt.
    """
    # Compute the discriminator scores
    Qt_scores = D1(Qt, rain_coors)
    Qs_scores = D1(Qs.detach(), coors.detach())

    # Compute the loss for Qs and Qt
    loss_Qs = torch.mean((Qs_scores - 0)**2)
    loss_Qt = torch.mean((Qt_scores - 1)**2)

    # Total discriminator loss
    L_d1 = loss_Qs + loss_Qt

    return L_d1


def compute_adv_shallow_loss(Qs, coors, D1):
    Qs_scores = D1(Qs, coors)

    L_adv_shallow = torch.mean((Qs_scores - 1)**2)

    return L_adv_shallow