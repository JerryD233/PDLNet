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
from typing import Dict, Optional, Tuple, Union
from mmdet3d.registry import MODELS
from mmengine.utils import is_list_of
from collections import OrderedDict
from mmengine.optim import OptimWrapper, OptimWrapperDict

from mmdet3d.utils import ConfigType

import pickle, numpy as np

import torch.nn.functional as F

from mmdet3d.utils import ConfigType

@MODELS.register_module()
class MinkUNetWeatherDropper_D1(EncoderDecoder3D):
    r"""MinkUNetWeatherDropper is the implementation of `4D Spatio-Temporal ConvNets.
    <https://arxiv.org/abs/1904.08755>`_ with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`EncoderDecoder3D`.
    """

    def __init__(self,
                discriminator1: ConfigType,
                # ada_in: ConfigType,
                # crossAttention: ConfigType,
                 **kwargs) -> None:
        super(MinkUNetWeatherDropper_D1, self).__init__(**kwargs)
        self.discriminator1 = MODELS.build(discriminator1)
        # self.ada_in = MODELS.build(ada_in)
        # self.crossAttention = MODELS.build(crossAttention)
        # self.GlobalRotScaleTrans = GlobalRotScaleTrans()

        with open('data/SemanticSTF/semanticstf_infos_train.pkl', 'rb') as f:
            self.data_dicts = pickle.load(f)


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
        x, Qs, coors = self.backbone(voxel_dict['voxels'], voxel_dict['coors'])
        if self.with_neck:
            x = self.neck(x)
        voxel_dict['voxel_feats'] = x
        return voxel_dict, Qs, coors

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
        # print('进入了分割的loss函数')
        losses = dict()

        voxel_dict, Qs, coors = self.extract_feat(batch_inputs_dict)
        loss_decode = self._decode_head_forward_train(voxel_dict,
                                                    batch_data_samples)
        losses.update(loss_decode)

        loss_adv = compute_adv_shallow_loss(Qs, coors, self.discriminator1)
        losses.update({'loss_adv': loss_adv})

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
        voxel_dict, _, _ = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(voxel_dict,
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
    
    # def train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapperDict) -> Dict[str, torch.Tensor]:
    # # # direct
    # # def train_step(self, data: Union[dict, tuple, list],optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
    #     """Define the training step to update both generator and discriminator separately."""
    #     """
    #     Example::

    #     # Creates a GradScaler once at the beginning of training.
    #     scaler = GradScaler()

    #     for epoch in epochs:
    #         for input, target in data:
    #             optimizer.zero_grad()
    #             output = model(input)
    #             loss = loss_fn(output, target)

    #             # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
    #             scaler.scale(loss).backward()

    #             # scaler.step() first unscales gradients of the optimizer's params.
    #             # If gradients don't contain infs/NaNs, optimizer.step() is then called,
    #             # otherwise, optimizer.step() is skipped.
    #             scaler.step(optimizer)

    #             # Updates the scale for next iteration.
    #             scaler.update()
    #     """
    #     data = self.data_preprocessor(data, True)

    #     ########################### direct ###########################
    #     # voxel_dict = data['inputs']['voxels'].copy()
    #     # generated_voxel_features, coors, rain_voxel_features, rain_coors = self.adainCrossAttention(voxel_dict['voxels'], voxel_dict['coors'])
    #     # # 更新分割网络的数据
    #     # data['inputs']['voxels']['voxels'] = generated_voxel_features

    #     # # 更新分割网络
    #     # with optim_wrapper.optim_context(self):
    #     #     loss_seg = self._run_forward(data, mode='loss')
    #     #     parse_loss, log_vars = self.parse_losses(loss_seg)

    #     #     optim_wrapper.update_params(parse_loss)
    #     ########################### direct ###########################

    #     ########################### D1 ###########################
    #     # 更新判别器
    #     optimizer_discriminator1 = optim_wrapper['discriminator1']
    #     # 更新分割网络
    #     optimizer_segbackbone = optim_wrapper['segbackbone']
    #     with optimizer_segbackbone.optim_context(self):
    #         data = self.data_preprocessor(data, True)
    #         losses, loss_d1 = self._run_forward(data, mode='loss')  # type: ignore
    #     parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
    #         # optimizer_discriminator1 = optim_wrapper['discriminator1']
    #         # with optim_wrapper.optim_context(self):
    #         #     # 更新判别器
    #         #     optimizer_discriminator1.update_params(loss_d1)
    #     optimizer_discriminator1.update_params(loss_d1)
    #     optimizer_segbackbone.update_params(parsed_losses)
    #     ########################### D1 ###########################
    #     return log_vars

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

class GlobalRotScaleTrans:
    def __init__(self,
                #  rot_range=[-0.78539816, 0.78539816],
                rot_range=[0., 6.28318531],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0], shift_height=False):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = np.array(translation_std, dtype=np.float32)
        self.shift_height = shift_height
    
    def _random_rotation_matrix(self, angle):
        """Generate a random rotation matrix around Z-axis."""
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]], dtype=np.float32)
        return torch.from_numpy(rotation_matrix)

    def _apply_translation(self, points, trans_factor):
        """Translate the points by given factor."""
        return points + trans_factor

    def _apply_rotation(self, points, rot_matrix):
        """Rotate the points using given rotation matrix."""
        rotated_points = torch.matmul(points, rot_matrix.t())
        return rotated_points

    def _apply_scaling(self, points, scale_factor):
        """Scale the points by given factor."""
        return points * scale_factor

    def transform(self, points):
        """Apply rotation, scaling, and translation to the points."""
        points = points.clone()
        
        # Translation
        trans_factor = torch.from_numpy(np.random.normal(scale=self.translation_std)).float().to(points.device)
        points = self._apply_translation(points, trans_factor)
        
        # Rotation
        noise_rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        rotation_matrix = self._random_rotation_matrix(noise_rotation)
        points = self._apply_rotation(points, rotation_matrix)
        
        # Scaling
        scale_factor = np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
        points = self._apply_scaling(points, scale_factor)
        
        return points