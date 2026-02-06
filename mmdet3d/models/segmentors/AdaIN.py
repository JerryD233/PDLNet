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

# class AdaIN:
#     """
#     Adaptive Instance Normalization adapted for point cloud style transfer.
#     """

#     def _compute_mean_std_per_batch(
#         self, feats: torch.Tensor, batch_indices: torch.Tensor, eps=1e-8
#     ) -> torch.Tensor:
#         """
#         Compute the mean and standard deviation of the features per batch.

#         Args:
#             feats (torch.Tensor): Input feature tensor with shape (N, C).
#             batch_indices (torch.Tensor): Batch indices tensor with shape (N,) indicating which batch each point belongs to.

#         Returns:
#             tuple: A tuple containing two tensors, each of shape (B, C) where B is the number of batches,
#                    representing the mean and standard deviation per batch.
#         """
#         unique_batches = torch.unique(batch_indices)
#         means, stds = [], []
        
#         for batch_id in unique_batches:
#             batch_feats = feats[batch_indices == batch_id]
#             mean = torch.mean(batch_feats, dim=0, keepdim=True)  # Shape: (1, C)
#             std = torch.std(batch_feats, dim=0, keepdim=True) + eps  # Shape: (1, C)
#             means.append(mean)
#             stds.append(std)

#         return torch.cat(means), torch.cat(stds)

#     def __call__(
#         self,
#         content_feats: torch.Tensor,
#         style_feats: torch.Tensor,
#         content_batch_indices: torch.Tensor,
#         style_batch_indices: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Apply Adaptive Instance Normalization to the content features using style features, considering batch information.

#         Args:
#             content_feats (torch.Tensor): Content features with shape (N, C).
#             style_feats (torch.Tensor): Style features with shape (M, C).
#             content_batch_indices (torch.Tensor): Batch indices tensor with shape (N,) for content points.
#             style_batch_indices (torch.Tensor): Batch indices tensor with shape (M,) for style points.

#         Returns:
#             torch.Tensor: Normalized content features with shape (N, C).
#         """
#         unique_batches = torch.unique(content_batch_indices)
#         normalized_feats = torch.zeros_like(content_feats)
        
#         for batch_id in unique_batches:
#             # 获取当前批次的内容和风格特征
#             c_batch_feats = content_feats[content_batch_indices == batch_id]
#             s_batch_feats = style_feats[style_batch_indices == batch_id]

#             # 计算当前批次的内容和风格特征的均值和标准差
#             c_mean, c_std = self._compute_mean_std_per_batch(c_batch_feats, torch.zeros(c_batch_feats.shape[0], dtype=torch.long))
#             s_mean, s_std = self._compute_mean_std_per_batch(s_batch_feats, torch.zeros(s_batch_feats.shape[0], dtype=torch.long))

#             # 应用AdaIN
#             normalized_batch = (s_std * (c_batch_feats - c_mean) / c_std) + s_mean
            
#             # 将结果放回normalized_feats中相应的位置
#             normalized_feats[content_batch_indices == batch_id] = normalized_batch
        
#         return normalized_feats

import torch.distributed as dist

@MODELS.register_module()
class AdaIN(nn.Module):
    def __init__(self, alpha=0.1, feat_channels=4):
        """
        初始化AdaIN模块。
        
        参数:
            alpha (float): 用于平滑更新全局统计量的系数。
            feat_channels (int): 特征通道数，用于初始化全局均值和标准差。
        """
        super(AdaIN, self).__init__()
        self.alpha = alpha
        # if feat_channels is not None:
        #     # 初始化全局均值为0，全局标准差为1
        #     self.register_buffer('global_mean', torch.zeros(1, feat_channels))
        #     self.register_buffer('global_std', torch.ones(1, feat_channels))
        # else:
        #     self.global_mean = None
        #     self.global_std = None
        self.global_mean = None
        self.global_std = None

    def _compute_mean_std_per_batch(self, feats: torch.Tensor, eps=1e-8) -> tuple:
        """
        计算给定特征的均值和标准差。
        
        参数:
            feats (torch.Tensor): 输入特征张量，形状为 (N, C)，其中 N 是样本数，C 是通道数。
            eps (float): 防止除零的小常数。
        
        返回:
            mean (torch.Tensor): 均值，形状为 (1, C)。
            std (torch.Tensor): 标准差，形状为 (1, C)。
        """
        mean = torch.mean(feats, dim=0, keepdim=True)  # Shape: (1, C)
        std = torch.std(feats, dim=0, keepdim=True) + eps  # Shape: (1, C)
        return mean, std

    def update_and_sync_global_stats(self, style_mean, style_std):
        """
        更新全局均值和标准差，并在分布式训练环境中同步这些统计量。
        """
        
        # if self.global_mean is None or self.global_std is None:
        #     raise RuntimeError("Global statistics not initialized. Please specify 'feat_channels' during initialization.")
        
        # # 平滑更新全局统计量
        # self.global_mean.data = (1 - self.alpha) * self.global_mean + self.alpha * style_mean
        # self.global_std.data = (1 - self.alpha) * self.global_std + self.alpha * style_std

        if self.global_mean is None:
            self.global_mean = style_mean.clone()
            self.global_std = style_std.clone()
        else:
            self.global_mean = (1 - self.alpha) * self.global_mean + self.alpha * style_mean
            self.global_std = (1 - self.alpha) * self.global_std + self.alpha * style_std

        
        # 如果是分布式训练环境，则同步全局统计量
        if dist.is_initialized():
            dist.all_reduce(self.global_mean, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.global_std, op=dist.ReduceOp.SUM)
            self.global_mean /= dist.get_world_size()
            self.global_std /= dist.get_world_size()

    def forward(self, content_feats: torch.Tensor, style_feats: torch.Tensor,
                content_batch_indices: torch.Tensor, style_batch_indices: torch.Tensor) -> torch.Tensor:
        """
        执行自适应实例归一化 (AdaIN)。
        
        参数:
            content_feats (torch.Tensor): 内容特征，形状为 (N_content, C)。
            style_feats (torch.Tensor): 风格特征，形状为 (N_style, C)。
            content_batch_indices (torch.Tensor): 内容特征的 batch 索引，形状为 (N_content,)。
            style_batch_indices (torch.Tensor): 风格特征的 batch 索引，形状为 (N_style,)。
        
        返回:
            normalized_feats (torch.Tensor): 归一化后的特征，形状与 content_feats 相同。
        """
        normalized_feats = torch.zeros_like(content_feats)
        
        for batch_id in torch.unique(content_batch_indices):
            # 获取当前 batch 的内容和风格特征
            c_batch_feats = content_feats[content_batch_indices == batch_id]
            s_batch_feats = style_feats[style_batch_indices == batch_id]

            # 计算当前 batch 的均值和标准差
            c_mean, c_std = self._compute_mean_std_per_batch(c_batch_feats)
            s_mean, s_std = self._compute_mean_std_per_batch(s_batch_feats)

            # 更新并同步全局统计量（如果适用）
            self.update_and_sync_global_stats(s_mean, s_std)

            # 使用全局统计量进行归一化
            normalized_batch = (self.global_std * (c_batch_feats - c_mean) / c_std) + self.global_mean
            # normalized_batch = (s_std * (c_batch_feats - c_mean) / c_std) + s_mean
            normalized_feats[content_batch_indices == batch_id] = normalized_batch

        return normalized_feats