# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial
from typing import List

import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor, nn

from mmdet3d.models.layers.minkowski_engine_block import (
    IS_MINKOWSKI_ENGINE_AVAILABLE, MinkowskiBasicBlock, MinkowskiBottleneck,
    MinkowskiConvModule)
from mmdet3d.models.layers.sparse_block import (SparseBasicBlock,
                                                SparseBottleneck,
                                                make_sparse_convmodule,
                                                replace_feature)
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.models.layers.torchsparse_block import (TorchSparseBasicBlock,
                                                     TorchSparseBottleneck,
                                                     TorchSparseConvModule)
from mmdet3d.utils import OptMultiConfig

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse


from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
import pickle, numpy as np

# class AdaIN:
#     """
#     Adaptive Instance Normalization adapted for point cloud style transfer.
#     """

#     def _compute_mean_std(
#         self, feats: torch.Tensor, eps=1e-8
#     ) -> torch.Tensor:
#         """
#         Compute the mean and standard deviation of the features.

#         Args:
#             feats (torch.Tensor): Input feature tensor with shape (N, C).

#         Returns:
#             tuple: A tuple containing the mean and standard deviation tensors.
#         """
#         assert (
#             len(feats.shape) == 2
#         ), "feature map should be 2-dimensional of the form N,C!"
        
#         # Calculate mean and std along the points dimension (dim=0)
#         mean = torch.mean(feats, dim=0, keepdim=True)  # Shape: (1, C)
#         std = torch.std(feats, dim=0, keepdim=True) + eps  # Shape: (1, C)

#         return mean, std

#     def __call__(
#         self,
#         content_feats: torch.Tensor,
#         style_feats: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Apply Adaptive Instance Normalization to the content features using style features.

#         Args:
#             content_feats (torch.Tensor): Content features with shape (N, C).
#             style_feats (torch.Tensor): Style features with shape (M, C).

#         Returns:
#             torch.Tensor: Normalized content features with shape (N, C).
#         """
#         c_mean, c_std = self._compute_mean_std(content_feats)
#         s_mean, s_std = self._compute_mean_std(style_feats)

#         normalized = (s_std * (content_feats - c_mean) / c_std) + s_mean

#         return normalized

class AdaIN:
    """
    Adaptive Instance Normalization adapted for point cloud style transfer.
    """

    def _compute_mean_std_per_batch(
        self, feats: torch.Tensor, batch_indices: torch.Tensor, eps=1e-8
    ) -> torch.Tensor:
        """
        Compute the mean and standard deviation of the features per batch.

        Args:
            feats (torch.Tensor): Input feature tensor with shape (N, C).
            batch_indices (torch.Tensor): Batch indices tensor with shape (N,) indicating which batch each point belongs to.

        Returns:
            tuple: A tuple containing two tensors, each of shape (B, C) where B is the number of batches,
                   representing the mean and standard deviation per batch.
        """
        unique_batches = torch.unique(batch_indices)
        means, stds = [], []
        
        for batch_id in unique_batches:
            batch_feats = feats[batch_indices == batch_id]
            mean = torch.mean(batch_feats, dim=0, keepdim=True)  # Shape: (1, C)
            std = torch.std(batch_feats, dim=0, keepdim=True) + eps  # Shape: (1, C)
            means.append(mean)
            stds.append(std)

        return torch.cat(means), torch.cat(stds)

    def __call__(
        self,
        content_feats: torch.Tensor,
        style_feats: torch.Tensor,
        content_batch_indices: torch.Tensor,
        style_batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Adaptive Instance Normalization to the content features using style features, considering batch information.

        Args:
            content_feats (torch.Tensor): Content features with shape (N, C).
            style_feats (torch.Tensor): Style features with shape (M, C).
            content_batch_indices (torch.Tensor): Batch indices tensor with shape (N,) for content points.
            style_batch_indices (torch.Tensor): Batch indices tensor with shape (M,) for style points.

        Returns:
            torch.Tensor: Normalized content features with shape (N, C).
        """
        unique_batches = torch.unique(content_batch_indices)
        normalized_feats = torch.zeros_like(content_feats)
        
        for batch_id in unique_batches:
            # 获取当前批次的内容和风格特征
            c_batch_feats = content_feats[content_batch_indices == batch_id]
            s_batch_feats = style_feats[style_batch_indices == batch_id]

            # 计算当前批次的内容和风格特征的均值和标准差
            c_mean, c_std = self._compute_mean_std_per_batch(c_batch_feats, torch.zeros(c_batch_feats.shape[0], dtype=torch.long))
            s_mean, s_std = self._compute_mean_std_per_batch(s_batch_feats, torch.zeros(s_batch_feats.shape[0], dtype=torch.long))

            # 应用AdaIN
            normalized_batch = (s_std * (c_batch_feats - c_mean) / c_std) + s_mean
            
            # 将结果放回normalized_feats中相应的位置
            normalized_feats[content_batch_indices == batch_id] = normalized_batch
        
        return normalized_feats

@MODELS.register_module()
class MinkUNetBackbone_directAdaIN(BaseModule):
    r"""MinkUNet backbone with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        in_channels (int): Number of input voxel feature channels.
            Defaults to 4.
        base_channels (int): The input channels for first encoder layer.
            Defaults to 32.
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        encoder_channels (List[int]): Convolutional channels of each encode
            layer. Defaults to [32, 64, 128, 256].
        encoder_blocks (List[int]): Number of blocks in each encode layer.
        decoder_channels (List[int]): Convolutional channels of each decode
            layer. Defaults to [256, 128, 96, 96].
        decoder_blocks (List[int]): Number of blocks in each decode layer.
        block_type (str): Type of block in encoder and decoder.
        sparseconv_backend (str): Sparse convolutional backend.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 num_stages: int = 4,
                 encoder_channels: List[int] = [32, 64, 128, 256],
                 encoder_blocks: List[int] = [2, 2, 2, 2],
                 decoder_channels: List[int] = [256, 128, 96, 96],
                 decoder_blocks: List[int] = [2, 2, 2, 2],
                 block_type: str = 'basic',
                 sparseconv_backend: str = 'torchsparse',
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        assert sparseconv_backend in [
            'torchsparse', 'spconv', 'minkowski'
        ], f'sparseconv backend: {sparseconv_backend} not supported.'
        self.num_stages = num_stages
        self.sparseconv_backend = sparseconv_backend
        self.use_target = True
        if sparseconv_backend == 'torchsparse':
            assert IS_TORCHSPARSE_AVAILABLE, \
                'Please follow `get_started.md` to install Torchsparse.`'
            input_conv = TorchSparseConvModule
            encoder_conv = TorchSparseConvModule
            decoder_conv = TorchSparseConvModule
            residual_block = TorchSparseBasicBlock if block_type == 'basic' \
                else TorchSparseBottleneck
            # for torchsparse, residual branch will be implemented internally
            residual_branch = None
        elif sparseconv_backend == 'spconv':
            if not IS_SPCONV2_AVAILABLE:
                warnings.warn('Spconv 2.x is not available,'
                              'turn to use spconv 1.x in mmcv.')
            input_conv = partial(
                make_sparse_convmodule, conv_type='SubMConv3d')
            encoder_conv = partial(
                make_sparse_convmodule, conv_type='SparseConv3d')
            decoder_conv = partial(
                make_sparse_convmodule, conv_type='SparseInverseConv3d')
            residual_block = SparseBasicBlock if block_type == 'basic' \
                else SparseBottleneck
            residual_branch = partial(
                make_sparse_convmodule,
                conv_type='SubMConv3d',
                order=('conv', 'norm'))
        elif sparseconv_backend == 'minkowski':
            assert IS_MINKOWSKI_ENGINE_AVAILABLE, \
                'Please follow `get_started.md` to install Minkowski Engine.`'
            input_conv = MinkowskiConvModule
            encoder_conv = MinkowskiConvModule
            decoder_conv = partial(
                MinkowskiConvModule,
                conv_cfg=dict(type='MinkowskiConvNdTranspose'))
            residual_block = MinkowskiBasicBlock if block_type == 'basic' \
                else MinkowskiBottleneck
            residual_branch = partial(MinkowskiConvModule, act_cfg=None)

        self.conv_input = nn.Sequential(
            input_conv(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                indice_key='subm0'),
            input_conv(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                indice_key='subm0'))

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encoder_channels.insert(0, base_channels)
        decoder_channels.insert(0, encoder_channels[-1])

        for i in range(num_stages):
            encoder_layer = [
                encoder_conv(
                    encoder_channels[i],
                    encoder_channels[i],
                    kernel_size=2,
                    stride=2,
                    indice_key=f'spconv{i+1}')
            ]
            for j in range(encoder_blocks[i]):
                if j == 0 and encoder_channels[i] != encoder_channels[i + 1]:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i],
                            encoder_channels[i + 1],
                            downsample=residual_branch(
                                encoder_channels[i],
                                encoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{i+1}'))
                else:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i + 1],
                            encoder_channels[i + 1],
                            indice_key=f'subm{i+1}'))
            self.encoder.append(nn.Sequential(*encoder_layer))

            decoder_layer = [
                decoder_conv(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                    transposed=True,
                    indice_key=f'spconv{num_stages-i}')
            ]
            for j in range(decoder_blocks[i]):
                if j == 0:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1] + encoder_channels[-2 - i],
                            decoder_channels[i + 1],
                            downsample=residual_branch(
                                decoder_channels[i + 1] +
                                encoder_channels[-2 - i],
                                decoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{num_stages-i-1}'))
                else:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1],
                            decoder_channels[i + 1],
                            indice_key=f'subm{num_stages-i-1}'))
            self.decoder.append(
                nn.ModuleList(
                    [decoder_layer[0],
                     nn.Sequential(*decoder_layer[1:])]))
        
        self.ada_in = AdaIN()
        self.GlobalRotScaleTrans = GlobalRotScaleTrans()
        with open('data/SemanticSTF/semanticstf_infos_train.pkl', 'rb') as f:
            self.data_dicts = pickle.load(f)
        self.data_preprocessor = Det3DDataPreprocessor(voxel=True,
                                            voxel_type='minkunet',
                                            batch_first=False,
                                            max_voxels=80000,
                                            voxel_layer=dict(
                                                max_num_points=-1,
                                                point_cloud_range=[-100, -100, -20, 100, 100, 20],
                                                voxel_size=[0.05, 0.05, 0.05],
                                                max_voxels=(-1, -1)))

    def forward(self, voxel_features: Tensor, coors: Tensor) -> Tensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            Tensor: Backbone features.
        """
        if self.training and self.use_target:
            rain_data = []
            idxs = np.random.choice(len(self.data_dicts['data_list']), size=2, replace=False)
            for i in range(2):
                idx = idxs[i]
                points = torch.tensor(np.fromfile('data/SemanticSTF/' + self.data_dicts['data_list'][idx]['lidar_points']['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :4])
                points[:, :3] = self.GlobalRotScaleTrans.transform(points[:, :3])
                rain_data.append(points.to(voxel_features.device))
            for data in rain_data:
                data[:, 3] = data[:, 3] / 255.0
        
            result = self.data_preprocessor.voxelize(rain_data, None)
            rain_voxel_features = result['voxels']
            rain_coors = result['coors']

            # 对晴朗天气和恶劣天气的特征使用AdaIN进行风格转换
            content_batch_indices = coors[:, 3]
            style_batch_indices = rain_coors[:, 3]
            voxel_features = self.ada_in(voxel_features, rain_voxel_features, content_batch_indices, style_batch_indices)
            # voxel_features = torch.add(voxel_features, self.ada_in(voxel_features, rain_voxel_features, content_batch_indices, style_batch_indices))

        if self.sparseconv_backend == 'torchsparse':
            x = torchsparse.SparseTensor(voxel_features, coors)
        elif self.sparseconv_backend == 'spconv':
            spatial_shape = coors.max(0)[0][1:] + 1
            batch_size = int(coors[-1, 0]) + 1
            x = SparseConvTensor(voxel_features, coors, spatial_shape,
                                 batch_size)

        x = self.conv_input(x)
        laterals = [x]
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        decoder_outs = []
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer[0](x)

            lateral_concat = laterals[i]

            if self.sparseconv_backend == 'torchsparse':
                x = torchsparse.cat((x, lateral_concat))
            elif self.sparseconv_backend == 'spconv':
                x = replace_feature(
                    x, torch.cat((x.features, lateral_concat.features), dim=1))

            x = decoder_layer[1](x)
            decoder_outs.append(x)

        if self.sparseconv_backend == 'spconv':
            return decoder_outs[-1].features
        else:
            return decoder_outs[-1].F


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