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

if IS_MINKOWSKI_ENGINE_AVAILABLE:
    import MinkowskiEngine as ME

from torchsparse.nn import BatchNorm, ReLU, LeakyReLU

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        # self.device = device  # 设备信息
        # 定义一个小型网络来计算注意力权重
        self.attention_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )#.to(self.device)  # 确保注意力层也在同一设备上
    
    @staticmethod
    def knn(x, y, k=1):
        """Find k-nearest neighbors for each point in x among points in y."""
        pairwise_distance = torch.cdist(x, y)
        idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)
        return idx.squeeze(-1)  # squeeze to remove the last dimension since k=1
    
    def align_features(self, clear_feature, rain_feature):
        if clear_feature.shape[0] == rain_feature.shape[0]:
            return rain_feature
        
        indices = self.knn(clear_feature.unsqueeze(0), rain_feature.unsqueeze(0)).squeeze(0)
        aligned_rain_feature = rain_feature[indices]
        
        return aligned_rain_feature
    
    def forward(self, clear_feature, rain_feature):
        # # 确保输入特征在正确的设备上
        # clear_feature = clear_feature.to(self.device)
        # rain_feature = rain_feature.to(self.device)
        
        # 对齐特征尺寸
        rain_feature_aligned = self.align_features(clear_feature, rain_feature)
        
        combined_feature = torch.cat([clear_feature, rain_feature_aligned], dim=-1)
        attention_weights = self.attention_layer(combined_feature)
        fused_feature = attention_weights * clear_feature + (1 - attention_weights) * rain_feature_aligned
        return fused_feature

@MODELS.register_module()
class WeatherGenerator(BaseModule):
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
                 use_target = False,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        assert sparseconv_backend in [
            'torchsparse', 'spconv', 'minkowski'
        ], f'sparseconv backend: {sparseconv_backend} not supported.'
        self.num_stages = num_stages
        self.sparseconv_backend = sparseconv_backend
        self.use_target = use_target
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
                indice_key='subm0'),
                BatchNorm(base_channels),
                LeakyReLU(negative_slope=0.01, inplace=True)  # 使用LeakyReLU代替ReLU
        )

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
                    encoder_layer.append(BatchNorm(encoder_channels[i + 1]))  # 添加BatchNorm
                    encoder_layer.append(LeakyReLU(negative_slope=0.01, inplace=True))  # 使用LeakyReLU代替ReLU
                else:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i + 1],
                            encoder_channels[i + 1],
                            indice_key=f'subm{i+1}'))
                    encoder_layer.append(BatchNorm(encoder_channels[i + 1]))  # 添加BatchNorm
                    encoder_layer.append(LeakyReLU(negative_slope=0.01, inplace=True))  # 使用LeakyReLU代替ReLU
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
                    decoder_layer.append(BatchNorm(decoder_channels[i + 1]))  # 添加BatchNorm
                    decoder_layer.append(LeakyReLU(negative_slope=0.01, inplace=True))  # 使用LeakyReLU代替ReLU
                else:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1],
                            decoder_channels[i + 1],
                            indice_key=f'subm{num_stages-i-1}'))
                    decoder_layer.append(BatchNorm(decoder_channels[i + 1]))  # 添加BatchNorm
                    decoder_layer.append(LeakyReLU(negative_slope=0.01, inplace=True))  # 使用LeakyReLU代替ReLU
            self.decoder.append(
                nn.ModuleList(
                    [decoder_layer[0],
                     nn.Sequential(*decoder_layer[1:])]))
            
        # if self.training and self.use_target:
        self.idx = 0
        self.attention_fusion = AttentionFusion(256)
        rain_encoder_channels = [32, 64, 128, 256]
        rain_encoder_blocks = [2, 2, 2, 2]
        self.rain_conv_input = nn.Sequential(
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
                indice_key='subm0'),
                BatchNorm(base_channels),
                LeakyReLU(negative_slope=0.01, inplace=True)  # 使用LeakyReLU代替ReLU
        )

        self.rain_encoder = nn.ModuleList()
        rain_encoder_channels.insert(0, base_channels)

        for i in range(num_stages):
            encoder_layer = [
                encoder_conv(
                    rain_encoder_channels[i],
                    rain_encoder_channels[i],
                    kernel_size=2,
                    stride=2,
                    indice_key=f'spconv{i+1}')
            ]
            for j in range(rain_encoder_blocks[i]):
                if j == 0 and rain_encoder_channels[i] != rain_encoder_channels[i + 1]:
                    encoder_layer.append(
                        residual_block(
                            rain_encoder_channels[i],
                            rain_encoder_channels[i + 1],
                            downsample=residual_branch(
                                rain_encoder_channels[i],
                                rain_encoder_channels[i + 1],
                                kernel_size=1)
                            if residual_branch is not None else None,
                            indice_key=f'subm{i+1}'))
                    encoder_layer.append(BatchNorm(encoder_channels[i + 1]))  # 添加BatchNorm
                    encoder_layer.append(LeakyReLU(negative_slope=0.01, inplace=True))  # 使用LeakyReLU代替ReLU
                else:
                    encoder_layer.append(
                        residual_block(
                            rain_encoder_channels[i + 1],
                            rain_encoder_channels[i + 1],
                            indice_key=f'subm{i+1}'))
                    encoder_layer.append(BatchNorm(encoder_channels[i + 1]))  # 添加BatchNorm
                    encoder_layer.append(LeakyReLU(negative_slope=0.01, inplace=True))  # 使用LeakyReLU代替ReLU
            self.rain_encoder.append(nn.Sequential(*encoder_layer))

        self.FeatureToPointcloudDecoder = FeatureToPointcloudDecoder(96, 4)
        
    def forward(self, voxel_features: Tensor, coors: Tensor) -> Tensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            Tensor: Backbone features.
        """
        if self.sparseconv_backend == 'torchsparse':
            x = torchsparse.SparseTensor(voxel_features, coors)
        elif self.sparseconv_backend == 'spconv':
            spatial_shape = coors.max(0)[0][1:] + 1
            batch_size = int(coors[-1, 0]) + 1
            x = SparseConvTensor(voxel_features, coors, spatial_shape,
                                 batch_size)
        elif self.sparseconv_backend == 'minkowski':
            x = ME.SparseTensor(voxel_features, coors)

        x = self.conv_input(x)
        laterals = [x]
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        if self.training and self.use_target:
            from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
            import pickle, numpy as np

            with open('data/SemanticSTF/semanticstf_infos_train_rain.pkl', 'rb') as f:
                data_dicts = pickle.load(f)
            # idx = np.random.randint(0, len(data_dicts['data_list']), size=2)
            rain_data = []
            rain_data.append(torch.tensor(np.fromfile('data/SemanticSTF/' + data_dicts['data_list'][self.idx]['lidar_points']['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :4]).to(voxel_features.device))
            self.idx = (self.idx + 1) % len(data_dicts['data_list'])
            rain_data.append(torch.tensor(np.fromfile('data/SemanticSTF/' + data_dicts['data_list'][self.idx]['lidar_points']['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :4]).to(voxel_features.device))
            self.idx = (self.idx + 1) % len(data_dicts['data_list'])
            for data in rain_data:
                data[:, 3] = data[:, 3] / 255.0

            preprocessor = Det3DDataPreprocessor(voxel=True,
                                                voxel_type='minkunet',
                                                batch_first=False,
                                                max_voxels=80000,
                                                voxel_layer=dict(
                                                    max_num_points=-1,
                                                    point_cloud_range=[-100, -100, -20, 100, 100, 20],
                                                    voxel_size=[0.05, 0.05, 0.05],
                                                    max_voxels=(-1, -1)))
        
            result = preprocessor.voxelize(rain_data, None)
            rain_voxel_features = result['voxels']
            rain_coors = result['coors']
        
            if self.sparseconv_backend == 'torchsparse':
                rain_x = torchsparse.SparseTensor(rain_voxel_features, rain_coors)
            elif self.sparseconv_backend == 'spconv':
                spatial_shape_rain = rain_coors.max(0)[0][1:] + 1
                batch_size_rain = int(rain_coors[-1, 0]) + 1
                rain_x = SparseConvTensor(rain_voxel_features, rain_coors, spatial_shape_rain, batch_size_rain)
            elif self.sparseconv_backend == 'minkowski':
                rain_x = ME.SparseTensor(rain_voxel_features, rain_coors)

            # 使用专门为恶劣天气设计的编码器
            rain_x = self.rain_conv_input(rain_x)  # 假设已定义了rain_conv_input
            rain_laterals = [rain_x]
            for rain_encoder_layer in self.rain_encoder:  # 假设已定义了rain_encoder
                rain_x = rain_encoder_layer(rain_x)
                rain_laterals.append(rain_x)
            rain_laterals = rain_laterals[:-1][::-1]

            # # 初始化注意力融合模块
            # attention_fusion = AttentionFusion(x.F.shape[-1], voxel_features.device)

            # # 将晴朗天气和恶劣天气的特征进行注意力机制融合
            # x.F = attention_fusion(x.F, rain_x.F)

            # 将晴朗天气和恶劣天气的特征进行注意力机制融合
            x.F = self.attention_fusion(x.F, rain_x.F)

        decoder_outs = []
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer[0](x)

            # if self.training and self.use_target:
            #     # 初始化注意力融合模块
            #     attention_fusion = AttentionFusion(laterals[i].F.shape[-1], voxel_features.device)
            #     # 对于每个解码器阶段也应用注意力机制
            #     lateral_concat_F = attention_fusion(laterals[i].F, rain_laterals[i].F)
            #     if self.sparseconv_backend == 'torchsparse':
            #         lateral_concat = torchsparse.SparseTensor(lateral_concat_F, laterals[i].C)
            #     elif self.sparseconv_backend == 'spconv':
            #         lateral_concat = SparseConvTensor(lateral_concat_F, laterals[i].C, laterals[i].spatial_shape, laterals[i].batch_size)
            #     elif self.sparseconv_backend == 'minkowski':
            #         lateral_concat = ME.SparseTensor(lateral_concat_F, laterals[i].C)
            # else:
            #     lateral_concat = laterals[i]

            lateral_concat = laterals[i]

            if self.sparseconv_backend == 'torchsparse':
                x = torchsparse.cat((x, lateral_concat))
            elif self.sparseconv_backend == 'spconv':
                x = replace_feature(
                    x, torch.cat((x.features, lateral_concat.features), dim=1))
            elif self.sparseconv_backend == 'minkowski':
                x = ME.cat(x, lateral_concat)

            x = decoder_layer[1](x)
            decoder_outs.append(x)

        if self.sparseconv_backend == 'spconv':
            return decoder_outs[-1].features
        else:
            # return decoder_outs[-1].F
            generated_voxel_features = self.FeatureToPointcloudDecoder(decoder_outs[-1].F)
            if self.use_target:
                return generated_voxel_features, coors, rain_voxel_features, rain_coors
            return generated_voxel_features


import torch.nn.functional as F

class FeatureToPointcloudDecoder(nn.Module):
    def __init__(self, input_dim, output_channels=4):
        super(FeatureToPointcloudDecoder, self).__init__()
        # 假设input_dim是特征维度大小，output_channels是点云每个点的属性数量
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, output_channels)
        )
        
        # 添加额外的层来分别处理xyz和i通道
        self.last_layer_xyz = nn.Linear(output_channels - 1, output_channels - 1)  # 线性激活
        self.last_layer_i = nn.Linear(1, 1)  # Sigmoid激活用于i值

    def forward(self, x):
        x = self.fc(x)
        
        # 分离xyz和i通道
        xyz = self.last_layer_xyz(x[:, :-1])
        i = self.last_layer_i(x[:, -1:])
        i = torch.sigmoid(i)  # 使用sigmoid激活函数确保i值在(0, 1)范围内
        
        # 合并xyz和i通道
        output = torch.cat([xyz, i], dim=1)
        return output