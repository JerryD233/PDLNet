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

@MODELS.register_module()
class WeatherDiscriminator(BaseModule):
    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 num_stages: int = 4,
                 encoder_channels: List[int] = [32, 64, 128, 256],
                 encoder_blocks: List[int] = [2, 2, 2, 2],
                 sparseconv_backend: str = 'torchsparse',
                 block_type: str = 'basic',
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(encoder_channels)
        self.num_stages = num_stages
        self.sparseconv_backend = sparseconv_backend

        if sparseconv_backend == 'torchsparse':
            input_conv = TorchSparseConvModule
            encoder_conv = TorchSparseConvModule
            residual_block = TorchSparseBasicBlock if block_type == 'basic' else TorchSparseBottleneck
        elif sparseconv_backend == 'spconv':
            if not IS_SPCONV2_AVAILABLE:
                warnings.warn('Spconv 2.x is not available, turn to use spconv 1.x in mmcv.')
            input_conv = partial(make_sparse_convmodule, conv_type='SubMConv3d')
            encoder_conv = partial(make_sparse_convmodule, conv_type='SparseConv3d')
            residual_block = SparseBasicBlock if block_type == 'basic' else SparseBottleneck
        elif sparseconv_backend == 'minkowski':
            input_conv = MinkowskiConvModule
            encoder_conv = MinkowskiConvModule
            residual_block = MinkowskiBasicBlock if block_type == 'basic' else MinkowskiBottleneck

        # 输入卷积层
        self.conv_input = nn.Sequential(
            input_conv(in_channels, base_channels, kernel_size=3, padding=1, indice_key='subm0'),
            input_conv(base_channels, base_channels, kernel_size=3, padding=1, indice_key='subm0'))

        # 编码器层
        self.encoder = nn.ModuleList()
        encoder_channels.insert(0, base_channels)

        for i in range(num_stages):
            encoder_layer = [
                encoder_conv(encoder_channels[i], encoder_channels[i + 1], kernel_size=2, stride=2, indice_key=f'spconv{i+1}')
            ]
            for j in range(encoder_blocks[i]):
                encoder_layer.append(residual_block(encoder_channels[i + 1], encoder_channels[i + 1], indice_key=f'subm{i+1}'))
            self.encoder.append(nn.Sequential(*encoder_layer))

        # 全连接层，输出单个标量值
        self.fc = nn.Sequential(
                    nn.Linear(encoder_channels[-1], 1),
                    nn.Sigmoid()  # 添加Sigmoid激活函数
                    )

    def forward(self, voxel_features: torch.Tensor, coors: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, D+1), where the last dimension is batch index.

        Returns:
            Tensor: A score for each input sample indicating whether it is real or fake.
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
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        # 使用全局平均池化将特征图转换为固定长度向量
        if self.sparseconv_backend == 'torchsparse':
            global_avg_pool = x.F.mean(dim=0)
        elif self.sparseconv_backend == 'spconv':
            global_avg_pool = x.features.mean(dim=0)
        elif self.sparseconv_backend == 'minkowski':
            global_avg_pool = x.F.mean(dim=0)

        score = self.fc(global_avg_pool)
        return score