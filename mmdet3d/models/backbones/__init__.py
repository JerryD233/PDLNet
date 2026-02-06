# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt

from .cylinder3d import Asymm3DSpconv
from .dgcnn import DGCNNBackbone
from .dla import DLANet
from .mink_resnet import MinkResNet
from .minkunet_backbone import MinkUNetBackbone
from .multi_backbone import MultiBackbone
from .nostem_regnet import NoStemRegNet
from .pointnet2_sa_msg import PointNet2SAMSG
from .pointnet2_sa_ssg import PointNet2SASSG
from .second import SECOND
from .spvcnn_backone import MinkUNetBackboneV2, SPVCNNBackbone
from .generator import WeatherGenerator
from .discriminator1 import Discriminator1
from .generator_AdaIN import WeatherGenerator_AdaIN
from .minkunet_backbone_AdaIN import MinkUNetBackbone_AdaIN
from .minkunet_backbone_directAdaIN import MinkUNetBackbone_directAdaIN
from .generator_crossAttention import WeatherGenerator_crossAttention
from .minkunet_backbone_directCrossAttention import MinkUNetBackbone_directCrossAttention
from .minkunet_backbone_directAdaIN_crossAttention import MinkUNetBackbone_directAdaIN_crossAttention
from .minkunet_backbone_directAdaIN_crossAttention_D1 import MinkUNetBackbone_directAdaIN_crossAttention_D1
from .minkunet_backbone_teacher import MinkUNetBackbone_teacher

__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND', 'DGCNNBackbone', 'PointNet2SASSG', 'PointNet2SAMSG',
    'MultiBackbone', 'DLANet', 'MinkResNet', 'Asymm3DSpconv',
    'MinkUNetBackbone', 'SPVCNNBackbone', 'MinkUNetBackboneV2',
    'WeatherGenerator','Discriminator1','WeatherGenerator_AdaIN', 'MinkUNetBackbone_AdaIN', 'MinkUNetBackbone_directAdaIN',
    'WeatherGenerator_crossAttention', 'MinkUNetBackbone_directCrossAttention', 'MinkUNetBackbone_directAdaIN_crossAttention',
    'MinkUNetBackbone_directAdaIN_crossAttention_D1',
    'MinkUNetBackbone_teacher'
]
