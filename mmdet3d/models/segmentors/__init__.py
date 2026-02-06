# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DSegmentor
from .cylinder3d import Cylinder3D
from .encoder_decoder import EncoderDecoder3D
from .lasermix import LaserMix
from .minkunet import MinkUNet
from .seg3d_tta import Seg3DTTAModel
from .semi_base import SemiBase3DSegmentor

### new added
from .minkunet_weather_dropper import MinkUNetWeatherDropper
from .GANModel import GANModel
from .minkunet_weather_dropper_useGAN import MinkUNetWeatherDropper_useGAN
from .minkunet_weather_dropper_withGAN import MinkUNetWeatherDropper_withGAN
from .minkunet_weather_dropper_AdaIN import MinkUNetWeatherDropper_AdaIN
from .minkunet_weather_dropper_direct import MinkUNetWeatherDropper_direct
from .minkunet_weather_dropper_direct_segonly import MinkUNetWeatherDropper_direct_segonly
from .AdaIN import AdaIN
from .CrossAttention import CrossAttention
from .minkunet_weather_dropper_D1 import MinkUNetWeatherDropper_D1
from .minkunet_weather_dropper_KT import MinkUNetWeatherDropper_KT
from .minkunet_weather_dropper_CL import MinkUNetWeatherDropper_CL

__all__ = [
    'Base3DSegmentor', 'EncoderDecoder3D', 'Cylinder3D', 'MinkUNet',
    'Seg3DTTAModel', 'SemiBase3DSegmentor', 'LaserMix',
    ### new added
    'MinkUNetWeatherDropper',
    'GANModel',
    'MinkUNetWeatherDropper_useGAN',
    'MinkUNetWeatherDropper_withGAN',
    'MinkUNetWeatherDropper_AdaIN',
    'MinkUNetWeatherDropper_direct',
    'AdaIN', 'CrossAttention',
    'MinkUNetWeatherDropper_direct_segonly',
    'MinkUNetWeatherDropper_D1',
    'MinkUNetWeatherDropper_KT',
    'MinkUNetWeatherDropper_CL'
]