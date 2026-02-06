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
   

@MODELS.register_module()
class MinkUNetWeatherDropper_AdaIN(EncoderDecoder3D):
    r"""MinkUNetWeatherDropper is the implementation of `4D Spatio-Temporal ConvNets.
    <https://arxiv.org/abs/1904.08755>`_ with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`EncoderDecoder3D`.
    """

    def __init__(self,
                #  GANModel: ConfigType,
                generator_AdaIN: ConfigType,
                discriminator1: ConfigType,
                #  gan_model_path: str,
                #  n_observations=4,
                #  n_actions=1,
                 **kwargs) -> None:
        super(MinkUNetWeatherDropper_AdaIN, self).__init__(**kwargs)
        self.generator_AdaIN = MODELS.build(generator_AdaIN)
        self.discriminator1 = MODELS.build(discriminator1)
        # self.GANModel = self.load_gan_model(gan_model_path, GANModel)

        # self.GANModel.eval()

        # # 冻结GAN模型参数
        # for param in self.GANModel.generator.parameters():
        #     param.requires_grad = False


    def load_gan_model(self, model_path, GANModel):
        """加载GAN模型"""
        gan_model =  MODELS.build(GANModel) # 根据实际情况初始化你的GAN模型
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            gan_model.load_state_dict(checkpoint['state_dict'])
        else:
            gan_model.load_state_dict(checkpoint)
        return gan_model


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

        voxel_dict = self.extract_feat(batch_inputs_dict)
        loss_decode = self._decode_head_forward_train(voxel_dict,
                                                    batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)

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
    
    def train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapperDict) -> Dict[str, torch.Tensor]:
        """Define the training step to update both generator and discriminator separately."""
        """
        Example::

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()
        """
        # 确保所有张量都在同一设备上
        # device = next(self.parameters()).device
                
        # # 更新判别器
        # optimizer_discriminator1 = optim_wrapper['discriminator1']
        # data = self.data_preprocessor(data, True)
        # data = data['inputs']
        
        # with optimizer_discriminator1.optim_context(self):
        #     # 提取特征并生成恶劣天气点云
        #     generated_voxel_features, coors, rain_voxel_features, rain_coors = self.extract_feat(data)

        #     # 判别器对真实和生成的数据进行评分
        #     loss_d1 = compute_discriminator1_loss(generated_voxel_features, coors, rain_voxel_features, rain_coors, self.discriminator1)

        #     # 计算判别器的损失
            
        # optimizer_discriminator1.update_params(loss_d1)

        # # 更新生成器
        # optimizer_generator = optim_wrapper['generator_AdaIN']
        # with optimizer_generator.optim_context(self):
        #     loss_g = compute_adv_shallow_loss(generated_voxel_features, coors, self.discriminator1)
        # optimizer_generator.update_params(loss_g)

        # # 更新分割网络
        # optimizer_segbackbone = optim_wrapper['segbackbone']
        # with optimizer_segbackbone.optim_context(self):
        #     data = self.data_preprocessor(data, True)
            
        #     # 更新判别器
        #     optimizer_discriminator1 = optim_wrapper['discriminator1']            
        #     with optimizer_discriminator1.optim_context(self):
        #         voxel_dict = data['inputs']['voxels'].copy()
        #         # 提取特征并生成恶劣天气风格点云
        #         generated_voxel_features, coors, rain_voxel_features, rain_coors = self.generator_AdaIN(voxel_dict['voxels'], voxel_dict['coors'])
        #         # 计算d1的损失
        #         loss_d1 = compute_discriminator1_loss(generated_voxel_features, coors, rain_voxel_features, rain_coors, self.discriminator1)
        #         optimizer_discriminator1.update_params(loss_d1)

        #     # 更新生成器
        #     optimizer_generator_AdaIN = optim_wrapper['generator_AdaIN']
        #     with optimizer_generator_AdaIN.optim_context(self):
        #         loss_g = compute_adv_shallow_loss(generated_voxel_features, coors, self.discriminator1)

        #         data['inputs']['voxels']['voxels'] = generated_voxel_features
        #         loss_seg = self._run_forward(data, mode='loss')  # type: ignore
        #         loss_seg.update({'loss_g': loss_g})
        #         parsed_losses, _ = self.parse_losses(loss_seg)
        #         optimizer_generator_AdaIN.update_params(parsed_losses)
        #     optimizer_segbackbone.update_params(parsed_losses)
        data = self.data_preprocessor(data, True)
        # 更新判别器
        optimizer_discriminator1 = optim_wrapper['discriminator1']
        with optimizer_discriminator1.optim_context(self):
            voxel_dict = data['inputs']['voxels'].copy()
            # 提取特征并生成恶劣天气风格点云
            generated_voxel_features, coors, rain_voxel_features, rain_coors = self.generator_AdaIN(voxel_dict['voxels'], voxel_dict['coors'])
            # 计算d1的损失
            loss_d1 = compute_discriminator1_loss(generated_voxel_features, coors, rain_voxel_features, rain_coors, self.discriminator1)
            
            optimizer_discriminator1.update_params(loss_d1)

        # 更新分割网络的数据
        data['inputs']['voxels']['voxels'] = generated_voxel_features

        # 更新生成器
        optimizer_generator_AdaIN = optim_wrapper['generator_AdaIN']
        with optimizer_generator_AdaIN.optim_context(self):
            loss_g = compute_adv_shallow_loss(generated_voxel_features, coors, self.discriminator1)

            loss_seg = self._run_forward(data, mode='loss')
            loss_seg.update({'loss_g': loss_g})

            # 解析损失并获取总损失
            parse_loss, log_vars = self.parse_losses(loss_seg)

            # # 只用loss_g
            # parse_loss, log_vars = self.parse_losses({'loss_g': loss_g})

            # 使用总损失进行一步反向传播
            optimizer_generator_AdaIN.update_params(parse_loss, need=True)
            # optimizer_generator_AdaIN.update_params(parse_loss)

        # 更新分割网络
        optimizer_segbackbone = optim_wrapper['segbackbone']
        with optimizer_segbackbone.optim_context(self):

            # loss_seg = self._run_forward(data, mode='loss')
            # loss_seg.update({'loss_g': loss_g})

            # # 解析损失并获取总损失
            # parse_loss, log_vars = self.parse_losses(loss_seg)
            
            # 使用相同的总损失进行分割网络的参数更新
            optimizer_segbackbone.update_params(parse_loss)

            # 返回损失值用于记录或展示
        _, log_vars = self.parse_losses({'loss_d1': loss_d1, 'loss_g': loss_g, 'loss_seg': loss_seg['decode.loss_ce']})
        return log_vars



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