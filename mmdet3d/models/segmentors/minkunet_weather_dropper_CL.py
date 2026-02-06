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

from mmengine.runner.checkpoint import (_load_checkpoint, _load_checkpoint_to_model)

from mmengine.optim import OptimWrapper
from typing import Dict, Optional, Tuple, Union

# def find_model_path(directory):
#     # 列出目录中所有的文件
#     files = os.listdir(directory)
    
#     matched_files = []

#     while len(matched_files) < 1:
#         # 筛选出符合条件的文件
#         matched_files = [f for f in files if f.startswith('best_miou_epoch_') and f.endswith('.pth')]
    
#     # 确保只有一个匹配项
#     if len(matched_files) != 1:
#         raise ValueError(f"Expected exactly one file starting with 'best_miou_epoch_', but found {len(matched_files)}")
    
#     # 构造完整路径并返回
#     model_path = os.path.join(directory, matched_files[0])
#     return model_path

import time
def find_model_path(directory, max_retries=10, retry_delay=1):
    """
    查找最新的 'best_miou_epoch_*.pth' 文件。
    
    :param directory: 搜索目录
    :param max_retries: 最大重试次数
    :param retry_delay: 每次重试前的延迟（秒）
    :return: 匹配的模型路径
    """
    for attempt in range(max_retries):
        # 列出目录中所有的文件
        files = os.listdir(directory)
        
        # 筛选出符合条件的文件
        matched_files = [f for f in files if f.startswith('best_miou_epoch_') and f.endswith('.pth')]
        
        # # 如果找到多个文件，按修改时间排序，选择最新的
        # if len(matched_files) > 1:
        #     matched_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
        
        # 如果找到了至少一个文件
        if len(matched_files) >= 1:
            latest_file = matched_files[0]
            file_path = os.path.join(directory, latest_file)
            
            # 检查文件是否存在且大小不为零
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return file_path
        
        # 如果没有找到匹配文件，等待一段时间再重试
        print(f"Attempt {attempt + 1}/{max_retries}: No matching file found or file is being updated. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
    
    raise ValueError("Failed to find a valid 'best_miou_epoch_*.pth' file after multiple attempts.")

@MODELS.register_module()
class MinkUNetWeatherDropper_CL(EncoderDecoder3D):
    r"""MinkUNetWeatherDropper is the implementation of `4D Spatio-Temporal ConvNets.
    <https://arxiv.org/abs/1904.08755>`_ with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`EncoderDecoder3D`.
    """

    def __init__(self,
                teacher_model_config: ConfigType,
                 **kwargs) -> None:
        super(MinkUNetWeatherDropper_CL, self).__init__(**kwargs)
        self.epoch = 1
        self.count = 1
        self.count_test = 1
        self.current_threshold = 0.8
        # self.current_threshold = 0
        # self.delta_tau = 0.2
        self.delta_tau = 0.1
        self.model_path = None
        self.teacher_model_config = teacher_model_config
        self.load_teacher_model('work_dirs/kitti2stf_KT_epoch45/best_miou_epoch_38.pth', teacher_model_config)

        checkpoint = torch.load('work_dirs/kitti2stf_KT_epoch45/best_miou_epoch_38.pth', map_location='cpu')
        if 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.load_state_dict(checkpoint)


    def load_teacher_model(self, model_path, teacher_model_config):
        """加载teacher模型"""
        teacher_model =  MODELS.build(teacher_model_config) # 根据实际情况初始化你的teacher模型
        # checkpoint = _load_checkpoint(model_path, map_location='cpu')
        # checkpoint = _load_checkpoint_to_model(
        #     teacher_model, checkpoint, strict=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            teacher_model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            teacher_model.load_state_dict(checkpoint)
        
        self.teacher_model = teacher_model

        # 冻结teacher_model模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False

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
        losses = dict()
        # if self.epoch <= 10:
        #     lam = 10
        # # elif self.epoch <=10:
        # #     lam = 10
        # # elif self.flag:
        # #     lam = 10
        # #     self.flag = False
        # else:
        #     lam = 1
        # print(lam)
        bt = batch_data_samples.copy()
        if self.epoch > self.count:
            if self.epoch > 20 and self.epoch % 10 == 0:
                self.current_threshold = update_confidence_threshold(self.current_threshold, self.delta_tau)
        #     self.model_path = find_model_path('work_dirs/kitti2stf_CL_epoch30_t1/')
        #     print('find teacher_model:', self.model_path)
        #     self.load_teacher_model(self.model_path, self.teacher_model_config)
        #     self.teacher_model.to(batch_inputs_dict['voxels']['voxels'].device)
            self.count = self.epoch

        # if self.teacher_model.training:
        #     self.teacher_model.eval()
        # with torch.no_grad():
        if self.epoch > 9:
            for i in range(len(bt)):
                file_path = 'data/pseudo_labels/SemanticSTF/train/epoch_' + str(self.epoch // 10 * 10) + '/' + batch_data_samples[i].lidar_path.split('/')[-1].split('.')[0] + '.label'
                pred_t = load_pseudo_labels_from_binary(file_path, batch_inputs_dict['voxels']['voxels'].device)
                bt[i].gt_pts_seg.pts_semantic_mask = pred_t
        else:
            for i in range(len(bt)):
                file_path = 'data/pseudo_labels/SemanticSTF/train/epoch_1/' + batch_data_samples[i].lidar_path.split('/')[-1].split('.')[0] + '.label'
                pred_t = load_pseudo_labels_from_binary(file_path, batch_inputs_dict['voxels']['voxels'].device)
                bt[i].gt_pts_seg.pts_semantic_mask = pred_t
        # else:
        #     result = self.teacher_model.data_preprocessor.all_voxelize(batch_inputs_dict['points'], None)
        #     results = {'voxels': result}
        #     pred_t = self.teacher_model.teacher_predict(results, bt, self.current_threshold)
        #     # print(len(pred_t))
        #     # print('use teacher_model:', self.model_path)
        #     for i in range(len(pred_t)):
        #         bt[i].gt_pts_seg.pts_semantic_mask = pred_t[i].pred_pts_seg.pts_semantic_mask

        voxel_dict = self.extract_feat(batch_inputs_dict)
        loss_decode = self._decode_head_forward_train(voxel_dict,
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
                confidence_threshold: float = 0.9,
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
        return self.postprocess_teacher_result(seg_logits_list, batch_data_samples, confidence_threshold)

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
    
    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        pred_t = None
        data = self.data_preprocessor(data, False)
        # return self._run_forward(data, mode='predict')  # type: ignore
        if self.epoch % 10 == 0:
            if self.epoch > self.count_test:
                self.model_path = find_model_path('work_dirs/kitti2stf_CL_epoch300_best/')
                print('find teacher_model:', self.model_path)
                self.load_teacher_model(self.model_path, self.teacher_model_config)
                self.teacher_model.to(data['inputs']['voxels']['voxels'].device)
                self.count_test = self.epoch
            batch_inputs_dict = data['inputs'].copy()
            bt = data['data_samples'].copy()
            # result = self.teacher_model.data_preprocessor.voxelize(batch_inputs_dict['points'], None)
            # results = {'voxels': result}
            pred_t = self.teacher_model.teacher_predict(batch_inputs_dict, bt, self.current_threshold)
            # pred_t = self.teacher_predict(batch_inputs_dict, bt, 0)
            # print(len(pred_t))
            for i in range(len(pred_t)):
                # bt[i].gt_pts_seg.pts_semantic_mask = pred_t[i].pred_pts_seg.pts_semantic_mask
                lidar_path = pred_t[i].lidar_path
                pseudo_labels = pred_t[i].pred_pts_seg.pts_semantic_mask
                output_dir = 'data/pseudo_labels/SemanticSTF/train/epoch_' + str(self.epoch // 10 * 10) + '/'
                save_pseudo_labels_as_binary(lidar_path, pseudo_labels, output_dir)
        if pred_t is None:
            return self.teacher_model._run_forward(data, mode='predict')  # type: ignore
        return pred_t
    
    # def val_step(self, data: Union[tuple, dict, list]) -> list:
    #     """Gets the predictions of given data.

    #     Calls ``self.data_preprocessor(data, False)`` and
    #     ``self(inputs, data_sample, mode='predict')`` in order. Return the
    #     predictions which will be passed to evaluator.

    #     Args:
    #         data (dict or tuple or list): Data sampled from dataset.

    #     Returns:
    #         list: The predictions of given data.
    #     """
    #     data = self.data_preprocessor(data, False)
    #     batch_inputs_dict = data['inputs'].copy()
    #     bt = data['data_samples'].copy()
    #     pred_t = self.teacher_predict(batch_inputs_dict, bt, self.current_threshold)
    #     # pred_t = self.teacher_model.teacher_predict(batch_inputs_dict, bt, 0)
    #     # print(len(pred_t))
    #     output_dir = 'data/pseudo_labels/SemanticSTF/train/epoch_' + str(self.epoch) + '/'
    #     for i in range(len(pred_t)):
    #         bt[i].gt_pts_seg.pts_semantic_mask = pred_t[i].pred_pts_seg.pts_semantic_mask
    #         lidar_path = pred_t[i].lidar_path
    #         pseudo_labels = pred_t[i].pred_pts_seg.pts_semantic_mask
    #         save_pseudo_labels_as_binary(lidar_path, pseudo_labels, output_dir)
    #     return self._run_forward(data, mode='predict')  # type: ignore

# 更新置信度阈值的函数
def update_confidence_threshold(current_threshold, delta_tau):
    new_threshold = current_threshold - delta_tau
    if new_threshold < 0:
        new_threshold = 0  # 确保阈值不会小于0
    return new_threshold


def save_pseudo_labels_as_binary(lidar_path, pseudo_labels, output_dir):
    """
    将伪标签保存为 .label 格式的二进制文件。
    
    参数:
        lidar_path (str): 点云文件路径。
        pseudo_labels (torch.Tensor): 点云的伪标签，类型为 torch.int64。
        output_dir (str): 输出目录，默认为 'data/pseudo_labels/SemanticSTF/'。
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 构建输出文件名
    file_name = lidar_path.split('/')[-1].split('.')[0] + '.label'
    output_path = os.path.join(output_dir, file_name)
    
    # 如果伪标签在GPU上，先将其移动到CPU并转为numpy数组
    if pseudo_labels.device.type == 'cuda':
        pseudo_labels_np = pseudo_labels.cpu().numpy()
    else:
        pseudo_labels_np = pseudo_labels.numpy()
    
    # 写入二进制文件
    pseudo_labels_np.astype('int32').tofile(output_path)  # 假定.label文件需要'int32'格式
    
    print(f"Pseudo labels saved to {output_path}")


def load_pseudo_labels_from_binary(file_path, device):
    """
    从 .label 二进制文件加载伪标签。
    
    参数:
        file_path (str): .label 文件的路径。
        device (str): 目标设备，默认为 'cuda:0'。
        
    返回:
        torch.Tensor: 加载的伪标签张量，类型为 torch.int64。
    """
    # 读取二进制文件为 numpy 数组
    pseudo_labels_np = np.fromfile(file_path, dtype=np.int32)
    
    # 将 numpy 数组转换为 torch 张量
    pseudo_labels_tensor = torch.from_numpy(pseudo_labels_np).long()  # long() 等同于 int64
    
    # # 如果需要的话，将张量移动到指定设备
    # if 'cuda' in device and torch.cuda.is_available():
    #     pseudo_labels_tensor = pseudo_labels_tensor.to(device)
    pseudo_labels_tensor = pseudo_labels_tensor.to(device)
    
    # print(f"Pseudo labels loaded from {file_path} to {device}.")
    return pseudo_labels_tensor