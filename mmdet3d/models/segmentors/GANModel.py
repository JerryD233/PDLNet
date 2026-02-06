import torch
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmdet3d.structures.det3d_data_sample import SampleList
from typing import Dict, Optional, Tuple, Union
from torch import Tensor
from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
from .encoder_decoder import EncoderDecoder3D
from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from mmengine.utils import is_list_of
from collections import OrderedDict
import pickle, numpy as np


@MODELS.register_module()
class GANModel(EncoderDecoder3D):
    def __init__(self, 
                 generator: ConfigType, 
                 discriminator: ConfigType,
                 **kwargs):
        super(GANModel, self).__init__(**kwargs)
        # 初始化生成器和判别器
        self.generator = MODELS.build(generator)
        self.discriminator = MODELS.build(discriminator)
        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()
        # self.idx = 0
        # self.GlobalRotScaleTrans = GlobalRotScaleTrans()

        # with open('data/SemanticSTF/semanticstf_infos_train.pkl', 'rb') as f:
        #     self.data_dicts = pickle.load(f)
        # idx = np.random.randint(0, len(data_dicts['data_list']), size=2)
        

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
        generated_voxel_features, coors, rain_voxel_features, rain_coors = self.generator(voxel_dict['voxels'], voxel_dict['coors'])
        
        return generated_voxel_features, coors, rain_voxel_features, rain_coors
        # generated_voxel_features, coors = self.generator(voxel_dict['voxels'], voxel_dict['coors'])
        
        # return generated_voxel_features, coors
    
    def extract_feat_forward(self, batch_inputs_dict: dict) -> dict:
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
        generated_voxel_features = self.generator(voxel_dict['voxels'], voxel_dict['coors'])
        
        return generated_voxel_features
        
    def loss(self, batch_inputs_dict: dict,) -> Dict[str, Tensor]:
        """计算生成器和判别器的损失"""

        # 生成恶劣天气点云
        generated_voxel_features, coors, rain_voxel_features, rain_coors = self.extract_feat(batch_inputs_dict)

        # 判别器对真实和生成的数据进行评分
        real_score = self.discriminator(rain_voxel_features, rain_coors)
        fake_score = self.discriminator(generated_voxel_features.detach(), coors.detach())

        # 计算判别器的损失
        loss_d_real = F.binary_cross_entropy(real_score, torch.ones_like(real_score))
        loss_d_fake = F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))
        loss_d = (loss_d_real + loss_d_fake) / 2

        # 更新生成器时再次评估生成的数据
        fake_score_g = self.discriminator(generated_voxel_features, coors)
        loss_g = F.binary_cross_entropy(fake_score_g, torch.ones_like(fake_score_g))

        return dict(loss_d=loss_d, loss_g=loss_g)
    
    def get_rain_features(self, voxel_features):
        rain_data = []
        idx = np.random.randint(0, len(self.data_dicts['data_list']), size=2)
        for i in range(2):
            self.idx = idx[i]
            points = torch.tensor(np.fromfile('data/SemanticSTF/' + self.data_dicts['data_list'][self.idx]['lidar_points']['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :4])
            points[:, :3] = self.GlobalRotScaleTrans.transform(points[:, :3])
            rain_data.append(points.to(voxel_features.device))
            # self.idx = (self.idx + 1) % len(self.data_dicts['data_list'])
        # rain_data.append(torch.tensor(np.fromfile('data/SemanticSTF/' + self.data_dicts['data_list'][self.idx]['lidar_points']['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :4]).to(voxel_features.device))
        # self.idx = (self.idx + 1) % len(self.data_dicts['data_list'])
        for data in rain_data:
            data[:, 3] = data[:, 3] / 255.0
    
        result = self.data_preprocessor.voxelize(rain_data, None)
        rain_voxel_features = result['voxels']
        rain_coors = result['coors']

        return rain_voxel_features, rain_coors

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
        device = next(self.parameters()).device
        
        # 更新判别器
        optimizer_discriminator = optim_wrapper['discriminator']
        # optimizer_discriminator.zero_grad()
        data = self.data_preprocessor(data, True)
        data = data['inputs']
        
        with optimizer_discriminator.optim_context(self):
            # 提取特征并生成恶劣天气点云
            generated_voxel_features, coors, rain_voxel_features, rain_coors = self.extract_feat(data)
            # generated_voxel_features, coors = self.extract_feat(data)
            # rain_voxel_features, rain_coors = self.get_rain_features(generated_voxel_features)

            # 判别器对真实和生成的数据进行评分
            real_score = self.discriminator(rain_voxel_features.to(device), rain_coors.to(device)).view(-1)
            fake_score = self.discriminator(generated_voxel_features.detach().to(device), coors.detach().to(device)).view(-1)

            # 计算判别器的损失
            loss_d_real = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
            loss_d_fake = F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
            loss_d = (loss_d_real + loss_d_fake) / 2
        optimizer_discriminator.update_params(loss_d)

        # 更新生成器
        optimizer_generator = optim_wrapper['generator']
        # optimizer_generator.zero_grad()
        with optimizer_generator.optim_context(self):
            fake_score_g = self.discriminator(generated_voxel_features.to(device), coors.to(device)).view(-1)
            loss_g = F.binary_cross_entropy_with_logits(fake_score_g, torch.ones_like(fake_score_g))
        optimizer_generator.update_params(loss_g)

        # 返回损失值用于记录或展示
        log_vars = self.parse_losses({'loss_d': loss_d, 'loss_g': loss_g})
        return log_vars

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        log_vars = OrderedDict(log_vars)  # type: ignore

        return log_vars  # type: ignore
    
    def forward(self, batch_inputs_dict: dict,) -> Dict[str, torch.Tensor]:
        # 生成恶劣天气点云
        # print('进入了GANModel!')
        generated_voxel_features = self.extract_feat_forward(batch_inputs_dict)  # use_targe = False
        # generated_voxel_features, _, _, _ = self.extract_feat(batch_inputs_dict) # use_targe = True
        batch_inputs_dict['voxels']['voxels'] = generated_voxel_features
        return batch_inputs_dict
    
    def train_withSeg(self, batch_inputs_dict: dict,) -> Dict[str, Tensor]:
        # 确保所有张量都在同一设备上
        device = next(self.parameters()).device
        
        # 提取特征并生成恶劣天气点云
        generated_voxel_features, coors, rain_voxel_features, rain_coors = self.extract_feat(batch_inputs_dict)
        # generated_voxel_features, coors = self.extract_feat(data)
        # rain_voxel_features, rain_coors = self.get_rain_features(generated_voxel_features)

        # 判别器对真实和生成的数据进行评分
        real_score = self.discriminator(rain_voxel_features.to(device), rain_coors.to(device)).view(-1)
        # fake_score = self.discriminator(generated_voxel_features.detach().to(device), coors.detach().to(device)).view(-1)
        fake_score = self.discriminator(generated_voxel_features.to(device), coors.to(device)).view(-1)

        # 计算判别器的损失
        loss_d_real = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
        loss_d_fake = F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
        
        # alpha = 0.001
        alpha = 1.0
        loss_d_real *= alpha
        loss_d_fake *= alpha

        # 返回损失值用于记录或展示
        log_vars = self.parse_losses({'loss_d_real': loss_d_real, 'loss_d_fake': loss_d_fake})

        batch_inputs_dict['voxels']['voxels'] = generated_voxel_features
        return batch_inputs_dict, log_vars

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