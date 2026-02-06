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
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
    
    def compute_chunked_attention(self, q_batch, kv_batch, embed_dim, dropout_layer):
        chunk_size = 5000  # Adjust based on your GPU memory size 3000->13000, 5000->23000
        output = torch.zeros_like(q_batch)
        
        for i in range(0, len(q_batch), chunk_size):
            q_chunk = q_batch[i:i + chunk_size]
            scores_chunk = torch.einsum('nc,mc->nm', q_chunk, kv_batch) / (embed_dim ** 0.5)
            attn_weights_chunk = F.softmax(scores_chunk, dim=-1)
            attn_weights_chunk = dropout_layer(attn_weights_chunk)
            output_chunk = torch.einsum('nm,mc->nc', attn_weights_chunk, kv_batch)
            
            output[i:i + chunk_size] = output_chunk
        
        return output

    def forward(self, query, key_value, query_coors, key_value_coors):
        """
        Args:
            query: Tensor of shape (N1, C) where N1 is the number of voxels for clear weather data.
            key_value: Tensor of shape (N2, C) where N2 is the number of voxels for adverse weather data.
            query_coors: Tensor of shape (N1, D+1), where D is spatial dimensions and +1 for batch index.
            key_value_coors: Tensor of shape (N2, D+1).
            
        Returns:
            attn_output: Tensor of shape (N1, C).
        """
        unique_batches = torch.unique(query_coors)
        attn_output = []
        
        for batch_idx in unique_batches:  # Iterate over each batch
            # Select points belonging to the current batch using mask
            query_batch_mask = query_coors == batch_idx
            key_value_batch_mask = key_value_coors == batch_idx
            
            query_batch = query[query_batch_mask]
            key_value_batch = key_value[key_value_batch_mask]
            
            # # Compute similarity scores between each pair of query and key_value within the batch
            # scores = torch.einsum('nc,mc->nm', query_batch, key_value_batch) / (self.embed_dim ** 0.5)
            
            # # Apply softmax to get attention weights
            # attn_weights = F.softmax(scores, dim=-1)
            # del scores  # Release memory used by scores
            # attn_weights = self.dropout(attn_weights)
            
            # # Apply attention weights to key_value features to get output
            # attn_output_batch = torch.einsum('nm,mc->nc', attn_weights, key_value_batch)
            # del attn_weights  # Release memory used by attn_weights

            attn_output_batch = self.compute_chunked_attention(query_batch, key_value_batch, self.embed_dim, self.dropout)
            
            # Release memory used by temporary variables
            del query_batch_mask, key_value_batch_mask, query_batch, key_value_batch
            
            # Append the result
            attn_output.append(attn_output_batch)
            
        # Concatenate all outputs along the first dimension
        final_attn_output = torch.cat(attn_output, dim=0)
        
        return final_attn_output

################### Wq, Wk, Wv #################
# @MODELS.register_module()
# class CrossAttention(nn.Module):
#     def __init__(self, embed_dim=4, dropout=0.1):
#         super(CrossAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.dropout = nn.Dropout(dropout)
        
#         # 使用 Conv1d 替代 Linear 层
#         self.proj_q = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
#         self.proj_k = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
#         self.proj_v = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

#     def forward(self, query, key_value, query_coors, key_value_coors):
#         """
#         Args:
#             query: Tensor of shape (N1, 4) where N1 is the number of voxels for clear weather data.
#             key_value: Tensor of shape (N2, 4) where N2 is the number of voxels for adverse weather data.
#             query_coors: Tensor of shape (N1, 1), indicating batch index for each voxel in query.
#             key_value_coors: Tensor of shape (N2, 1), indicating batch index for each voxel in key_value.
            
#         Returns:
#             attn_output: Tensor of shape (N1, 4).
#         """
#         # Apply convolution transformations to get Q, K, V without chunking
#         query_transformed = self.proj_q(query.unsqueeze(-1)).squeeze(-1)
#         key_transformed = self.proj_k(key_value.unsqueeze(-1)).squeeze(-1)
#         value_transformed = self.proj_v(key_value.unsqueeze(-1)).squeeze(-1)
        
#         unique_batches = torch.unique(query_coors.view(-1))  # 获取唯一的批次ID
        
#         final_attn_output = []
#         for batch_idx in unique_batches:
#             query_batch_mask = query_coors == batch_idx
#             key_value_batch_mask = key_value_coors == batch_idx
            
#             query_batch = query_transformed[query_batch_mask.squeeze()]
#             key_batch = key_transformed[key_value_batch_mask.squeeze()]
#             value_batch = value_transformed[key_value_batch_mask.squeeze()]
            
#             attn_output_batch = self.compute_chunked_attention(query_batch, key_batch, value_batch)
            
#             final_attn_output.append(attn_output_batch)

#         final_attn_output = torch.cat(final_attn_output, dim=0)
        
#         return final_attn_output
    
#     def compute_chunked_attention(self, q_batch, k_batch, v_batch, query_chunk_size=7000):
#         output = []
#         for i in range(0, len(q_batch), query_chunk_size):
#             q_chunk = q_batch[i:i + query_chunk_size]
            
#             with torch.no_grad():  # 禁用梯度跟踪
#                 scores_chunk = torch.einsum('nc,mc->nm', q_chunk, k_batch) / (self.embed_dim ** 0.5)
#                 attn_weights_chunk = F.softmax(scores_chunk, dim=-1)
            
#             attn_weights_chunk = self.dropout(attn_weights_chunk)  # Dropout 需要梯度信息
#             output_chunk = torch.einsum('nm,mc->nc', attn_weights_chunk, v_batch)
            
#             output.append(output_chunk.detach())

#         return torch.cat(output, dim=0)