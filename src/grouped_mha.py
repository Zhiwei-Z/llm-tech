from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from device import DEVICE
from shared_config import SharedConfig

@dataclass
class GroupedMHAConfig:
  num_q_head: int # Number of query heads.
  num_kv_head: int # Number of key/value heads.

'''
Grouped Multi-Head Attention

Implements the Grouped Multi-Head Attention mechanism: 

Input: (B, s, d_model) batch_size x sequence_length x d_model
Output: (B, s, d_model) batch_size x sequence_length x d_model
'''
class MHA(nn.Module):
  def __init__(self,
               shared_config: SharedConfig,
               grouped_mha_config: GroupedMHAConfig,
               ):
    super().__init__()
    assert shared_config.d_model % grouped_mha_config.num_q_head == 0, \
        "d_model must be divisible by the number of query heads."
    assert grouped_mha_config.num_q_head % grouped_mha_config.num_kv_head == 0, \
        "Number of Q heads must be divisible by the number of KV heads."
    
    self._num_q_head = grouped_mha_config.num_q_head
    self._num_kv_head = grouped_mha_config.num_kv_head
    self._d_model = shared_config.d_model
    self._ratio = self._num_q_head // self._num_kv_head
    
    self._head_dim = self._d_model // self._num_q_head

    # Q, K, V projection matrix
    self._qkv_proj = nn.Linear(self._d_model, \
        (self._num_q_head + 2 * self._num_kv_head) * self._head_dim)
    
    # Output projection matrix
    self._o_proj = nn.Linear(self._d_model, self._d_model)

  def forward(self,
              x: Tensor, # (B, s, d_model)
              mask: Tensor, # (s, s), optional: usually a causal mask
              ):
    assert len(x.size()) == 3
    assert x.size()[-1] == self._d_model
    B, s, _ = x.shape

    qkv: Tensor = self._qkv_proj(x) # (B, s, (q + 2kv) * d_head)
    
    q_width = self._num_q_head * self._head_dim
    kv_width = self._num_kv_head * self._head_dim

    Q, K, V = torch.split(qkv, [q_width, kv_width, kv_width], dim=-1)
    Q = Q.view(B, s, self._num_q_head, self._head_dim) # (B, s, q_head, d_head)
    K = K.view(B, s, self._num_kv_head, self._head_dim) # (B, s, kv_head, d_head)
    V = V.view(B, s, self._num_kv_head, self._head_dim) # (B, s, kv_head, d_head)
    
    Q = Q.transpose(1, 2) # (B, q_head, s, d_head)
    K = K.transpose(1, 2) # (B, kv_head, s, d_head)
    V = V.transpose(1, 2) # (B, kv_head, s, d_head)
    
    # Expand K, V
    K = K.unsqueeze(2) # (B, kv_head, 1, s, d_head)
    V = V.unsqueeze(2) # (B, kv_head, 1, s, d_head)
    K = K.expand(-1, -1, self._ratio, -1, -1) # (B, kv_head, ratio, s, d_head)
    V = V.expand(-1, -1, self._ratio, -1, -1) # (B, kv_head, ratio, s, d_head)
    K = K.view(B, self._num_q_head, s, self._head_dim) # (B, q_head, s, d_head)
    V = V.view(B, self._num_q_head, s, self._head_dim) # (B, q_head, s, d_head)

    a = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self._head_dim) # (B, q_head, s, s)
    if mask is not None:
      a = a.masked_fill(mask == 0, float('-inf'))
    a = F.softmax(a, dim=-1) # (B, q_head, s, s)
    
    o: Tensor = torch.matmul(a, V) # (B, q_head, s, d_head)
    o = o.transpose(1, 2).contiguous() # (B, s, q_head, d_head)
    o = o.view(B, s, self._d_model) # (B, s, d_model)

    o = self._o_proj(o) # (B, s, d_model)
    return o
