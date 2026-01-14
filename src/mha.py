from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from device import DEVICE

@dataclass
class MHAConfig:
  num_head: int # Number of attention heads.
  d_model: int # The dimension of the model's embeddings.

'''
Multi-Head Attention

Implements the Multi-Head Attention mechanism as described in the "Attention is All You Need" paper.
It takes a sequence of embeddings and applies self-attention, allowing the model to weigh the
importance of different words in the sequence when processing each word.

Input: (B, s, d_model) batch_size x sequence_length x d_model
Output: (B, s, d_model) batch_size x sequence_length x d_model
'''
class MHA(nn.Module):
  def __init__(self,
               mha_config: MHAConfig,
               ):
    super().__init__()
    # d_model must be divisible by the number of heads.
    assert mha_config.d_model % mha_config.num_head == 0
    self._num_head = mha_config.num_head
    self._d_model = mha_config.d_model
    self._d_k = self._d_model // self._num_head
    
    self._qkv_proj = nn.Linear(self._d_model, self._d_model * 3)
    # This final projection layer combines the outputs of the different heads.
    self._o_proj = nn.Linear(self._d_model, self._d_model)

  def forward(self,
              x: Tensor, # (B, s, d_model)
              mask: Tensor, # (s, s), optional: used to hide future tokens in a sequence.
              ):
    assert len(x.size()) == 3
    assert x.size()[-1] == self._d_model
    B, s, _ = x.shape

    qkv: Tensor = self._qkv_proj(x) # (B, s, d_model * 3)
    qkv = qkv.view(B, s, self._num_head, self._d_k * 3) # (B, s, num_head, d_k * 3)

    # Split the last dimension into Q, K, V
    Q: Tensor = qkv[:, :, :, :self._d_k] # (B, s, num_head, d_k)
    K: Tensor = qkv[:, :, :, self._d_k:(2 * self._d_k)] # (B, s, num_head, d_k)
    V: Tensor = qkv[:, :, :, (2 * self._d_k):] # (B, s, num_head, d_k)

    # Transpose to get dimensions (B, num_head, s, d_k) for attention calculation
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    # Calculate scaled dot-product attention scores
    a: Tensor = torch.matmul(Q, (K.transpose(-2, -1))) / np.sqrt(self._d_k) # (B, num_head, s, s)

    if mask is not None:
      a.masked_fill(mask == 0, -1e9)

    a = F.softmax(a, dim=-1) # (B, num_head, s, s)
    o: Tensor = torch.matmul(a, V) # (B, num_head, s, d_k)
    o = o.transpose(1, 2).contiguous() # (B, s ,num_head, d_k)
    
    # Combine the heads back into a single d_model vector
    o = o.view(B, s, self._d_model) # (B, s, d_model)
    o = self._o_proj(o)
    return o