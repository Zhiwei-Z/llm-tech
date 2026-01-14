import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from device import DEVICE

'''
Multi-Head Attention
Input: (B, s, d_model) batch_size x sequence_length x d_model
Output: (B, s, d_model) batch_size x sequence_length x d_model
Legends:
- d_model: dimension of the model, common for embeddings, transformer block
           and output
- d_k: dimension of a k, q, v vector per head
- num_head: number of heads
- ** d_model = num_head * d_k **
- s: sequence length
'''
class MHA(nn.Module):
  def __init__(self,
               num_head: int,
               d_model: int,
               mask: Tensor):
    super().__init__()
    # d_model is num_head * d_k (dimension of k, also same as q, v)
    assert d_model % num_head == 0
    self._num_head = num_head
    self._d_model = d_model
    self._d_k = d_model // num_head
    # Assumes Q, K, V have the same dimensions.
    # We can split Q, K, V at the end.
    self._qkv_proj = nn.Linear(d_model, d_model * 3)
    self._o_proj = nn.Linear(d_model, d_model)
    self._mask = mask

  def forward(self,
              x: Tensor, # (B, s, d_model)
              ):
    assert len(x.size()) == 3
    assert x.size()[-1] == self._d_model
    B, s, _ = x.shape

    # (B, s, d_model * 3)
    qkv: Tensor = self._qkv_proj(x)
    # (B, s, num_head, d_k * 3)
    qkv = qkv.view(B, s, self._num_head, self._d_k * 3)

    Q: Tensor = qkv[:, :, :, :self._d_k] # (B, s, num_head, d_k)
    K: Tensor = qkv[:, :, :, self._d_k:(2 * self._d_k)] # (B, s, num_head, d_k)
    V: Tensor = qkv[:, :, :, (2 * self._d_k):] # (B, s, num_head, d_k)

    Q = Q.transpose(1, 2) # (B, num_head, s, d_k)
    K = K.transpose(1, 2) # (B, num_head, s, d_k)
    V = V.transpose(1, 2) # (B, num_head, s, d_k)

    # (B, num_head, s, s)
    a: Tensor = torch.matmul(Q, (K.transpose(-2, -1))) / np.sqrt(self._d_k)

    if self._mask is not None:
      a.masked_fill(self._mask == 0, -1e9)

    a = F.softmax(a, dim=-1) # (B, num_head, s, s)
    o: Tensor = torch.matmul(a, V) # (B, num_head, s, d_k)
    o = o.transpose(1, 2).contiguous() # (B, s ,num_head, d_k)
    # Combining the last two dimensions
    o = o.view(B, s, self._d_model) # (B, s, d_model)
    o = self._o_proj(o)
    return o
