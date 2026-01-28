from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from device import DEVICE
from mha import MHA, CrossMHA, MHAConfig
from res_net import ResNet, ResNetConfig

@dataclass
class DecoderLayerConfig:
  mha_config: MHAConfig
  res_net_config: ResNetConfig

class DecoderLayer(nn.Module):
  def __init__(self,
               decoder_layer_config: DecoderLayerConfig,
               ):
    super().__init__()

    self._decoder_layer_config = decoder_layer_config
    self._mha = MHA(decoder_layer_config.mha_config)
    self._cross_mha = CrossMHA(decoder_layer_config.mha_config)
    self._res_net = ResNet(decoder_layer_config.res_net_config)

    self._mha_layer_norm = nn.LayerNorm(decoder_layer_config.mha_config.d_model)
    self._cross_mha_layer_norm = nn.LayerNorm(decoder_layer_config.mha_config.d_model)
  
  def forward(self,
              x: Tensor # (B, s_1, d_model)
              cross_x: Tensor # (B, s_2, d_model)
             ):
    assert len(x.size()) == 3
    B, seq_len, d_model = x.size()
    
    # Upper triangle causal mask
    # TODO: buffer this and not create it every time
    target_causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(DEVICE)

    # Self attention to already decoded output
    mha_out = self._mha(x, mask=target_causal_mask) # (B, s_1, d_model)
    mha_out = self._mha_layer_norm(mha_out + x) # (B, s_1, d_model)

    # Cross attention to the encoder output
    cross_mha_out = self._cross_mha(mha_out, cross_x) # (B, s_1, d_model)
    cross_mha_out = self._cross_mha_layer_norm(cross_mha_out + mha_out) # (B, s_1, d_model)

    # The ResNet (FFN) sub-layer already includes its own residual connection and layer norm.
    return self._res_net(cross_mha_out) # (B, s_1, d_model)
