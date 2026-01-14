from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from device import DEVICE
from mha import MHA, MHAConfig
from res_net import ResNet, ResNetConfig

@dataclass
class EncoderLayerConfig:
  mha_config: MHAConfig
  res_net_config: ResNetConfig

'''
Implements a single encoder layer from the classical Transformer architecture.
An encoder layer consists of two main sub-layers: Multi-Head Attention and a
Position-wise Feed-Forward Network. Residual connections and layer normalization
are applied after each sub-layer.
'''
class EncoderLayer(nn.Module):
    def __init__(self,
                 encoder_layer_config: EncoderLayerConfig
                 ):
        super().__init__()
        self.encoder_layer_config = encoder_layer_config
        self.mha = MHA(encoder_layer_config.mha_config)
        self.res_net = ResNet(encoder_layer_config.res_net_config)
        self.mha_layer_norm = nn.LayerNorm(encoder_layer_config.mha_config.d_model)

    def forward(self,
                x: Tensor, # (B, s, d_model)
                ):
        # In a standard Transformer encoder, the mask is None because each token
        # in the input sequence is allowed to attend to all other tokens.
        mha_out = self.mha(x, mask=None) # (B, s, d_model)

        # Residual connection and layer normalization for the MHA sub-layer
        mha_out = self.mha_layer_norm(mha_out + x) # (B, s, d_model)

        # The ResNet (FFN) sub-layer already includes its own residual connection and layer norm.
        return self.res_net(mha_out) # (B, s, d_model)

@dataclass
class EncoderConfig:
  encoder_layer_config: EncoderLayerConfig
  num_encoder_layers: int

class Encoder(nn.Module):
    def __init__(self,
                 encoder_config: EncoderConfig
                 ):
        super().__init__()
        self.encoder_config = encoder_config
        self._num_encoder_layers = encoder_config.num_encoder_layers
        encoder_layers = [EncoderLayer(encoder_config.encoder_layer_config) \
                                for _ in range(self._num_encoder_layers)]
        self._encoder_layers = nn.Sequential(*encoder_layers)
    
    def forward(self,
                x: Tensor, # (B, s, d_model)
                ):
        return self._encoder_layers(x)
