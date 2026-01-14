from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from device import DEVICE

@dataclass
class ResNetConfig:
  d_model: int
  hidden_dims: list[int] # dimensions of the intermediate layers
  activation_module_class: type # class, not instance, e.g. nn.ReLU

'''
Residual Network: classical ffn component of a transformer
'''
class ResNet(nn.Module):
  def __init__(self,
               res_net_config: ResNetConfig,
               ):
    super().__init__()
    self.res_net_config = res_net_config
    self._d_model = res_net_config.d_model
    self._action_module_class = res_net_config.activation_module_class
    all_dims = [self._d_model] + res_net_config.hidden_dims

    layers = []
    # All but the last layer
    for i in range(len(all_dims) - 1):
      layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
      layers.append(self._activation_module_class())
    
    # Add the final projection layer back to d_model
    layers.append(nn.Linear(all_dims[-1], self._d_model))
    
    # Unpack the list of layers into nn.Sequential
    self._ffn = nn.Sequential(*layers)

    self._layer_norm = nn.LayerNorm(self._d_model)
  
  def forward(self,
              x: Tensor, # (B, s, d_model)
              ):
    o = self._ffn(x) + x
    return self._layer_norm(o)
