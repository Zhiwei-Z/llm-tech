import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from mha import MHA

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
print(f"Torch using device {DEVICE}")

'''
Residual Network: classical ffn component of a transformer
'''
class ResNet(nn.Module):
  def __init__(self,
               d_model,
               hidden_dims: list[int], # dimensions of the intermediate layers
               activation_module_class # class, not instance, e.g. nn.ReLU
               ):
    super().__init__()
    self._d_model = d_model
    all_dims = [d_model] + hidden_dims

    layers = []
    # All but the last layer
    for i in range(len(all_dims) - 1):
      layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
      layers.append(activation_module_class())
    
    # Add the final projection layer back to d_model
    layers.append(nn.Linear(all_dims[-1], d_model))
    
    # Unpack the list of layers into nn.Sequential
    self._ffn = nn.Sequential(*layers)

    self._layer_norm = nn.LayerNorm(d_model)
  
  def forward(self,
              x: Tensor, # (B, s, d_model)
              ):
    o = self._ffn(x) + x
    return self._layer_norm(o)
