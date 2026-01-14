import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from device import DEVICE
from mha import MHA
from res_net import ResNet

'''
Implements a single encoder layer from the classical Transformer architecture

* d_model = num_head * d_k, where d_k is the dimension of a q,k,v vector
* num_head: number of heads
* res_net_hidden_dims: dimensions of the hidden layers in the residual network
* activation_module_class: activation used inside res_net
'''

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_head: int,
                 res_net_hidden_dims: list[int],
                 activation_module_class):
        super().__init__()
        self.mha = MHA(num_head, d_model, mask=None)
        self.res_net = ResNet(d_model, res_net_hidden_dims, activation_module_class)
        self.mha_layer_norm = nn.LayerNorm(d_model)

    
    def forward(self,
                x: Tensor, # (B, s, d_model)
                ):
        mha_out = self.mha(x) # (B, s, d_model)
        mha_out = self.mha_layer_norm(mha_out + x) # (B, s, d_model)
        return self.res_net(mha_out) # (B, s, d_model)

class Encoder(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 d_model: int,
                 num_head: int,
                 res_net_hidden_dims: list[int],
                 activation_module_class):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        encoder_layers = [EncoderLayer(d_model, num_head, res_net_hidden_dims, activation_module_class) \
                                for _ in range(num_encoder_layers)]
        self.encoder_layers = nn.Sequential(*encoder_layers)
    
    def forward(self,
                x: Tensor, # (B, s, d_model)
                ):
        return self.encoder_layers(x)


