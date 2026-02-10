from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from device import DEVICE
from mha import MHA, MHAConfig
from res_net import ResNet, ResNetConfig

'''
Implements a classic sinusoidal positional encoding scheme.
'''
def SinusoidalPE(x: Tensor, # (B, seq_length, d_model)
                 max_seq_length: int = 10000# 10000 in the original paper
                ):
    assert len(x.size()) == 3
    B, s, d_model = x.size()
    assert d_model % 2 == 0

    # Enforce no grad to avoid having PE in the computation graph
    # instead of being a fixed-value operation
    with torch.no_grad():
        pe = torch.zeros(s, d_model, device=DEVICE)

        position = torch.arange(0, s, device=DEVICE).unsqueeze(1) # (s, 1)

        # Calculate the denominator
        # Using log space for numerical stability
        # Already mutiplied by -1 in the exponent to convert to the denominator
        # (d_model // 2, )
        denominator = torch.exp(-torch.log(max_seq_length) * \
                torch.arange(0, d_model, 2, device=DEVICE) / d_model)
        
        # populate the sin part
        # Shape: (s, d_model)
        pe[:, 0::2] = torch.sin(position * denominator.unsqueeze(0))

        # populate the cos part
        # Shape: (s, d_model)
        pe[:, 1::2] = torch.cos(position * denominator.unsqueeze(0))

    # Add PE
    return x + pe.unsqueeze(0)
        