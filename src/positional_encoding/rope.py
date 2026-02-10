from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from device import DEVICE

'''
Implements a rotary embedding positional encoding scheme with optimized computation.

## Original RoPE idea:
We have position `pos` and index `idx`, representing the location
in the sequence length and `d_model` dimension respectively. Originally, we want to

1) Pair index 2i and 2i + 1
2) Each pair will be rotated with t_i = pos * (10000^(2i/d_model)) 

|cos(t_0), -sin(t_0)| * |x_0| = |cos(t_0)x_0 - sin(t_0)x_1|
|sin(t_0),  cos(t_0)|   |x_1|   |sin(t_0)x_0 + cos(t_0)x_1|

or equivalently

|cos(t_0)x_0 - sin(t_0)x_1|
|cos(t_0)x_1 + sin(t_0)x_0|

and the resulting computation will be

[x_0, x_1, ..., x_d-1] * [cos(t_0), cos(t_0), cos(t_1), cos(t_1), ... cos(t_d/2), cos(t_d/2)] +
    [-x_1, x_0, -x_2, x_3, ...] * [sin(t_0), sin(t_0), sin(t_1), sin(t_1), ... sin(t_d/2), sin(t_d/2)]

However the second part of the addition requires complicated slicing, so we can rearrange the pairing to
speed up the computation.

1) Pair index i and i + d/2
2) Each pair will be rotated with t_i = pos / (10000^(2i/d_model)) 

Notice that In the Original, 2i corresponds to the actual dimension index (0, 2, 4...). While in the optimized,
2i corresponds to 2 times the pair index. So the angle assignments are very different, but still following the
same idea.

|cos(t_0), -sin(t_0)| * | x_0 | = |cos(t_0)x_0 - sin(t_0)x_d/2|
|sin(t_0),  cos(t_0)|   |x_d/2|   |sin(t_0)x_0 + cos(t_0)x_d/2|
 
or equivalently

|cos(t_0)x_0   - sin(t_0)x_d/2|
|cos(t_0)x_d/2 +   sin(t_0)x_0|

So the whole computation for an x at a single position is:

[x_0, x_1, ..., x_d] * [cos(t_0), cos(t_1), ..., cos(t_(d/2-1)), cos(t_0), cos(t_1), .., cos(t_(d/2-1))] + 
    [-x_d/2, -x_(d/2+1), ..., x_d, x_0, x_1, ..., x_(d/2-1)] * [sin(t_0), sin(t_1), ..., sin(t_(d/2-1)), sin(t_0), sin(t_1), ..., sin(t_(d/2-1))]

'''
def RoPE(x: Tensor, # (B, seq_length, d_model)
         rotary_base: int = 10000
        ):
    assert len(x.size()) == 3
    B, s, d_model = x.size()
    assert d_model % 2 == 0
    

    # Enforce no grad to avoid having PE in the computation graph
    # instead of being a fixed-value operation
    with torch.no_grad():
        position = torch.arange(0, s, device=DEVICE).unsqueeze(1) # (s, 1)

        # Calculate the angles coefficient, corresponding to
        # 10000 ^ (2i / d_model) where i ranges from 0 to d_model/2
        # Using log space for numerical stability
        # Shape: (d_model // 2,)
        angles_coef = torch.exp(-torch.log(rotary_base) * \
                torch.arange(0, d_model, 2, device=DEVICE) / d_model)
        
        # pos * (10000^(2i/d_model)) 
        angles = position * angles_coef.unsqueeze(0) # (s, d_model // 2)

        # [cos(t_0), cos(t_1), ..., cos(t_(d/2-1))]
        # Shape: (s, d_model // 2)
        cos_angles = torch.cos(angles)
        # [sin(t_0), sin(t_1), ..., sin(t_(d/2-1))]
        # Shape: (s, d_model // 2)
        sin_angles = torch.sin(angles)

        # [cos(t_0), cos(t_1), ..., cos(t_(d/2-1)), cos(t_0), cos(t_1), .., cos(t_d/2-1)]
        # Shape: (s, d_model)
        angles_1 = torch.cat([cos_angles, cos_angles], dim=-1)
        # [sin(t_0), sin(t_1), ..., sin(t_(d/2-1)), sin(t_0), sin(t_1), ..., sin(t_(d/2-1))]
        # Shape: (s, d_model)
        angles_2 = torch.cat([sin_angles, sin_angles], dim=-1)
        
        half_rotate_x = _half_rotate(x) # (B, s, d_model)

    return x * (angles_1.unsqueeze(0)) + half_rotate_x * (angles_2.unsqueeze(0))


'''
Given [x1, x2, ..., x_d] at the last dimension, return
[-x_d/2, -x_(d/2+1), ..., x_d, x_0, x_1, ..., x_(d/2-1)]
'''
def _half_rotate(x: Tensor # (B, s, d_model)
                ):
    assert len(x.size()) == 3
    B, s, d_model = x.size()
    assert d_model % 2 == 0

    return torch.cat([-x[..., d_model // 2:], x[..., 0:d_model // 2]], dim=-1)
