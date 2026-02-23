from dataclasses import dataclass
import torch
from torch import nn
from torch import Tensor

from device import DEVICE

"""
Implements a fixed sized kv cache.
Memory allocation will happen during the initialization
"""
class FixedKvCache:
    def __init__(self,
                 num_batches: int, # number of batches in the dataset
                 max_sequence_length: int, # max sequence length allowed
                 num_layers: int, # number of attention layers in the model
                 d_model: int # dimension of a q/k/v already mutiplied by head number
                ):
        self._num_layers = num_layers
        self._max_sequence_length = max_sequence_length
        self._d_model = d_model
        self._num_batches = num_batches

        # Multiplied by 2 to store both k and v
        self._cache = torch.zeros( \
                        (num_layers, num_batches, max_sequence_length, d_model * 2), \
                            device=DEVICE)
    
    def store(self,
              layer_idx: int,
              start_pos: int,
              kv: Tensor # (B, s, d_model * 2)
             ):
        assert 0 <= layer_idx < self._num_layers
        assert len(kv.size()) == 3
        assert kv.size(-1) == self._d_model * 2
        assert kv.size(0) == self._num_batches
        _, s, _ = kv.size()
        assert start_pos + s <= self._max_sequence_length

        self._cache[layer_idx, :, start_pos:start_pos + s, :] = kv
    
    def fetch(self,
              layer_idx: int,
              sequence_length: int,
             ):
        assert 0 <= layer_idx < self._num_layers
        assert sequence_length <= self._max_sequence_length
        return self._cache[layer_idx, :, :sequence_length, :]
