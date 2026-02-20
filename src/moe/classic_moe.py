from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from device import DEVICE
from mha import MHA, MHAConfig
from ..res_net import ResNet, ResNetConfig
from ..shared_config import SharedConfig

@dataclass
class RouterConfig:
    num_experts: int # dimension of the output layer, represening expert weights

class Router(nn.Module):
    def __init__(self,
                 shared_config: SharedConfig,
                 router_config: RouterConfig):
        super().__init__()
        self._config = router_config
        self._d_model = shared_config.d_model

        self._linear = nn.Linear(self._d_model, self._config.num_experts)

    def forward(self, 
                x: Tensor, # (B, s, d_model)
                ):
        return self._linear(x)


@dataclass
class ClassicMoEConfig:
    num_experts: int # total number of experts
    k: int # number of selected experts to generate the output
    expert_resnet_config: ResnetConfig # config for each expert

class ClassicMoE(nn.Module):
    def __init__(self,
                 shared_config: SharedConfig,
                 classic_moe_config: ClassicMoEConfig,
                 ):
        super().__init__()
        self._config = classic_moe_config
        self._d_model = classic_moe_config.d_model
        self._expert_resnet_config = classic_moe_config.expert_resnet_config

        # Set up the router
        self._router = Router(shared_config, RouterConfig(self._config.num_experts))
        
        # Set up the experts
        self._experts = nn.ModuleList()
        for i in range(classic_moe_config.num_experts):
            self._experts.append(ResNet(self._expert_resnet_config))
    
    def forward(self,
                x: Tensor # (B, s, d_model)
               ):
        logits = self._router(x) # (B, s, num_experts)

        # (B, s, k)
        top_k_logits, top_k_indices = torch.topk(logits, self._config.k, dim=-1)

        top_k_weights = torch.softmax(top_k_logits, dim=-1) # (B, s, k)

        out = torch.zeros_like(x)

        for i in range(self._config.k):
            expert_idx = top_k_indices[..., i] # (B, s)
            expert_weight = top_k_weights[..., i].unsqueeze(-1) # (B, s, 1)

            for j in range(self._config.num_experts):
                expert_mask = (expert_idx == j) # (B, s)

                # Ignore if this expert is not selected
                if expert_mask.any():
                    # Pass through the experts
                    expert_out = self._experts[j](x[expert_mask]) # (B, s, d_model)
                    # Apply weights
                    out[expert_mask] += expert_weight[expert_mask] * expert_out # (B, s, d_model)
         
        return out
