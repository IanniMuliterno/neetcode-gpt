import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        loc = 0.0
        scale = math.sqrt(2/(fan_in + fan_out))
        empty = torch.empty([fan_out,fan_in])
        normal_w = nn.init.normal_(empty, loc, scale)
        return torch.round(normal_w,decimals = 4).tolist()

        

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        loc = 0.0
        scale = math.sqrt(2/(fan_in))
        empty = torch.empty([fan_out,fan_in])
        normal_w = nn.init.normal_(empty,loc, scale)
        return torch.round(normal_w,decimals = 4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        w = []
        stds = []
        for l in range(num_layers):

            f_in = input_dim if l == 0 else hidden_dim
            
            if init_type == 'xavier':
                std = math.sqrt(2/(f_in + hidden_dim))
            elif init_type == 'kaiming':
                std = math.sqrt(2/(f_in))
            else: 
                std = 1.0

            w.append(torch.randn([hidden_dim, f_in]) * std) 
            
        x = torch.randn([1,input_dim])
        for w_ in w: 
            x = torch.relu(x @ w_.T)
            stds.append(round(x.std().item(), 2))
        return stds