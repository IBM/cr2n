#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
from torch import nn
from src.weights import BinaryWeight


class ORLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super(ORLayer, self).__init__()
        self.weight = BinaryWeight(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        weight = self.weight()
        prod = torch.matmul(x, weight)
        output = torch.clamp(prod, max=1)
        return output
    

class ANDLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super(ANDLayer, self).__init__()
        self.weight = BinaryWeight(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        weight = self.weight()
        neg_input = torch.ones_like(x) - x
        prod = torch.matmul(neg_input, weight)
        output = torch.ones_like(prod) - torch.clamp(prod, max=1)
        return output

    
class StackORLayer(nn.Module):

    def __init__(self, distribution):
        super(StackORLayer, self).__init__()
        layers = []
        for i in distribution:
            layers.append(ORLayer(input_size=i, output_size=1))
        self.blocks = nn.ModuleList(layers)
        self.distribution = distribution
        self.input_size = sum(distribution)
        self.output_size = len(distribution)

    def forward(self, x):
        output = []
        for i, block in enumerate(self.blocks):
            start = sum(self.distribution[:i])
            end = start + self.distribution[i]
            output.append(block(x[:, start:end]))
        output = torch.stack(output)
        output = output.squeeze().permute(1, 0)
        return output
