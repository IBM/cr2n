#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable

# Inspired by https://github.com/moskomule/l0.pytorch


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


class BinaryWeight(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(BinaryWeight, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.loc = torch.nn.Parameter(torch.empty(dim_in, dim_out), requires_grad=True)
        init.xavier_uniform_(self.loc)
        self.register_buffer("uniform", torch.zeros(dim_in, dim_out))
        self.temp = 2/3
        self.gamma = -0.1
        self.zeta = 1.1

    def forward(self):
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
        else:
            return self.get_binary_value()
        return hard_sigmoid(s)

    def get_binary_value(self):
        return torch.heaviside(self.loc, torch.ones_like(self.loc))

    
class BinaryWeight3D(BinaryWeight):
    
    def __init__(self, dim_in, dim_out, kernel_size):
        super(BinaryWeight3D, self).__init__(dim_in, dim_out)
        self.kernel_size = kernel_size
        self.loc = torch.nn.Parameter(torch.empty(dim_out, dim_in, kernel_size), requires_grad=True)
        init.xavier_uniform_(self.loc)
        self.register_buffer("uniform", torch.zeros(dim_out, dim_in, kernel_size))


class StackedBinaryWeight3D(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, binary_weight_class):
        super(StackedBinaryWeight3D, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        weights = []
        for i in range(kernel_size):
            weight = binary_weight_class(dim_in=dim_in, dim_out=dim_out)
            weights.append(weight)
        self.weights = nn.ModuleList(weights)
        self.loc = torch.nn.Parameter(torch.block_diag(*[weight.loc for weight in self.weights]))

    def forward(self):
        if self.training:
            # (dim_in x kernel size, dim_out x kernel size) to (kernel size, dim_in, dim_out x kernel size)
            w = torch.block_diag(*[weight() for weight in self.weights]).reshape(self.kernel_size, self.dim_in,
                                                                                 self.kernel_size * self.dim_out)
        else:
            w = self.get_binary_value().reshape(self.kernel_size, self.dim_in, self.kernel_size * self.dim_out)
        # (dim_in, dim_out x kernel size, kernel_size)
        w = w.permute(2, 1, 0)
        return w

    def get_binary_value(self):
        w = torch.block_diag(*[weight.get_binary_value() for weight in self.weights])
        return w
    

class FixedOnesWeight(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super(FixedOnesWeight, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.values = torch.nn.Parameter(torch.ones(dim_in, dim_out), requires_grad=False)

    def forward(self):
        return self.values
