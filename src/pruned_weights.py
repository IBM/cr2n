#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
from torch.nn import Parameter
from src.weights import BinaryWeight, BinaryWeight3D

# Inspired by https://github.com/INCHEON-CHO/Dynamic_Model_Pruning_with_Feedback


class Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        return grad, None

    
class PrunedWeight(BinaryWeight):
    def __init__(self, dim_in, dim_out):
        super(PrunedWeight, self).__init__(dim_in, dim_out)
        self.pruning_mask = Parameter(torch.ones(self.loc.size()), requires_grad=False)

    def forward(self):
        weight = super(PrunedWeight, self).forward()
        masked_weight = Masker.apply(weight, self.pruning_mask)
        return masked_weight
    
    def get_binary_value(self):
        binary_value = super(PrunedWeight, self).get_binary_value()
        binary_value = Masker.apply(binary_value, self.pruning_mask)
        return binary_value
    
    
class PrunedWeight3D(BinaryWeight3D):
    def __init__(self, dim_in, dim_out, kernel_size):
        super(PrunedWeight3D, self).__init__(dim_in, dim_out, kernel_size)
        self.pruning_mask = Parameter(torch.ones(self.loc.size()), requires_grad=False)

    def forward(self):
        weight = super(PrunedWeight3D, self).forward()
        pruned_weight = Masker.apply(weight, self.pruning_mask)
        return pruned_weight

    def get_binary_value(self):
        binary_value = super(PrunedWeight3D, self).get_binary_value()
        binary_value = Masker.apply(binary_value, self.pruning_mask)
        return binary_value
