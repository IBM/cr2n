#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
from src.weights import FixedOnesWeight


def count_terminal_conditions(model):
    model = model.model
    stack_layer_weights = torch.stack([weight() for weight in model[0].weight.weights])
    
    and_layer_weights = model[1].weight().squeeze(-1)
    or_layer_weights = model[2].weight().squeeze(-1)
    
    count = stack_layer_weights.sum(dim=1)
    count = torch.matmul(and_layer_weights, count)
    count = torch.matmul(or_layer_weights, count)
    
    if not isinstance(model[3].weight, FixedOnesWeight):
        conv_or_layer_weights = model[3].weight()
        # count = torch.matmul(conv_or_layer_weights.permute(1,0), count.repeat(conv_or_layer_weights.shape[0], 1))
        count = conv_or_layer_weights.sum()*count
    return count
