#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import numpy as np
import torch

# Inspired by https://github.com/INCHEON-CHO/Dynamic_Model_Pruning_with_Feedback


class Pruning:
    
    def __init__(self, start=0, freq=16, rate=0.99):
        self.start = start
        self.freq = freq
        self.rate = rate


def get_pruning_rate(end_rate, iteration, tot_iterations):
    return end_rate - end_rate * (1 - iteration / tot_iterations) ** 3


def prune_weights(model, rate):
    state = model.state_dict()
    for name, item in model.named_parameters():
        if '.loc' in name:
            key = name.replace('.loc', '.pruning_mask')
            if key in state.keys():
                weights = item.data.view(-1).cpu()
                importance = weights[weights > 0].numpy()
                if len(importance) > 0:
                    thres = np.sort(importance)[-1] * rate                
                    mat = item.data.abs()
                    new_pruned_mask = torch.ge(mat, thres).float()
                    state[key].data.copy_(new_pruned_mask)
