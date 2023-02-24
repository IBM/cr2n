#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

def is_best_model(current, best):
    acc, loss, penalty = current
    best_acc, best_loss, best_sparsity_penalty = best

    is_best = acc > best_acc or (acc == best_acc and penalty < best_sparsity_penalty)
    return is_best
