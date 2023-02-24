#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

DEFAULT_PATH = "checkpoint.pt"


def save_checkpoint(state, path=DEFAULT_PATH):
    torch.save(state, path)


def load_checkpoint(model, optimizer, path=DEFAULT_PATH, verbose=False):
    if path:
        if os.path.isfile(path):
            checkpoint = torch.load(path)

            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_loss = checkpoint['best_loss']
            best_penalty = checkpoint['best_penalty']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if verbose:
                print("=> loaded checkpoint '{}' (epoch {}, acc {}, loss {}, penalty {})"
                      .format(path, start_epoch, best_acc, best_loss, best_penalty))
        else:
            print("=> no checkpoint found at '{}'".format(path))
            
            
basic_metrics_binary = dict(
    acc=accuracy_score,
    bal_acc=balanced_accuracy_score,
)


def compute_metrics(y_true, y_pred, labels):
    res = dict()
    for metric in basic_metrics_binary:
        res[metric] = basic_metrics_binary[metric](y_true, y_pred)
    res['tn'], res['fp'], res['fn'], res['tp'] = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return res
