#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from src.data.sampler import BucketBatchSampler
from src.data.datasets import SequenceDataset


def construct_loader(dataset, indices, batch_size):
    if indices is not None:
        # create sampler
        sampler = BucketBatchSampler(np.array(dataset.x, dtype=object)[indices], batch_size)

        # generate subset based on indices
        dataset = Subset(dataset, indices)
    else:
        # create sampler
        sampler = BucketBatchSampler(np.array(dataset.x, dtype=object), batch_size)

    # create batches
    loader = DataLoader(dataset, batch_sampler=sampler, batch_size=1)
    return loader


def construct_data(dataset: SequenceDataset, batch_size: int, val_size: float, test_size: float, seed: int):
    # generate indices: instead of the actual data we pass in integers instead
    train_indices, remain_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.y,
        stratify=dataset.y,
        test_size=test_size + val_size,
        random_state=seed
    )
    remain_y = torch.index_select(dataset.y, 0, torch.tensor(remain_indices))

    # generate indices: instead of the actual data we pass in integers instead
    val_indices, test_indices, _, _ = train_test_split(
        range(len(remain_y)),
        remain_y,
        stratify=remain_y,
        test_size=test_size/(test_size + val_size),
        random_state=seed
    )
    val_indices = list(np.array(remain_indices)[val_indices])
    test_indices = list(np.array(remain_indices)[test_indices])

    train_loader = construct_loader(dataset, train_indices, batch_size)
    val_loader = construct_loader(dataset, val_indices, len(val_indices))
    test_loader = construct_loader(dataset, test_indices, len(test_indices))
    
    return train_loader, val_loader, test_loader
