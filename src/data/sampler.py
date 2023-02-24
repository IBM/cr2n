#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

from collections import OrderedDict
from random import shuffle, Random

from torch.utils.data import Sampler

# Inspired by https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/12


class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, inputs, batch_size):
        self.batch_size = batch_size
        ind_n_len = []
        max_length = 0
        for i, p in enumerate(inputs):
            length = p.shape[0]
            ind_n_len.append((i, length))
            if length > max_length:
                max_length = length
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)
        self.max_length = max_length

    def _generate_batch_map(self):
        Random(4).shuffle(self.ind_n_len)
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        Random(4).shuffle(self.batch_list)
        for i in self.batch_list:
            yield i
