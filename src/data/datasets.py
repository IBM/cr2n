#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.utils import resample

torch.manual_seed(0)


class SequenceDataset(Dataset):

    def __init__(self, x, y, features_name):
        self.x = x
        self.y = torch.tensor(y).unsqueeze(dim=1)
        self.features_name = features_name

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i].permute(1, 0), self.y[i]


class SynthCSVDownSampledDataset(SequenceDataset):

    def __init__(
            self,
            path: str
    ):
        data, y, features_name, rule = self.read_dataset_folder(path)
        super(SynthCSVDownSampledDataset, self).__init__(data, y, features_name)

    @staticmethod
    def read_dataset_folder(path):

        df = pd.read_csv(path + '/data.csv')
        df_0 = resample(
            df[df["y"] == 0.],
            replace=True,
            n_samples=len(df[df["y"] == 1.]),
            random_state=42
        )
        print(len(df_0))
        df = pd.concat([df_0, df[df["y"] == 1.]])
        print(df["y"].value_counts())
        data = df['sequence'].to_numpy()

        char_sequences = [np.array(list(seq_el)) for seq_el in data]
        dictionary = sorted(list(set("".join(data))))
        char_to_int = dict((c, i) for i, c in enumerate(dictionary))

        # convert to 1-hot encoding
        data = []
        for seq in char_sequences:
            onehot_encoded = list()
            for char in seq:
                letter = [0 for _ in range(len(dictionary))]
                letter[char_to_int[char]] = 1
                onehot_encoded.append(letter)
            data.append(torch.tensor(onehot_encoded, dtype=torch.float32))

        y_class = df["y"].values
        classes = sorted(list(set(y_class)))
        class_to_int_y = dict((c, i) for i, c in enumerate(classes))
        int_to_class_y = dict((i, c) for i, c in enumerate(classes))
        y = [float(class_to_int_y[val]) for val in y_class]

        with open(path + '/rule.txt') as f:
            rule = f.read()

        print('Distrib', sum(y) / len(y), sum(y), len(y))

        return data, y, dictionary, rule


class SynthCSVDataset(SequenceDataset):

    def __init__(
            self,
            path: str
    ):
        data, y, features_name, rule, max_length = self.read_dataset_folder(path)
        super(SynthCSVDataset, self).__init__(data, y, features_name)
        self.max_length = max_length

    @staticmethod
    def read_dataset_folder(path):

        df = pd.read_csv(path + '/data.csv')
        data = df['sequence'].to_numpy()

        char_sequences = [np.array(list(seq_el)) for seq_el in data]
        dictionary = sorted(list(set("".join(data))))
        char_to_int = dict((c, i) for i, c in enumerate(dictionary))

        # convert to 1-hot encoding
        data = []
        max_length = 0
        for seq in char_sequences:
            if len(seq) > max_length:
                max_length = len(seq)
            onehot_encoded = list()
            for char in seq:
                letter = [0 for _ in range(len(dictionary))]
                letter[char_to_int[char]] = 1
                onehot_encoded.append(letter)
            data.append(torch.tensor(onehot_encoded, dtype=torch.float32))

        y_class = df["y"].values
        classes = sorted(list(set(y_class)))
        class_to_int_y = dict((c, i) for i, c in enumerate(classes))
        int_to_class_y = dict((i, c) for i, c in enumerate(classes))
        y = [float(class_to_int_y[val]) for val in y_class]

        with open(path + '/rule.txt') as f:
            rule = f.read()

        print('Distrib', sum(y) / len(y), sum(y), len(y))

        return data, y, dictionary, rule, max_length


class UCIAnticancerDataset(SequenceDataset):

    def __init__(
            self,
            df,
    ):
        data, y, features_name = self.get_data(df)
        super(UCIAnticancerDataset, self).__init__(data, y, features_name)

    @staticmethod
    def get_data(df):
        df.loc[df["class"].isin(["very active", "inactive - exp", "mod active", 'mod. active']), "class"] = "neg"
        df.loc[df["class"].isin(["inactive - virtual"]), "class"] = "pos"
        # X
        sequences = df["sequence"].values
        char_sequences = [np.array(list(seq_el)) for seq_el in sequences]

        dictionary = sorted(list(set("".join(sequences))))

        char_to_int = dict((c, i) for i, c in enumerate(dictionary))
        int_to_char = dict((i, c) for i, c in enumerate(dictionary))

        # convert to 1-hot encoding
        data = []
        for seq in char_sequences:
            onehot_encoded = list()
            for char in seq:
                letter = [0 for _ in range(len(dictionary))]
                letter[char_to_int[char]] = 1
                onehot_encoded.append(letter)
            data.append(torch.tensor(onehot_encoded, dtype=torch.float32))

        # Y
        y_class = df["class"].values

        classes = sorted(list(set(y_class)))

        class_to_int_y = dict((c, i) for i, c in enumerate(classes))
        int_to_class_y = dict((i, c) for i, c in enumerate(classes))

        y = [float(class_to_int_y[val]) for val in y_class]
        print('Distrib', sum(y) / len(y), sum(y), len(y))

        return data, y, dictionary
