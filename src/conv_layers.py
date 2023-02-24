#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
from torch import nn
from torch.nn import functional as F


class Conv1DCustom(nn.Module):

    def __init__(self, weight, bias, kernel_size, input_size, output_size, stride=1, pad_border=False):
        super().__init__()
        self.weight = weight
        self.bias = bias

        self.kernel_size = kernel_size
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        weight = self.weight()
        assert weight.size() == (self.output_size, self.input_size, self.kernel_size)
        bias = None
        if self.bias is not None:
            bias = self.bias(weight)
            assert bias.size() == (self.output_size,)

        return F.conv1d(x, weight, bias=bias)


class ConvLayer(nn.Module):

    def __init__(self, weight, bias, kernel_size, input_size,
                 output_size, conv_out, stride=1, pad_border=False, activation=None, **kwargs):
        super(ConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv_out = conv_out

        # Conv1d expects input : batch x features x sequence length
        self.conv_layer = Conv1DCustom(weight=weight, bias=bias, input_size=input_size, output_size=output_size,
                                       kernel_size=kernel_size)

        self.internal_input_size = input_size
        self.kernel_size = kernel_size

        self.pad_border = pad_border

        layers = [self.conv_layer]
        if activation:
            layers.append(activation)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # time is ordered by t-N, t-N-1,... t-1, t
        # input : (sample, sequence_length, features)
        # conv_layer input: sample x features x sequence length

        if self.pad_border and self.kernel_size > 1:
            # Padding input at the beginning and end of sequence to fill blanks for rolling window
            # example "ABCD" window size 2 --> *A, AB, BC, CD, D*
            # input : (sample, sequence_length + (window_size - 1)*2, features)
            # sequence_length = sequence_length + (window_size - 1)*2
            x = torch.nn.functional.pad(input=x,
                                        pad=(self.kernel_size - 1, self.kernel_size - 1),
                                        mode='constant', value=0)

        # conv_layer input: sample x features x sequence length
        # conv_layer output:  sample x output size x (N - window_size + 1)
        output = self.model(x)

        # padding for output to be of max size, corresponding to seq of size max_sequence_length
        # sample x output size x (max_sequence_length - window_size + 1)
        # Padding at beginning due to chronological input

        if self.conv_out and self.conv_out != output.shape[-1]:
            output = torch.nn.functional.pad(input=output,
                                             pad=(self.conv_out - output.size(-1), 0),
                                             mode='constant', value=0)

        # if not self.training and self.input_size==6:
        #    print(output)
        return output


class ANDConvLayer2(ConvLayer):

    def __init__(self, weight, kernel_size, input_size,
                 output_size, conv_out, pad_border=False):
        # an alternate way of writing the same expression
        def bias(w):
            weight_reshaped = torch.reshape(w, (w.size(0), w.size(1) * w.size(2)))
            return 1 - weight_reshaped.sum(1)

        super(ANDConvLayer2, self).__init__(weight=weight, bias=bias, kernel_size=kernel_size,
                                            input_size=input_size, output_size=output_size, pad_border=pad_border,
                                            activation=nn.ReLU(), conv_out=conv_out)
        self.weight = weight


class MinActivation(nn.Module):
    def __init__(self, value: int = 1):
        super(MinActivation, self).__init__()
        self.value = value

    def forward(self, x):
        return -torch.clamp(x, min=-self.value)  # reason : -torch.clamp(-a, min=-1) == torch.clamp(a, max=1)


class ANDConvLayer(ConvLayer):

    def __init__(self, weight, kernel_size, input_size,
                 output_size, conv_out, pad_border=False):
        # an alternate way of writing the same expression
        def bias(w):
            weight_reshaped = torch.reshape(w, (w.size(0), w.size(1) * w.size(2)))
            return -weight_reshaped.sum(1)

        super(ANDConvLayer, self).__init__(weight=weight, bias=bias, kernel_size=kernel_size,
                                           input_size=input_size, output_size=output_size, pad_border=pad_border,
                                           activation=MinActivation(), conv_out=conv_out)
        self.weight = weight

    def forward(self, x):
        output = super(ANDConvLayer, self).forward(x)
        output = torch.ones_like(output) - output
        return output


class MaxActivation(nn.Module):
    def __init__(self, max_value: int = 1):
        super(MaxActivation, self).__init__()
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, max=self.max_value)


class ORConvLayer(ConvLayer):

    def __init__(self, weight, kernel_size, input_size, output_size, conv_out, pad_border=False, parenthesis=False):
        bias = None

        super(ORConvLayer, self).__init__(weight=weight, bias=bias, kernel_size=kernel_size,
                                          input_size=input_size, output_size=output_size, pad_border=pad_border,
                                          activation=MaxActivation(), conv_out=conv_out)
        self.weight = weight
