#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from src.conv_layers import ANDConvLayer, ORConvLayer
from src.base_layers import ORLayer
from src.pruned_weights import PrunedWeight, PrunedWeight3D
from src.weights import FixedOnesWeight, StackedBinaryWeight3D


class StackedORConvLayer(ORConvLayer):

    def __init__(self, input_size, window_size, pad_border, max_sequence_length, output_size=1):
        weight = StackedBinaryWeight3D(dim_in=input_size, dim_out=output_size, kernel_size=window_size,
                                       binary_weight_class=PrunedWeight)

        super(StackedORConvLayer, self).__init__(weight=weight, kernel_size=window_size, input_size=input_size,
                                                 output_size=window_size, pad_border=pad_border,
                                                 conv_out=None, parenthesis=True)


class BaseModelANDConvLayer(ANDConvLayer):

    def __init__(self, window_size, base_model_hidden_size, conv_dim_out):
        weight = PrunedWeight3D(dim_in=window_size, dim_out=base_model_hidden_size,
                                kernel_size=1)

        super(BaseModelANDConvLayer, self).__init__(weight=weight, kernel_size=1, input_size=window_size,
                                                    output_size=base_model_hidden_size, pad_border=False,
                                                    conv_out=None)


class BaseModelORConvLayer(ORConvLayer):

    def __init__(self, base_model_hidden_size, base_or_output_size, conv_dim_out):
        weight = PrunedWeight3D(dim_in=base_model_hidden_size, dim_out=base_or_output_size,
                                kernel_size=1)

        super(BaseModelORConvLayer, self).__init__(weight=weight, kernel_size=1, input_size=base_model_hidden_size,
                                                   output_size=base_or_output_size, pad_border=False,
                                                   conv_out=conv_dim_out)


class ConvORLayer(ORLayer):

    def __init__(self, input_size, output_size):
        super(ConvORLayer, self).__init__(input_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[1] * x.shape[2])
        return super(ConvORLayer, self).forward(x)


class FixedConvORLayer(ConvORLayer):

    def __init__(self, input_size, output_size):
        super(FixedConvORLayer, self).__init__(input_size, output_size)
        self.weight = FixedOnesWeight(input_size, output_size)
