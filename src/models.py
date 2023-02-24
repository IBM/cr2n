#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from torch import nn
from copy import deepcopy
from collections import OrderedDict

from src.cr2n_layers import BaseModelANDConvLayer, BaseModelORConvLayer, ConvORLayer, FixedConvORLayer, \
    StackedORConvLayer
from src.rule_language import Conjunction, Disjunction, Node, Evaluated


def extract_base_model_rule(or_layer_weights, and_layer_weights, stack_layer_weights_t, features_names, window_size,
                            verbose=False):
    # Extract base rule model rules
    base_model_rules = []
    for rule in or_layer_weights:
        disjunction_list = []
        for dis in or_layer_weights[0].nonzero(as_tuple=True)[0]:
            conjunction_list = []
            for conj in and_layer_weights[dis].nonzero(as_tuple=True)[0]:
                stack_disjunction_list = []
                for stack_dis in stack_layer_weights_t[conj].nonzero(as_tuple=True)[0]:
                    node = Node(features_names[stack_dis % len(features_names)],
                                t=len(and_layer_weights[dis]) - conj.item() - 1, index=stack_dis % len(features_names))
                    stack_disjunction_list.append(node)
                if len(stack_disjunction_list) > 0:
                    stack_disjunction = Disjunction(stack_disjunction_list,
                                                    t=(len(and_layer_weights[dis]) - conj.item() - 1))
                    conjunction_list.append(stack_disjunction)
            if len(conjunction_list) > 0:
                conjunction = Conjunction(conjunction_list)
                disjunction_list.append(conjunction)
        if len(disjunction_list) > 0:
            base_rule = Disjunction(disjunction_list)
        else:
            base_rule = Evaluated(0)
        if verbose:
            print('Base Rule:', base_rule, sep='\n')
        base_model_rules.append(base_rule)
    return base_model_rules


class LocalModel(nn.Module):

    def __init__(self, input_size, window_size, pad_border, max_sequence_length, base_model_hidden_size, conv_dim_out,
                 base_or_output_size, output_size):
        super(LocalModel, self).__init__()

        stacked_or_layer = StackedORConvLayer(input_size, window_size, pad_border, max_sequence_length, output_size=1)
        conv1d_and_layer = BaseModelANDConvLayer(window_size, base_model_hidden_size, conv_dim_out)
        conv1d_or_layer = BaseModelORConvLayer(base_model_hidden_size, base_or_output_size, conv_dim_out)
        conv_or_layer = ConvORLayer(input_size=conv1d_or_layer.output_size * conv_dim_out, output_size=output_size)

        self.model = nn.Sequential(OrderedDict([
            ('stack', stacked_or_layer),
            ('and', conv1d_and_layer),
            ('or', conv1d_or_layer),
            ("conv", conv_or_layer)
        ]))
        self.pad_border = pad_border
        self.max_sequence_length = max_sequence_length

    def forward(self, x):
        return self.model(x)

    def extract_rule(self, features_names, verbose=False):
        self.model.eval()
        stack_layer_weights = self.model[0].weight.get_binary_value()
        and_layer_weights = self.model[1].weight.get_binary_value().squeeze(-1)
        or_layer_weights = self.model[2].weight.get_binary_value().squeeze(-1)
        conv_or_layer_weights = self.model[3].weight.get_binary_value()

        window_size = self.model[0].weight.kernel_size

        stack_layer_weights_t = stack_layer_weights.permute(1, 0)
        and_layer_weights_t = and_layer_weights.permute(1, 0)
        or_layer_weights_t = or_layer_weights.permute(1, 0)
        conv_or_layer_weights_t = conv_or_layer_weights.permute(1, 0)

        base_model_rules = extract_base_model_rule(or_layer_weights, and_layer_weights, stack_layer_weights_t,
                                                   features_names, window_size, verbose=verbose)

        for conv_or_activated_mask_i in conv_or_layer_weights_t:
            rule_list = []
            nb_filters = len(base_model_rules)

            if self.pad_border:
                basic_shift_activated_mask = conv_or_activated_mask_i[
                                             nb_filters * window_size - 1:-nb_filters * window_size + 1]
                far_shift_activated_mask = conv_or_activated_mask_i[:nb_filters * window_size - 1]
                close_shift_activated_mask = conv_or_activated_mask_i[-nb_filters * window_size + 1:]

                # Far shift
                for rule in far_shift_activated_mask.nonzero(as_tuple=True)[0]:
                    shift = conv_or_layer_weights.shape[0] - (window_size - 1) - 1 - rule.item()
                    id_filter = rule % nb_filters
                    max_time_dependency = base_model_rules[id_filter].get_max_time_dependency() if hasattr(
                        base_model_rules[id_filter], "get_max_time_dependency") else None
                    if max_time_dependency is not None:
                        if self.max_sequence_length > shift + max_time_dependency:
                            tmp_model_rules = deepcopy(base_model_rules[id_filter])
                            if hasattr(tmp_model_rules, "incr_t"):
                                tmp_model_rules.incr_t(shift)
                                rule_list.append(tmp_model_rules)
                        else:
                            for el in base_model_rules[id_filter].element:
                                max_time_dependency = el.get_max_time_dependency() if hasattr(el,
                                                                                              "get_max_time_dependency") else None
                                if max_time_dependency is not None:
                                    if self.max_sequence_length > shift + max_time_dependency:
                                        tmp_model_rules = deepcopy(el)
                                        if hasattr(tmp_model_rules, "incr_t"):
                                            tmp_model_rules.incr_t(shift)
                                            rule_list.append(tmp_model_rules)
                # Basic shift
                for rule in basic_shift_activated_mask.nonzero(as_tuple=True)[0]:
                    id_filter = rule % len(base_model_rules)
                    tmp_model_rules = deepcopy(base_model_rules[id_filter])
                    shift = conv_or_layer_weights.shape[0] - (window_size - 1) * 2 - 1 - rule.item()
                    if hasattr(tmp_model_rules, "incr_t"):
                        tmp_model_rules.incr_t(shift)
                        rule_list.append(tmp_model_rules)

                # Close shift
                for rule in close_shift_activated_mask.nonzero(as_tuple=True)[0]:
                    shift = - (rule.item() + 1)
                    id_filter = rule % nb_filters
                    min_time_dependency = base_model_rules[id_filter].get_min_time_dependency() if hasattr(
                        base_model_rules[id_filter], "get_min_time_dependency") else None
                    if min_time_dependency is not None:
                        if shift + min_time_dependency >= 0:
                            tmp_model_rules = deepcopy(base_model_rules[id_filter])
                            if hasattr(tmp_model_rules, "incr_t"):
                                tmp_model_rules.incr_t(shift)
                                rule_list.append(tmp_model_rules)
                        else:
                            for el in base_model_rules[id_filter].element:
                                min_time_dependency = el.get_min_time_dependency() if hasattr(el,
                                                                                              "get_min_time_dependency") else None
                                if min_time_dependency is not None:
                                    if shift + min_time_dependency >= 0:
                                        tmp_model_rules = deepcopy(el)
                                        if hasattr(tmp_model_rules, "incr_t"):
                                            tmp_model_rules.incr_t(shift)
                                            rule_list.append(tmp_model_rules)

            else:
                # Basic shift
                for rule in conv_or_activated_mask_i.nonzero(as_tuple=True)[0]:
                    id_filter = rule % len(base_model_rules)
                    tmp_model_rules = deepcopy(base_model_rules[id_filter])
                    if hasattr(tmp_model_rules, "incr_t"):
                        shift = conv_or_layer_weights.shape[0] - 1 - rule.item()
                        tmp_model_rules.incr_t(shift)
                        rule_list.append(tmp_model_rules)
            if len(rule_list) > 0:
                final_rule = Disjunction(rule_list, separator='\n')
            else:
                final_rule = Evaluated(0)
        if verbose:
            print('\nFinal Rule:', final_rule, sep='\n')
        return final_rule


class GlobalModel(nn.Module):

    def __init__(self, input_size, window_size, pad_border, max_sequence_length, base_model_hidden_size, conv_dim_out,
                 base_or_output_size, output_size):
        super(GlobalModel, self).__init__()

        stacked_or_layer = StackedORConvLayer(input_size, window_size, pad_border, max_sequence_length, output_size=1)
        conv1d_and_layer = BaseModelANDConvLayer(window_size, base_model_hidden_size, conv_dim_out)
        conv1d_or_layer = BaseModelORConvLayer(base_model_hidden_size, base_or_output_size, conv_dim_out)
        fixed_conv_or_layer = FixedConvORLayer(input_size=conv1d_or_layer.output_size * conv_dim_out,
                                               output_size=output_size)

        self.model = nn.Sequential(OrderedDict([
            ('stack', stacked_or_layer),
            ('and', conv1d_and_layer),
            ('or', conv1d_or_layer),
            ("conv", fixed_conv_or_layer)
        ]))

    def forward(self, x):
        return self.model(x)

    def extract_rule(self, features_names, verbose=False):
        self.model.eval()
        stack_layer_weights = self.model[0].weight.get_binary_value()
        and_layer_weights = self.model[1].weight.get_binary_value().squeeze(-1)
        or_layer_weights = self.model[2].weight.get_binary_value().squeeze(-1)

        window_size = self.model[0].weight.kernel_size

        stack_layer_weights_t = stack_layer_weights.permute(1, 0)
        and_layer_weights_t = and_layer_weights.permute(1, 0)
        or_layer_weights_t = or_layer_weights.permute(1, 0)

        base_model_rules = extract_base_model_rule(or_layer_weights, and_layer_weights, stack_layer_weights_t,
                                                   features_names, window_size, verbose=verbose)

        for base_model_rule in base_model_rules:
            final_rule = base_model_rule.get_occurrence_predicate()
        if verbose:
            print('\nFinal Rule:', final_rule, sep='\n')
        return final_rule
