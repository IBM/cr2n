#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
from abc import abstractmethod
import numpy as np


class AnyEvaluated:
    def __init__(self, t):
        self.t = t
        self.ignore_t = False

    def __str__(self):
        any_str = "*" + ' at t-' + str(self.t)
        if self.ignore_t:
            any_str = "*"
        return any_str

    def __call__(self, x=None):
        if self.t is not None and len(x.shape) == 2:
            x = np.transpose(x)
            if self.t < len(x):
                val_condition = 1
            else:
                return 0
        else:
            val_condition = 1
        return val_condition

    def incr_t(self, value=1):
        self.t = self.t + value

    def set_ignore_t(self, value):
        self.ignore_t = value

    def get_max_time_dependency(self):
        return self.t

    def get_min_time_dependency(self):
        return self.t


class Evaluated:
    def __init__(self, value):
        self.value = value  # 0, 1

    def __str__(self):
        predicate_str = ''
        if self.value == 1:
            predicate_str = 'TRUE'
        if self.value == 0:
            predicate_str = 'FALSE'
        return predicate_str

    def __call__(self, x=None, first_triggered=False):
        evaluation = self.value
        if first_triggered:
            triggered = self.value if self.value else None
            return evaluation, triggered
        return evaluation

    def batch_apply_rule(self, xs):
        test = list(map(lambda x: self.__call__(x, first_triggered=True), xs))
        res = list(map(list, zip(*test)))
        return res[0], res[1]

    def get_occurrence_predicate(self):
        return self


class Node:
    def __init__(self, name, index, t):
        self.name = name
        self.index = index
        self.t = t
        self.ignore_t = False

    def __str__(self):
        node_str = self.name + ' at t-' + str(self.t)
        if self.ignore_t:
            node_str = self.name
        return node_str

    def __call__(self, x):
        if self.t is not None and len(x.shape) == 2:
            x = np.transpose(x)
            if self.t < len(x):
                val_condition = x[-self.t - 1][self.index]
                # val_condition = x[self.index][-self.t - 1]
            else:
                return 0
        else:
            val_condition = x[self.index]
        return val_condition

    def incr_t(self, value=1):
        self.t = self.t + value

    def set_ignore_t(self, value):
        self.ignore_t = value


class LogicalOperator:
    def __init__(self, element):
        self.element = element

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    def __len__(self):
        return len(self.element)

    def incr_t(self, value=1):
        for el in self.element:
            if hasattr(el, "incr_t"):
                el.incr_t(value)
            elif hasattr(el, "t"):
                el.t = el.t + value

    def set_ignore_t(self, value):
        for el in self.element:
            if hasattr(el, "set_ignore_t"):
                el.set_ignore_t(value)

    def get_max_time_dependency(self):
        t_values = []
        for el in self.element:
            value = None
            if hasattr(el, "t"):
                value = el.t
            elif hasattr(el, "get_max_time_dependency"):
                value = el.get_max_time_dependency()
            if value is not None:
                t_values.append(value)
        return max(t_values, default=None)

    def get_min_time_dependency(self):
        t_values = []
        for el in self.element:
            value = None
            if hasattr(el, "t"):
                value = el.t
            elif hasattr(el, "get_min_time_dependency"):
                value = el.get_min_time_dependency()
            if value is not None:
                t_values.append(value)
        return min(t_values, default=None)


class Disjunction(LogicalOperator):
    def __init__(self, element, t=None, separator=' '):
        super(Disjunction, self).__init__(element=element)
        self.separator = separator
        self.t = t

    def __str__(self):
        dis_str = (self.separator + 'OR' + self.separator).join([str(element) for element in self.element])
        if len(self.element) > 1:
            dis_str = '(' + dis_str + ')'
        return dis_str

    def __call__(self, x, first_triggered=False):
        elements_evaluation = [element(x) for element in self.element]
        evaluation = any(elements_evaluation)
        if first_triggered:
            first_trig_disjunction = elements_evaluation.index(True) if True in elements_evaluation else None
            return evaluation, first_trig_disjunction
        return evaluation

    def get_occurrence_predicate(self):
        elements = []
        for element in self.element:
            elements.append(element.get_occurrence_predicate())
        return Disjunction(elements, separator='\n')

    def batch_apply_rule(self, xs):
        test = list(map(lambda x: self.__call__(x, first_triggered=True), xs))
        res = list(map(list, zip(*test)))
        return res[0], res[1]


class Conjunction(LogicalOperator):
    def __init__(self, element):
        super(Conjunction, self).__init__(element=element)

    def __str__(self):
        conj_str = ' AND '.join([str(element) for element in self.element])
        if len(self.element) > 1:
            conj_str = '(' + conj_str + ')'
        return conj_str

    def __call__(self, x):
        evaluation = all([element(x) for element in self.element])
        return evaluation

    def get_occurrence_predicate(self):
        main = dict()
        for el in self.element:
            if hasattr(el, 't') and len(el.element) > 0:
                if el.t not in main:
                    main[el.t] = [el]
                else:
                    main[el.t].append(el)

        t_values = sorted(main)
        new_element = []

        if len(t_values) > 0:
            for t in range(t_values[-1], t_values[0] - 1, -1):
                if t in main:
                    new_element.append(Conjunction(main[t]))
                else:
                    new_element.append(AnyEvaluated(t))
            return OccurrencePredicate(new_element)
        else:
            return self


class OccurrencePredicate:
    """

    OccurencePredicate is an ordered pattern
    last element refers to the latest observation
    """

    def __init__(self, element):
        self.element = element
        for el in element:
            if hasattr(el, "set_ignore_t"):
                el.set_ignore_t(True)

    def __str__(self):
        predicate_str = '-'.join([str(element) for element in self.element]) + ' in sequence'
        return predicate_str

    def __call__(self, x):
        x = np.transpose(x)
        m, n = len(x), len(self.element)
        if m >= n:
            idx = [np.s_[i:m - n + 1 + i] for i in range(n)]

            val = [np.array(list(map(self.element[i], x[idx[i]]))) for i in range(n)]

            val_condition = np.any(np.all(val, axis=0))
        else:
            val_condition = False
        return val_condition
