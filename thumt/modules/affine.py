# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thumt.utils as utils
import thumt.nn.op as ops
from thumt.modules.module import Module
from thumt.modules.quantization import fake_quant_with_min_max_args


class Affine(Module):

    def __init__(self, in_features, out_features, bias=True,
                 quantization=False, input_threshold=6.0, weight_threshold=1.0,
                 name="affine"):
        super(Affine, self).__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.quantization = quantization

        with utils.scope(name):
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.add_name(self.weight, "weight")
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_features))
                self.add_name(self.bias, "bias")
            else:
                self.register_parameter('bias', None)

            if quantization:
                self.input_range = torch.nn.Parameter(
                    torch.tensor(input_threshold))
                self.weight_range = torch.nn.Parameter(
                    torch.tensor(weight_threshold))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def zero_parameters(self):
        nn.init.zeros_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, shared=None):
        if shared:
            weight = self.weight + shared.weight

            if self.bias is not None and shared.bias is not None:
                bias = self.bias + shared.bias
            else:
                bias = None
        else:
            weight = self.weight
            bias = self.bias

        if self.quantization is False:
            return nn.functional.linear(input, weight, bias)
        else:
            input_range = self.input_range
            weight_range = self.weight_range

            input = fake_quant_with_min_max_args(
                input, -input_range, input_range)
            weight = fake_quant_with_min_max_args(
                weight, -weight_range, weight_range)

        return nn.functional.linear(input, weight,bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
