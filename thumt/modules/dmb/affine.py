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
from thumt.modules.affine import Affine as Linear
from thumt.modules.quantization import fake_quant_with_min_max_args


class Affine(Module):

    def __init__(self, in_features, out_features, bias=True, n=1,
                 shared_private=False, quantization=False, input_threshold=6.0,
                 weight_threshold=1.0,
                 name="affine"):
        super(Affine, self).__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.quantization = quantization
        self.n = n

        with utils.scope(name):
            if n == 1:
                self.weight = nn.Parameter(
                    torch.Tensor(out_features, in_features))
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
            else:
                self.branches = nn.ModuleList([
                    Linear(in_features, out_features, bias=bias,
                            quantization=quantization,
                            input_threshold=input_threshold,
                            weight_threshold=weight_threshold,
                            name="affine_%d" % i)
                    for i in range(n)])
                if shared_private:
                    self.shared_branch = Linear(in_features, out_features,
                            bias=bias,
                            quantization=quantization,
                            input_threshold=input_threshold,
                            weight_threshold=weight_threshold,
                            name="shared_affine")
                    self.shared_branch.zero_parameters()
                else:
                    self.shared_branch = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, partitions=None):
        if self.n == 1:
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
        else:
            # input: [B, input_size]
            # partition: [B]
            #print(self.name, "input", input.shape)
            input_shape = input.shape
            input = torch.reshape(input, [-1, input_shape[-1]])
            partitions = torch.reshape(partitions, [-1])
            indices = torch.arange(0, input.shape[0], dtype=torch.int64)

            outputs = ops.dynamic_partition(input, partitions, self.n)
            indices = ops.dynamic_partition(indices, partitions, self.n)
            results = []

            for i in range(self.n):
                if outputs[i].shape[0] != 0:
                    results.append(self.branches[i](outputs[i],
                                                    shared=self.shared_branch))
                else:
                    results.append(torch.empty([0, self.out_features],
                                               dtype=input.dtype))

            result = ops.dynamic_stitch(indices, results)
            output_shape = list(input_shape[:-1]) + [result.shape[-1]]
            result = torch.reshape(result, output_shape)

            #print(self.name, "result", result.shape)
            assert result.shape[0] == input_shape[0]

            return result

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
