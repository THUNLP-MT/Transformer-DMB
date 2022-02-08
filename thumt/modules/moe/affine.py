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


class Affine(Module):

    def __init__(self, in_features, out_features, bias=True, n=4,
                 name="affine"):
        super(Affine, self).__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.n = n

        with utils.scope(name):
            self.experts = nn.ModuleList([
                Linear(in_features, out_features, bias=bias,
                       name="affine_%d" % i)
                for i in range(n)])

    def forward(self, input, gates):
        input_shape = input.shape
        input = torch.reshape(input, [-1, input_shape[-1]])
        gates = torch.reshape(gates, [-1, gates.shape[-1]])
        batch_size = int(input.shape[0])

        part_sizes = list(torch.sum((gates > 0).long(), [0]))
        index = torch.nonzero(gates.t())

        cell_index, batch_index = torch.unbind(index, 1)
        input = torch.nn.functional.embedding(batch_index, input)
        inputs = torch.split(input, part_sizes, 0)
        results = []

        for i in range(self.n):
            if inputs[i].shape[0] != 0:
                results.append(self.experts[i](inputs[i]))
            else:
                results.append(torch.empty([0, self.out_features],
                                            device=input.device,
                                            dtype=input.dtype))

        # combine
        gate_index = batch_index * self.n + cell_index
        nonzero_gates = torch.gather(torch.reshape(gates, [-1]), 0, gate_index)
        stitched = torch.cat(results, dim=0)
        stitched = stitched * nonzero_gates[:, None]

        output = ops.unsorted_segment_sum_2d(stitched, batch_index, batch_size)

        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        output = torch.reshape(output, output_shape)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
