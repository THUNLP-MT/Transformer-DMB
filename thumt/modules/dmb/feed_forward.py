# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import thumt.utils as utils

from thumt.modules.module import Module
from thumt.modules.dmb.affine import Affine


class FeedForward(Module):

    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0,
                 n=1, shared_private=False, quantization=False,
                 input_threshold=6.0, weight_threshold=1.0,
                 name="feed_forward"):
        super(FeedForward, self).__init__(name=name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.dropout = dropout
        self.n = n

        with utils.scope(name):
            self.input_transform = Affine(input_size, hidden_size, n=n,
                                          shared_private=shared_private,
                                          quantization=quantization,
                                          input_threshold=input_threshold,
                                          weight_threshold=weight_threshold,
                                          name="input_transform")
            self.output_transform = Affine(hidden_size, self.output_size, n=n,
                                           shared_private=shared_private,
                                           quantization=quantization,
                                           input_threshold=input_threshold,
                                           weight_threshold=weight_threshold,
                                           name="output_transform")

        self.reset_parameters()

    def forward(self, x, partitions):
        x = self.input_transform(x, partitions)

        h = nn.functional.relu(x)
        h = nn.functional.dropout(h, self.dropout, self.training)

        return self.output_transform(h, partitions)

    def reset_parameters(self):
        if self.n == 1:
            nn.init.xavier_uniform_(self.input_transform.weight)
            nn.init.xavier_uniform_(self.output_transform.weight)
            nn.init.constant_(self.input_transform.bias, 0.0)
            nn.init.constant_(self.output_transform.bias, 0.0)
        else:
            for i in range(self.n):
                nn.init.xavier_uniform_(
                    self.input_transform.branches[i].weight)
                nn.init.xavier_uniform_(
                    self.output_transform.branches[i].weight)
                nn.init.constant_(
                    self.input_transform.branches[i].bias, 0.0)
                nn.init.constant_(
                    self.output_transform.branches[i].bias, 0.0)
