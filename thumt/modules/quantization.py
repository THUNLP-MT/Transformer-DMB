# coding=utf-8
# Author: Zhixing Tan
# Contact: playinf@stu.xmu.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numbers


def _check_type_and_shape(input, min, max):
    min_is_number = isinstance(min, numbers.Real)
    max_is_number = isinstance(max, numbers.Real)
    min_is_tensor = isinstance(min, torch.Tensor)
    max_is_tensor = isinstance(max, torch.Tensor)

    if min_is_tensor and max_is_tensor:
        min_ndim = min.dim()
        max_ndim = max.dim()

        if min_ndim > 1 or max_ndim > 1:
            raise ValueError("Unsupported dimension: min: %d, max: %d" %
                             (min_ndim, max_ndim))

        if min_ndim != max_ndim:
            raise ValueError("dim(min) != dim(max): %d vs %d" %
                             (min_ndim, max_ndim))

        if min_ndim == 1:
            if input.shape[-1] != min.shape[-1]:
                raise ValueError("Unmatched channels: %d vs %d" %
                                 (input.shape[-1], min.shape[-1]))
    elif not (max_is_number and min_is_number):
        raise ValueError("min and max must both be numbers or Tensors.")


def _choose_quantization_params(min, max):
    scale = (max - min) / 254.0
    initial_zero_point = 1.0 - min / scale

    if isinstance(initial_zero_point, torch.Tensor):
        nudged_zero_point = initial_zero_point.clamp_(1.0, 255.0).round_()
    else:
        if initial_zero_point > 255.0:
            nudged_zero_point = 255.0
        elif initial_zero_point < 1.0:
            nudged_zero_point = 1.0
        else:
            nudged_zero_point = round(initial_zero_point)

    return scale, nudged_zero_point


class FakeQuantWithMinMaxArgs(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, min, max):
        mask_min = input < min
        mask_max = input > max
        ctx.save_for_backward(mask_min, mask_max)

        output = input.clone()
        output[mask_min] = min
        output[mask_max] = max

        scale, zero_point = _choose_quantization_params(min, max)
        output.div_(scale).add_(zero_point).clamp_(1.0, 255.0).round_()
        output.sub_(zero_point).mul_(scale)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask_min, mask_max = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[mask_min] = 0.0
        grad_input[mask_max] = 0.0

        if ctx.needs_input_grad[1]:
            grad_min = grad_output[mask_min].sum()
        else:
            grad_min = None

        if ctx.needs_input_grad[2]:
            grad_max = grad_output[mask_max].sum()
        else:
            grad_max = None

        return grad_input, grad_min, grad_max


class FakeQuantWithMinMaxArgs1D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, min, max):
        _check_type_and_shape(input, min, max)
        mask_min = input < min
        mask_max = input > max
        ctx.save_for_backward(mask_min, mask_max)

        output = torch.where(mask_min, min, input)
        output = torch.where(mask_max, max, output)

        scale, zero_point = _choose_quantization_params(min, max)
        output.div_(scale).add_(zero_point).clamp_(1.0, 255.0).round_()
        output.sub_(zero_point).mul_(scale)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask_min, mask_max = ctx.saved_tensors

        zero_tensor = torch.zeros_like(grad_output)

        grad_hidden = torch.where(mask_min, zero_tensor, grad_output)
        grad_min = grad_output - grad_hidden
        grad_input = torch.where(mask_max, zero_tensor, grad_hidden)
        grad_max = grad_hidden - grad_input

        return grad_input, grad_min, grad_max


fake_quant_with_min_max_args = FakeQuantWithMinMaxArgs.apply
fake_quant_with_min_max_args_1d = FakeQuantWithMinMaxArgs1D.apply
