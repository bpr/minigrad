"""Layers, as in `tinygrad.nn`. Optimizers are in `nn.optim`, parameter
and weight-file helpers in `nn.state`."""

from __future__ import annotations

import math

from ..dtype import dtypes
from ..tensor import Tensor
from . import datasets, optim, state


class Linear:
    """`x @ weight.T + bias`, initialized uniformly in
    `±1 / sqrt(in_features)` as tinygrad does."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        bound = 1 / math.sqrt(in_features)
        self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
        self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        return x.linear(self.weight.transpose(), self.bias)


class Conv2d:
    """A 2-d convolution with square kernels and equal strides, padding
    and dilation on both axes."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        if not isinstance(kernel_size, int) and len(set(kernel_size)) != 1:
            raise NotImplementedError("non-square kernels")
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        bound = 1 / math.sqrt(in_channels // groups * k * k)
        self.weight = Tensor.uniform(out_channels, in_channels // groups, k, k, low=-bound, high=bound)
        self.bias = Tensor.uniform(out_channels, low=-bound, high=bound) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        return x.conv2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding)


class BatchNorm:
    """Batch normalization over dimension 1. In training mode
    (`Tensor.train()`) it normalizes by the batch's statistics and
    updates the running estimates lazily; an optimizer's `step` realizes
    them (they are parameters with `requires_grad = False`)."""

    def __init__(self, sz: int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
        self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum
        self.weight = Tensor.ones(sz).contiguous() if affine else None
        self.bias = Tensor.zeros(sz).contiguous() if affine else None
        if track_running_stats:
            self.running_mean = Tensor.zeros(sz).contiguous().realize().requires_grad_(False)
            self.running_var = Tensor.ones(sz).contiguous().realize().requires_grad_(False)

    def __call__(self, x: Tensor) -> Tensor:
        shape = [1, -1] + [1] * (x.ndim - 2)
        if self.track_running_stats and not Tensor.training:
            mean, var = self.running_mean, self.running_var
        else:
            axes = tuple(d for d in range(x.ndim) if d != 1)
            mean = x.mean(axis=axes)
            y = x - mean.detach().reshape(shape)  # d(var)/d(mean) is zero
            var = (y * y).mean(axis=axes)
            if self.track_running_stats:
                n = x.numel() / x.shape[1]
                m = self.momentum
                self.running_mean.assign(self.running_mean * (1 - m) + mean.detach() * m)
                self.running_var.assign(self.running_var * (1 - m) + var.detach() * (m * n / max(1, n - 1)))
        w = self.weight if self.weight is not None else Tensor.ones(x.shape[1])
        b = self.bias if self.bias is not None else Tensor.zeros(x.shape[1])
        return x.batchnorm(w, b, mean, (var + self.eps).rsqrt())


BatchNorm2d = BatchNorm3d = BatchNorm


class LayerNorm:
    """Normalization over the last dimension, with a learned scale and
    shift."""

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        if not isinstance(normalized_shape, int):
            if len(normalized_shape) != 1:
                raise NotImplementedError("LayerNorm over more than the last dimension")
            normalized_shape = normalized_shape[0]
        self.eps = eps
        self.weight = Tensor.ones(normalized_shape).contiguous() if elementwise_affine else None
        self.bias = Tensor.zeros(normalized_shape).contiguous() if elementwise_affine else None

    def __call__(self, x: Tensor) -> Tensor:
        x = x.layernorm(-1, self.eps)
        if self.weight is None:
            return x
        return x * self.weight + self.bias


class RMSNorm:
    """`x / sqrt(mean(x²) + eps)` over the last dimension, scaled."""

    def __init__(self, dim: int, eps=1e-6, elementwise_affine=True):
        self.eps = eps
        self.weight = Tensor.ones(dim).contiguous() if elementwise_affine else None

    def __call__(self, x: Tensor) -> Tensor:
        x = x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt()
        return x if self.weight is None else x * self.weight


class Embedding:
    """A lookup table of `vocab_size` rows, as a one-hot product (so its
    gradient reaches the table)."""

    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_sz = vocab_size
        self.weight = Tensor.glorot_uniform(vocab_size, embed_size)

    def __call__(self, idx: Tensor) -> Tensor:
        if not dtypes.is_int(idx.dtype):
            raise TypeError(f"Expected integer dtype for index in embedding, got {idx.dtype}")
        return idx.one_hot(self.vocab_sz).float() @ self.weight


__all__ = ["Linear", "Conv2d", "BatchNorm", "BatchNorm2d", "BatchNorm3d",
           "LayerNorm", "RMSNorm", "Embedding", "datasets", "optim", "state"]
