"""Neural network layers (`minigrad.nn`).

Layers are plain structs holding their parameter tensors. Each implements
`Module`, whose `parameters()` lists the tensors an optimizer trains; a
model built from layers implements it by concatenating its layers' lists
(see `collect`).
"""

from std.math import sqrt

from minigrad.dtype import DT
from minigrad.tensor import Tensor
from .optim import SGD, Adam, AdamW


trait Module:
    """Something with tensors an optimizer manages."""

    def parameters(self) -> List[Tensor]:
        """The tensors an optimizer should train, and any marked
        `requires_grad_(False)` that it should only realize each step."""
        ...


def collect[*Ms: Module](*modules: *Ms) -> List[Tensor]:
    """The parameters of several modules, in order.

    Args:
        modules: The modules.

    Returns:
        Every module's parameters, concatenated.
    """
    var out = List[Tensor]()
    comptime for i in range(modules.__len__()):
        for p in modules[i].parameters():
            out.append(p)
    return out^


struct Linear(Copyable, Module):
    """An affine map `y = x @ weight.T + bias`."""

    var weight: Tensor
    """Shape (out_features, in_features)."""
    var bias: Optional[Tensor]
    """Shape (out_features), or `None`."""

    def __init__(
        out self, in_features: Int, out_features: Int, bias: Bool = True
    ):
        """Initializes weights and bias uniformly in
        `±1 / sqrt(in_features)`, as tinygrad and PyTorch do.

        Args:
            in_features: The size of the last dimension of the input.
            out_features: The size of the last dimension of the output.
            bias: Whether to add a learned bias.
        """
        var bound = 1.0 / sqrt(Float64(in_features))
        self.weight = Tensor.uniform([out_features, in_features], -bound, bound)
        if bias:
            self.bias = Tensor.uniform([out_features], -bound, bound)
        else:
            self.bias = None

    def __call__(self, x: Tensor) raises -> Tensor:
        """Applies the layer to the last dimension of `x`."""
        return x.linear(self.weight.transpose(), self.bias)

    def parameters(self) -> List[Tensor]:
        var out = List[Tensor]()
        out.append(self.weight)
        if self.bias:
            out.append(self.bias.value())
        return out^


struct Conv2d(Copyable, Module):
    """A 2-dimensional convolution."""

    var weight: Tensor
    """Shape (out_channels, in_channels / groups, kernel, kernel)."""
    var bias: Optional[Tensor]
    var stride: Int
    var padding: Int
    var dilation: Int
    var groups: Int

    def __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = True,
    ):
        """Initializes weights and bias uniformly in
        `±1 / sqrt(in_channels / groups * kernel_size**2)`.

        Args:
            in_channels: The channels of the input.
            out_channels: The number of filters.
            kernel_size: The side of the square kernel.
            stride: The step between windows.
            padding: Zeros added on every side.
            dilation: The step between kernel elements.
            groups: The number of groups the channels are split into.
            bias: Whether to add a learned bias per filter.
        """
        var fan = in_channels // groups * kernel_size * kernel_size
        var bound = 1.0 / sqrt(Float64(fan))
        self.weight = Tensor.uniform(
            [out_channels, in_channels // groups, kernel_size, kernel_size],
            -bound,
            bound,
        )
        if bias:
            self.bias = Tensor.uniform([out_channels], -bound, bound)
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x: Tensor) raises -> Tensor:
        """Convolves `x`, of shape (batch, in_channels, height, width)."""
        return x.conv2d(
            self.weight,
            self.bias,
            self.groups,
            self.stride,
            self.dilation,
            self.padding,
        )

    def parameters(self) -> List[Tensor]:
        var out = List[Tensor]()
        out.append(self.weight)
        if self.bias:
            out.append(self.bias.value())
        return out^


struct BatchNorm(Copyable, Module):
    """Batch normalization over every dimension but 1 (the channels).

    In training mode (`Tensor.train()`) it normalizes by the batch's
    statistics and updates running estimates of them; otherwise it
    normalizes by the running estimates.

    The running estimates are updated lazily, like everything else, and
    are listed by `parameters()` marked `requires_grad_(False)`, so an
    optimizer's `step` realizes them. Without an optimizer, realize
    `running_mean` and `running_var` after each training call, or each
    call stacks another update on the last. (They must not be realized
    inside the call: that would cut the gradient through the batch
    statistics.)
    """

    var weight: Tensor
    var bias: Tensor
    var running_mean: Tensor
    var running_var: Tensor
    var eps: Float64
    var momentum: Float64

    def __init__(
        out self, features: Int, eps: Float64 = 1e-5, momentum: Float64 = 0.1
    ) raises:
        """Starts with scale 1, shift 0, mean 0, and variance 1.

        Args:
            features: The number of channels.
            eps: Added to the variance for stability.
            momentum: The weight of a new batch in the running estimates.
        """
        self.weight = Tensor.ones(features).contiguous()
        self.bias = Tensor.zeros(features).contiguous()
        self.running_mean = Tensor.zeros(features).contiguous().realize()
        self.running_var = Tensor.ones(features).contiguous().realize()
        _ = self.running_mean.requires_grad_(False)
        _ = self.running_var.requires_grad_(False)
        self.eps = eps
        self.momentum = momentum

    def __call__(self, x: Tensor) raises -> Tensor:
        """Normalizes `x`, of shape (batch, features, ...)."""
        if not Tensor.training():
            return x.batchnorm(
                self.weight,
                self.bias,
                self.running_mean,
                (self.running_var + self.eps).rsqrt(),
            )
        var n = x.ndim()
        var shape = List[Int](length=n, fill=1)
        shape[1] = x.shape()[1]
        var axes = List[Int]()
        for d in range(n):
            if d != 1:
                axes.append(d)
        var count = x.numel() // x.shape()[1]
        var mean = x._reduce(0, axes, False) * (1.0 / Float64(count))
        # d(var)/d(mean) is zero: the mean is detached, as in tinygrad.
        var y = x - mean.detach().reshape(shape)
        var variance = (y * y)._reduce(0, axes, False) * (1.0 / Float64(count))
        var unbiased = Float64(count) / Float64(max(1, count - 1))
        _ = self.running_mean.assign(
            self.running_mean * (1.0 - self.momentum)
            + mean.detach() * self.momentum
        )
        _ = self.running_var.assign(
            self.running_var * (1.0 - self.momentum)
            + variance.detach() * (self.momentum * unbiased)
        )
        return x.batchnorm(
            self.weight, self.bias, mean, (variance + self.eps).rsqrt()
        )

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias, self.running_mean, self.running_var]


struct LayerNorm(Copyable, Module):
    """Normalization over the last dimension with a learned scale and
    shift."""

    var weight: Tensor
    var bias: Tensor
    var eps: Float64

    def __init__(out self, features: Int, eps: Float64 = 1e-5):
        """Starts as the plain normalization: scale 1, shift 0.

        Args:
            features: The size of the last dimension.
            eps: Added to the variance for stability.
        """
        self.weight = Tensor.ones(features).contiguous()
        self.bias = Tensor.zeros(features).contiguous()
        self.eps = eps

    def __call__(self, x: Tensor) raises -> Tensor:
        """Normalizes the last dimension of `x`."""
        return x.layernorm(-1, self.eps) * self.weight + self.bias

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]
