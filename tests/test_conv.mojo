"""Sliding windows, convolution, and pooling against direct loops."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false

from minigrad.dtype import DT
from minigrad.graph import graph
from minigrad.nn import Conv2d, BatchNorm
from minigrad.tensor import Tensor
from helpers import assert_close, assert_shape


def _seq(shape: List[Int], dtype: DT = DT.float32) -> Tensor:
    """Distinct, non-monotone values."""
    var n = 1
    for s in shape:
        n *= s
    var d = List[Float64]()
    for i in range(n):
        d.append(Float64((i * 37) % 23) * 0.25 - 2.0 + 0.01 * Float64(i))
    return Tensor.from_list(d, shape, dtype)


def conv_reference(
    x: List[Float64],
    xs: List[Int],
    w: List[Float64],
    ws: List[Int],
    groups: Int,
    stride: Int,
    dilation: Int,
    padding: Int,
) -> List[Float64]:
    """A 2-d convolution by its definition."""
    var bs = xs[0]
    var h = xs[2]
    var wd = xs[3]
    var cout = ws[0]
    var cin = ws[1]
    var kh = ws[2]
    var kw = ws[3]
    var oh = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    var ow = (wd + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    var rcout = cout // groups
    var out = List[Float64]()
    for b in range(bs):
        for co in range(cout):
            var grp = co // rcout
            for oy in range(oh):
                for ox in range(ow):
                    var s = 0.0
                    for ci in range(cin):
                        for ky in range(kh):
                            for kx in range(kw):
                                var iy = oy * stride + ky * dilation - padding
                                var ix = ox * stride + kx * dilation - padding
                                if iy < 0 or iy >= h or ix < 0 or ix >= wd:
                                    continue
                                var c = grp * cin + ci
                                s += x[((b * xs[1] + c) * h + iy) * wd + ix] * (
                                    w[((co * cin + ci) * kh + ky) * kw + kx]
                                )
                    out.append(s)
    return out^


def check_conv(
    xs: List[Int],
    ws: List[Int],
    groups: Int,
    stride: Int,
    dilation: Int,
    padding: Int,
) raises:
    var x = _seq(xs)
    var w = _seq(ws)
    var got = x.conv2d(w, None, groups, stride, dilation, padding)
    var expect = conv_reference(
        x.tolist(), xs, w.tolist(), ws, groups, stride, dilation, padding
    )
    assert_close(got, expect, 1e-3)


def test_conv_basic() raises:
    check_conv([2, 3, 6, 5], [4, 3, 3, 3], 1, 1, 1, 0)


def test_conv_padding() raises:
    check_conv([2, 3, 6, 5], [4, 3, 3, 3], 1, 1, 1, 1)
    check_conv([1, 2, 5, 5], [3, 2, 5, 5], 1, 1, 1, 2)


def test_conv_stride() raises:
    check_conv([2, 3, 7, 8], [4, 3, 3, 3], 1, 2, 1, 0)
    check_conv([2, 3, 7, 8], [4, 3, 3, 3], 1, 2, 1, 1)
    check_conv([1, 1, 9, 9], [2, 1, 2, 2], 1, 3, 1, 0)


def test_conv_dilation() raises:
    check_conv([1, 2, 9, 8], [3, 2, 3, 3], 1, 1, 2, 0)
    check_conv([1, 2, 9, 8], [3, 2, 3, 3], 1, 2, 2, 2)


def test_conv_groups() raises:
    check_conv([2, 4, 5, 5], [6, 2, 3, 3], 2, 1, 1, 1)
    check_conv([1, 3, 5, 5], [3, 1, 3, 3], 3, 1, 1, 0)


def test_conv_1x1() raises:
    check_conv([2, 3, 4, 4], [5, 3, 1, 1], 1, 1, 1, 0)


def test_conv_bias_and_layer() raises:
    var x = _seq([1, 1, 3, 3])
    var w = Tensor.ones(2, 1, 2, 2)
    var b: List[Float64] = [10, 20]
    var y = x.conv2d(w, Tensor(b))
    assert_shape(y, [1, 2, 2, 2])
    var nob = x.conv2d(w).tolist()
    assert_close(
        y,
        [
            nob[0] + 10,
            nob[1] + 10,
            nob[2] + 10,
            nob[3] + 10,
            nob[4] + 20,
            nob[5] + 20,
            nob[6] + 20,
            nob[7] + 20,
        ],
    )
    var layer = Conv2d(3, 8, 3, padding=1)
    assert_shape(layer(Tensor.rand(2, 3, 6, 6)), [2, 8, 6, 6])
    assert_equal(len(layer.parameters()), 2)


def test_conv_uses_gemm() raises:
    """The convolution kernel is recognized as a contraction."""
    var x = _seq([2, 3, 6, 5])
    var w = _seq([4, 3, 3, 3])
    var before = graph()[].stats.gemm_kernels
    _ = x.conv2d(w).realize()
    assert_equal(graph()[].stats.gemm_kernels, before + 1)


def test_pool() raises:
    var d: List[Float64] = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ]
    var x = Tensor.from_list(d, [1, 1, 4, 4])
    assert_close(x.max_pool2d(), [6, 8, 14, 16])
    assert_close(x.avg_pool2d(), [3.5, 5.5, 11.5, 13.5])
    assert_close(x.max_pool2d(2, 1), [6, 7, 8, 10, 11, 12, 14, 15, 16])
    assert_close(x.max_pool2d(3, 1), [11, 12, 15, 16])
    assert_close(x.avg_pool2d(3, 1), [6, 7, 10, 11])
    # A 5x5 input pooled by 2 drops the last row and column.
    assert_shape(_seq([1, 1, 5, 5]).max_pool2d(), [1, 1, 2, 2])
    assert_shape(_seq([2, 3, 7, 6]).max_pool2d(3, 2), [2, 3, 3, 2])


def check_grad[
    F: def(Tensor) raises -> Tensor
](f: F, shape: List[Int], tol: Float64 = 1e-6) raises:
    var x0 = _seq(shape, DT.float64)
    var data = x0.tolist()
    var x = Tensor.from_list(data, shape, DT.float64).requires_grad_()
    f(x).backward()
    var got = x.grad().tolist()
    var eps = 1e-6
    for i in range(len(data)):
        var hi = data.copy()
        var lo = data.copy()
        hi[i] += eps
        lo[i] -= eps
        var num = (
            f(Tensor.from_list(hi, shape, DT.float64)).item()
            - f(Tensor.from_list(lo, shape, DT.float64)).item()
        ) / (2 * eps)
        var d = got[i] - num
        if d < 0:
            d = -d
        var scale = num if num > 0 else -num
        if not (d <= tol * (1.0 + scale)):
            raise Error(
                t"d/dx[{i}]: backward gave {got[i]}, finite differences {num}"
            )


def f_conv_x(x: Tensor) raises -> Tensor:
    var w = _seq([3, 2, 3, 3], DT.float64)
    return (
        x.conv2d(w, None, 1, 1, 1, 1) * _seq([1, 3, 5, 4], DT.float64)
    ).sum()


def f_conv_w(w: Tensor) raises -> Tensor:
    var x = _seq([2, 2, 5, 4], DT.float64)
    return (
        x.conv2d(w, None, 1, 1, 1, 0) * _seq([2, 3, 3, 2], DT.float64)
    ).sum()


def f_conv_strided_x(x: Tensor) raises -> Tensor:
    var w = _seq([2, 2, 3, 3], DT.float64)
    return (
        x.conv2d(w, None, 1, 2, 1, 1) * _seq([1, 2, 3, 3], DT.float64)
    ).sum()


def f_conv_dilated_groups_x(x: Tensor) raises -> Tensor:
    var w = _seq([4, 1, 2, 2], DT.float64)
    return (
        x.conv2d(w, None, 2, 1, 2, 0) * _seq([1, 4, 4, 3], DT.float64)
    ).sum()


def f_maxpool(x: Tensor) raises -> Tensor:
    return (x.max_pool2d() * _seq([1, 2, 2, 2], DT.float64)).sum()


def f_maxpool_overlap(x: Tensor) raises -> Tensor:
    return (x.max_pool2d(3, 2) * _seq([1, 2, 2, 2], DT.float64)).sum()


def f_avgpool(x: Tensor) raises -> Tensor:
    return (x.avg_pool2d(2, 1) * _seq([1, 2, 3, 4], DT.float64)).sum()


def test_grad_conv_input() raises:
    check_grad(f_conv_x, [1, 2, 5, 4])


def test_grad_conv_weight() raises:
    check_grad(f_conv_w, [3, 2, 3, 3])


def test_grad_conv_strided() raises:
    check_grad(f_conv_strided_x, [1, 2, 6, 5])


def test_grad_conv_dilated_groups() raises:
    check_grad(f_conv_dilated_groups_x, [1, 2, 6, 5])


def test_grad_pool() raises:
    check_grad(f_maxpool, [1, 2, 4, 5])
    check_grad(f_maxpool_overlap, [1, 2, 5, 6])
    check_grad(f_avgpool, [1, 2, 4, 5])


def test_batchnorm() raises:
    var bn = BatchNorm(3)
    var x = _seq([4, 3, 2, 2])
    assert_equal(len(bn.parameters()), 4)
    Tensor.train(True)
    var y = bn(x)
    # Per channel: zero mean and unit variance over batch and space.
    assert_close(y.mean(0, 2, 3), [0, 0, 0], 1e-4)
    assert_close((y * y).mean(0, 2, 3), [1, 1, 1], 1e-3)
    var rm = bn.running_mean.tolist()
    var xm = x.mean(0, 2, 3).tolist()
    assert_close(bn.running_mean, [0.1 * xm[0], 0.1 * xm[1], 0.1 * xm[2]], 1e-5)
    Tensor.train(False)
    # In eval mode the running estimates are used, so a constant input
    # is not normalized to zero.
    var z = bn(Tensor.ones(1, 3, 1, 1))
    assert_close(
        z,
        [
            (1.0 - rm[0]) / bn.running_var.tolist()[0] ** 0.5,
            (1.0 - rm[1]) / bn.running_var.tolist()[1] ** 0.5,
            (1.0 - rm[2]) / bn.running_var.tolist()[2] ** 0.5,
        ],
        1e-3,
    )


def f_batchnorm_layer(x: Tensor) raises -> Tensor:
    """Through the layer in training mode: the batch statistics depend
    on the input, and the gradient must flow through them."""
    Tensor.train(True)
    var bn = BatchNorm(2)
    bn.weight = _seq([2], DT.float64)
    bn.bias = _seq([2], DT.float64) * 0.5
    bn.running_mean = bn.running_mean.cast(DT.float64)
    bn.running_var = bn.running_var.cast(DT.float64)
    var out = (bn(x) * _seq([3, 2, 2, 2], DT.float64)).sum()
    Tensor.train(False)
    return out


def test_grad_batchnorm_layer() raises:
    check_grad(f_batchnorm_layer, [3, 2, 2, 2], 1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
