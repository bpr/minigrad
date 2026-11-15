"""Gradients against central finite differences, in float64."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false

from minigrad.dtype import DT
from minigrad.tensor import Tensor
from helpers import assert_close


def check_grad[
    F: def(Tensor) raises -> Tensor
](f: F, data: List[Float64], shape: List[Int], tol: Float64 = 1e-6) raises:
    """Compares `backward` on the scalar `f(x)` with finite differences
    at `x = data`."""
    var x = Tensor.from_list(data, shape, DT.float64).requires_grad_()
    f(x).backward()
    var got = x.grad().tolist()
    var eps = 1e-6
    for i in range(len(data)):
        var hi = data.copy()
        var lo = data.copy()
        hi[i] += eps
        lo[i] -= eps
        var fh = f(Tensor.from_list(hi, shape, DT.float64)).item()
        var fl = f(Tensor.from_list(lo, shape, DT.float64)).item()
        var num = (fh - fl) / (2 * eps)
        var d = got[i] - num
        if d < 0:
            d = -d
        var scale = num if num > 0 else -num
        if not (d <= tol * (1.0 + scale)):
            raise Error(
                t"d/dx[{i}]: backward gave {got[i]}, finite differences {num}"
            )


def _data() -> List[Float64]:
    return [0.3, -1.2, 2.5, 0.7, -0.4, 1.9]


def _pos() -> List[Float64]:
    return [0.3, 1.2, 2.5, 0.7, 0.4, 1.9]


def _w(shape: List[Int]) -> Tensor:
    """Fixed weights, so sums are not symmetric in their inputs."""
    var n = 1
    for s in shape:
        n *= s
    var d = List[Float64]()
    for i in range(n):
        d.append(0.5 + 0.37 * Float64((i * 7) % 5) - 0.11 * Float64(i))
    return Tensor.from_list(d, shape, DT.float64)


def f_sum(x: Tensor) raises -> Tensor:
    return (x * _w([2, 3])).sum()


def f_arith(x: Tensor) raises -> Tensor:
    return ((x * x - 2.0 * x + 1.0) / (x * x + 3.0) * _w([2, 3])).sum()


def f_exp_log(x: Tensor) raises -> Tensor:
    return ((x.exp() + x.log() * x.sqrt()) * _w([2, 3])).sum()


def f_trig(x: Tensor) raises -> Tensor:
    return ((x.sin() * x.cos() + x.tan()) * _w([2, 3])).sum()


def f_recip(x: Tensor) raises -> Tensor:
    return ((x.reciprocal() + x.rsqrt()) * _w([2, 3])).sum()


def f_pow_const(x: Tensor) raises -> Tensor:
    return ((x**1.7 + x**3 + x ** (-2)) * _w([2, 3])).sum()


def f_pow_tensor(x: Tensor) raises -> Tensor:
    return ((x ** _w([2, 3])) + (_w([2, 3]).abs() + 0.5) ** x).sum()


def f_relu(x: Tensor) raises -> Tensor:
    return (x.relu() * _w([2, 3])).sum()


def f_activations(x: Tensor) raises -> Tensor:
    return (
        (x.sigmoid() + x.tanh() + x.gelu() + x.silu() + x.leaky_relu(0.2))
        * _w([2, 3])
    ).sum()


def f_abs_max_min(x: Tensor) raises -> Tensor:
    var w = _w([2, 3])
    return ((x.abs() + x.maximum(w) - x.minimum(w * 0.5)) * w).sum()


def f_where(x: Tensor) raises -> Tensor:
    return (x.gt(0.5).where(x * x, x * 3.0) * _w([2, 3])).sum()


def f_broadcast(x: Tensor) raises -> Tensor:
    var row = _w([3])
    var col = _w([2, 1])
    return ((x + row) * col * x.reshape(2, 3, 1).expand(2, 3, 4).sum(2)).sum()


def f_reduce_axes(x: Tensor) raises -> Tensor:
    return (x.sum(0) * _w([3])).sum() + (x.sum(1) * _w([2])).sum() * 2.0


def f_mean_var(x: Tensor) raises -> Tensor:
    return (x.mean(1) * _w([2])).sum() + x.`var`(0).sum() + x.std()


def f_max(x: Tensor) raises -> Tensor:
    return (x.max(1) * _w([2])).sum() + x.max() + (x.min(0) * _w([3])).sum()


def f_movement(x: Tensor) raises -> Tensor:
    var a = x.T().reshape(2, 3) * _w([2, 3])
    var b = x.flip(1).pad([1, 1, 0, 2]) * _w([4, 5])
    var c = x.shrink([0, 2, 1, 3]) * _w([2, 2])
    return a.sum() + b.sum() + c.sum()


def f_cat_slice(x: Tensor) raises -> Tensor:
    var y = Tensor.cat([x, x * 2.0], 1)  # (2, 6)
    return (y.slice(1, 1, 5) * _w([2, 4])).sum() + (x[1] * _w([3])).sum()


def f_matmul(x: Tensor) raises -> Tensor:
    var a = x @ _w([3, 4])  # (2, 4)
    var b = x.T() @ x  # (3, 3)
    return (a * _w([2, 4])).sum() + (b * _w([3, 3])).sum()


def f_softmax(x: Tensor) raises -> Tensor:
    return (
        (x.softmax() * _w([2, 3])).sum()
        + (x.log_softmax(0) * _w([2, 3])).sum()
        + x.logsumexp(1).sum()
    )


def f_crossentropy(x: Tensor) raises -> Tensor:
    var y: List[Int] = [2, 0]
    return x.sparse_categorical_crossentropy(Tensor(y))


def f_layernorm(x: Tensor) raises -> Tensor:
    return (x.layernorm() * _w([2, 3])).sum()


def f_reuse(x: Tensor) raises -> Tensor:
    """The same subexpression used on several paths."""
    var y = x * x + 1.0
    var z = y.exp() * y + y.sum(1, keepdim=True)
    return (z * y).sum()


def f_detach(x: Tensor) raises -> Tensor:
    return (x * (x * x).detach()).sum()


def test_sum() raises:
    check_grad(f_sum, _data(), [2, 3])


def test_arith() raises:
    check_grad(f_arith, _data(), [2, 3])


def test_exp_log() raises:
    check_grad(f_exp_log, _pos(), [2, 3])


def test_trig() raises:
    check_grad(f_trig, _data(), [2, 3], 1e-5)


def test_recip() raises:
    check_grad(f_recip, _pos(), [2, 3])


def test_pow_const() raises:
    check_grad(f_pow_const, _pos(), [2, 3])


def test_pow_tensor() raises:
    check_grad(f_pow_tensor, _pos(), [2, 3])


def test_relu() raises:
    check_grad(f_relu, _data(), [2, 3])


def test_activations() raises:
    check_grad(f_activations, _data(), [2, 3])


def test_abs_max_min() raises:
    check_grad(f_abs_max_min, _data(), [2, 3])


def test_where() raises:
    check_grad(f_where, _data(), [2, 3])


def test_broadcast() raises:
    check_grad(f_broadcast, _data(), [2, 3])


def test_reduce_axes() raises:
    check_grad(f_reduce_axes, _data(), [2, 3])


def test_mean_var() raises:
    check_grad(f_mean_var, _data(), [2, 3])


def test_max() raises:
    check_grad(f_max, _data(), [2, 3])


def test_movement() raises:
    check_grad(f_movement, _data(), [2, 3])


def test_cat_slice() raises:
    check_grad(f_cat_slice, _data(), [2, 3])


def test_matmul() raises:
    check_grad(f_matmul, _data(), [2, 3])


def test_softmax() raises:
    check_grad(f_softmax, _data(), [2, 3])


def test_crossentropy() raises:
    check_grad(f_crossentropy, _data(), [2, 3])


def test_layernorm() raises:
    check_grad(f_layernorm, _data(), [2, 3], 1e-5)


def test_reuse() raises:
    check_grad(f_reuse, _data(), [2, 3])


def test_detach() raises:
    var x = Tensor.from_list(_data(), [2, 3], DT.float64).requires_grad_()
    f_detach(x).backward()
    var expect = List[Float64]()
    for v in _data():
        expect.append(v * v)
    assert_close(x.grad(), expect)


def test_max_ties_share() raises:
    var d: List[Float64] = [1, 3, 3, 2]
    var x = Tensor(d).requires_grad_()
    x.max().backward()
    assert_close(x.grad(), [0, 0.5, 0.5, 0])


def test_accumulate_and_zero() raises:
    var d: List[Float64] = [1, 2, 3]
    var x = Tensor(d).requires_grad_()
    (x * x).sum().backward()
    (x * 3.0).sum().backward()
    assert_close(x.grad(), [5, 7, 9])
    x.zero_grad()
    assert_false(x.has_grad())
    (x * 3.0).sum().backward()
    assert_close(x.grad(), [3, 3, 3])


def test_gradient_function() raises:
    var d: List[Float64] = [1, 2, 3]
    var x = Tensor(d)
    var y = Tensor(d) * 2.0
    var gs = (x * y).sum().gradient([x, y])
    assert_close(gs[0], [2, 4, 6])
    assert_close(gs[1], [1, 2, 3])


def test_second_derivative() raises:
    var d: List[Float64] = [1, 2, 3]
    var x = Tensor(d)
    var g = (x * x * x).sum().gradient([x])
    var gg = g[0].sum().gradient([x])
    assert_close(gg[0], [6, 12, 18])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
