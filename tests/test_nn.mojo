"""Layers and optimizers: small problems with known answers."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false

from minigrad.graph import graph
from minigrad.nn import (
    Linear,
    LayerNorm,
    SGD,
    Adam,
    AdamW,
    collect,
    quantize_int8,
)
from minigrad.dtype import DT
from minigrad.tensor import Tensor
from helpers import assert_close, assert_shape


def test_linear_shapes() raises:
    var l = Linear(3, 5)
    assert_shape(l.weight, [5, 3])
    assert_shape(l(Tensor.rand(4, 3)), [4, 5])
    assert_shape(l(Tensor.rand(2, 4, 3)), [2, 4, 5])
    assert_equal(len(l.parameters()), 2)
    assert_equal(len(Linear(3, 5, bias=False).parameters()), 1)
    assert_equal(len(collect(l, LayerNorm(5))), 4)


def test_sgd_quadratic() raises:
    # Minimize (w - 3)^2: after one step with lr 0.25, w = w - 0.5 (w - 3).
    var d: List[Float64] = [0, 1, 5]
    var w = Tensor(d)
    var opt = SGD([w], lr=0.25)
    for _ in range(2):
        opt.zero_grad()
        ((w - 3.0) ** 2).sum().backward()
        opt.step()
    assert_close(w, [2.25, 2.5, 3.5])


def test_sgd_momentum() raises:
    var d: List[Float64] = [1.0]
    var w = Tensor(d)
    var opt = SGD([w], lr=0.1, momentum=0.9)
    # Gradient of 2w is 2 at w = 1: b = 2, w = 0.8; then b = 1.8 + 1.6.
    for _ in range(2):
        opt.zero_grad()
        (w * w).sum().backward()
        opt.step()
    assert_close(w, [0.8 - 0.1 * (0.9 * 2.0 + 1.6)])


def test_adam_first_step() raises:
    # The first Adam step moves each weight by lr against its gradient's
    # sign, whatever the gradient's size.
    var d: List[Float64] = [1.0, -2.0, 0.5]
    var w = Tensor(d)
    var opt = Adam([w], lr=0.01)
    opt.zero_grad()
    (w * w * w).sum().backward()
    opt.step()
    assert_close(w, [0.99, -2.01, 0.49], 1e-5)


def test_adamw_decay() raises:
    var d: List[Float64] = [1.0]
    var w = Tensor(d)
    var opt = AdamW([w], lr=0.1, weight_decay=0.5)
    opt.zero_grad()
    (w * 2.0).sum().backward()
    opt.step()
    # Step of lr * (1 + wd * w).
    assert_close(w, [1.0 - 0.1 * 1.5], 1e-5)


def test_xor() raises:
    Tensor.manual_seed(7)
    var xs: List[List[Float64]] = [[0, 0], [0, 1], [1, 0], [1, 1]]
    var x = Tensor(xs)
    var ys: List[Float64] = [0, 1, 1, 0]
    var y = Tensor(ys).reshape(4, 1)
    var l1 = Linear(2, 8)
    var l2 = Linear(8, 1)
    var opt = Adam(collect(l1, l2), lr=0.05)
    var loss = 1.0
    for _ in range(300):
        opt.zero_grad()
        var out = l2(l1(x).tanh()).sigmoid()
        var l = ((out - y) ** 2).mean()
        l.backward()
        opt.step()
        loss = l.item()
    assert_true(loss < 0.01)
    assert_close(l2(l1(x).tanh()).sigmoid().gt(0.5).float(), [0, 1, 1, 0])


def test_no_leak_across_steps() raises:
    """A training loop must not grow the graph or the buffer table."""
    Tensor.manual_seed(1)
    var x = Tensor.rand(8, 4)
    var y = Tensor.rand(8, 2)
    var l1 = Linear(4, 6)
    var l2 = Linear(6, 2)
    var opt = SGD(collect(l1, l2), lr=0.01, momentum=0.9)
    # Sampled inside the loop: Mojo destroys the model and optimizer
    # right after their last use, so nothing is alive after it.
    var nodes = List[Int]()
    var buffers = List[Int]()
    var plans = List[Int]()
    for i in range(20):
        opt.zero_grad()
        var loss = ((l2(l1(x).relu()) - y) ** 2).mean()
        loss.backward()
        opt.step()
        _ = loss.item()
        if i == 5 or i == 19:
            nodes.append(graph()[].live_nodes())
            buffers.append(graph()[].live_buffers())
            plans.append(graph()[].stats.kernels_compiled)
    assert_equal(nodes[1], nodes[0])
    assert_equal(buffers[1], buffers[0])
    assert_equal(plans[1], plans[0])
    assert_equal(graph()[].live_nodes(), 0)
    assert_equal(graph()[].live_buffers(), 0)


def test_equal_parameters_are_distinct() raises:
    """Parameters that start with equal values get separate gradients."""
    var a = Tensor.ones(3)
    var b = Tensor.ones(3)
    var d: List[Float64] = [1, 2, 3]
    var x = Tensor(d)
    var opt = SGD([a, b], lr=1.0)
    opt.zero_grad()
    (a * x + b * 10.0).sum().backward()
    assert_close(a.grad(), [1, 2, 3])
    assert_close(b.grad(), [10, 10, 10])
    opt.step()
    assert_close(a, [0, -1, -2])
    assert_close(b, [-9, -9, -9])


def test_backward_after_realize_raises() raises:
    var d: List[Float64] = [1, 2, 3]
    var x = Tensor(d).requires_grad_()
    var loss = (x * x).sum()
    _ = loss.item()
    var raised = False
    try:
        loss.backward()
    except:
        raised = True
    assert_true(raised)


def test_step_realizes_loss() raises:
    var d: List[Float64] = [1, 2, 3]
    var w = Tensor(d)
    var opt = SGD([w], lr=0.1)
    opt.zero_grad()
    var loss = (w * w).sum()
    loss.backward()
    opt.step([loss])
    assert_true(loss.uop().is_realized())
    assert_equal(loss.item(), 14.0)
    assert_close(w, [0.8, 1.6, 2.4])


def test_adam_reuses_plans() raises:
    """Adam's bias correction changes a constant every step; the plans
    must not."""
    Tensor.manual_seed(2)
    var x = Tensor.rand(8, 4)
    var y = Tensor.rand(8, 2)
    var l = Linear(4, 2)
    var opt = Adam(l.parameters(), lr=0.01)
    var plans = List[Int]()
    var losses = List[Float64]()
    for i in range(12):
        opt.zero_grad()
        var loss = ((l(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if i == 3 or i == 11:
            plans.append(graph()[].stats.kernels_compiled)
    assert_equal(plans[1], plans[0])
    assert_true(losses[11] < losses[0])


def test_quantize_int8() raises:
    Tensor.manual_seed(5)
    var w = Tensor.randn(64, 10) * Tensor.arange(1, 11).float()
    var qs = quantize_int8(w, 1)
    assert_equal(qs.q.dtype(), DT.int8)
    assert_equal(qs.scale.shape(), [1, 10])
    # Each column's largest magnitude maps to 127, and the error is at
    # most half a step.
    assert_close(qs.q.float().abs().max(0), List[Float64](length=10, fill=127))
    var err = (qs.q.float() * qs.scale - w).abs().max(0).tolist()
    var step = qs.scale.tolist()
    for j in range(10):
        assert_true(err[j] <= 0.5 * step[j] + 1e-6)
    var zero = quantize_int8(Tensor.zeros(3, 2), 0)
    assert_close(zero.q, [0, 0, 0, 0, 0, 0])


def test_layernorm_layer() raises:
    var ln = LayerNorm(3)
    var d: List[Float64] = [1, 2, 3, 2, 4, 6]
    var y = ln(Tensor.from_list(d, [2, 3]))
    assert_close(y.mean(1), [0, 0], 1e-5)
    assert_close((y * y).mean(1), [1, 1], 1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
