"""MiniJit: a replay computes what running the function would, for
pure functions, training steps (state through `assign`), and eager
operations indexed by tensors."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false

from minigrad.dtype import DT
from minigrad.graph import graph
from minigrad.jit import MiniJit
from minigrad.nn import Adam, Linear, SGD, collect
from minigrad.tensor import Tensor
from helpers import assert_close


def _data(n: Int, seed: Int) -> List[Float64]:
    var d = List[Float64]()
    for i in range(n):
        d.append(Float64((i * 37 + seed * 11) % 23) * 0.25 - 2.0)
    return d^


def f_pure(xs: List[Tensor]) raises -> List[Tensor]:
    var h = (xs[0] @ xs[1]).relu()
    return [h.sum(1), (h * 2.0).max()]


def test_pure_function() raises:
    var jit = MiniJit()
    for call in range(5):
        var a = Tensor.from_list(_data(12, call), [3, 4]).contiguous().realize()
        var b = (
            Tensor.from_list(_data(20, call + 7), [4, 5]).contiguous().realize()
        )
        var before = graph()[].stats.schedules
        var got = jit(f_pure, [a, b])
        if call >= 2:
            # A replay runs no scheduler at all.
            assert_equal(graph()[].stats.schedules, before)
        var want = f_pure([a, b])
        assert_close(got[0], want[0].tolist())
        assert_close(got[1], want[1].tolist())


def f_view_out(xs: List[Tensor]) raises -> List[Tensor]:
    return [(xs[0] * 2.0).contiguous().reshape(4, 3).T()]


def test_views_as_inputs_and_outputs() raises:
    # Inputs that are views (copied into buffers of their own) and an
    # output that is one: a replay reads the new inputs, and the copies
    # stay alive.
    var jit = MiniJit()
    for call in range(5):
        var a = Tensor.from_list(_data(12, call), [12]).reshape(3, 4)
        var got = jit(f_view_out, [a])
        var want = f_view_out([Tensor.from_list(_data(12, call), [3, 4])])
        assert_close(got[0], want[0].tolist())


def test_shape_change_raises() raises:
    var jit = MiniJit()
    var a = Tensor.ones(3, 4)
    var b = Tensor.ones(4, 5)
    _ = jit(f_pure, [a, b])
    _ = jit(f_pure, [a, b])
    var raised = False
    try:
        _ = jit(f_pure, [Tensor.ones(2, 4), b])
    except:
        raised = True
    assert_true(raised)


def _train(use_jit: Bool, adam: Bool) raises -> List[Float64]:
    """Ten steps of a small model; returns the losses and the weights."""
    Tensor.manual_seed(3)
    var l1 = Linear(6, 8)
    var l2 = Linear(8, 3)
    var params = collect(l1, l2)
    var sgd = SGD(params, lr=0.05, momentum=0.9)
    var adm = Adam(params, lr=0.01)
    var jit = MiniJit()
    var out = List[Float64]()

    def step(
        xs: List[Tensor],
    ) raises {ref l1, ref l2, mut sgd, mut adm, var adam} -> List[Tensor]:
        sgd.zero_grad()
        var loss = l2(l1(xs[0]).relu()).sparse_categorical_crossentropy(xs[1])
        loss.backward()
        if adam:
            adm.step([loss])
        else:
            sgd.step([loss])
        return [loss]

    for i in range(10):
        var x = Tensor.from_list(_data(24, i), [4, 6])
        var y = Tensor(List[Int]([i % 3, (i + 1) % 3, 0, 2]))
        var loss = jit(step, [x, y])[0] if use_jit else step([x, y])[0]
        out.append(loss.item())
    for p in params:
        for v in p.tolist():
            out.append(v)
    return out^


def test_training_sgd() raises:
    var plain = _train(False, False)
    var jitted = _train(True, False)
    assert_close(Tensor(jitted), plain, 1e-6)


def test_training_adam() raises:
    var plain = _train(False, True)
    var jitted = _train(True, True)
    assert_close(Tensor(jitted), plain, 1e-6)


def f_cache(xs: List[Tensor]) raises -> List[Tensor]:
    # xs: table (5, 3), rows to gather, a cache, and where to write.
    var rows = xs[0].take_rows(xs[1])
    xs[2].write_(0, xs[3], rows * 10.0)
    return [xs[2].sum(1)]


def test_eager_ops_replay() raises:
    var table = Tensor.arange(15).reshape(5, 3).float().contiguous().realize()
    var cache = Tensor.zeros(6, 3).contiguous().realize()
    var jit = MiniJit()
    for pos in range(6):
        var idx = Tensor(List[Int]([(pos * 2) % 5]))
        var at = Tensor(List[Int]([pos]))
        var sums = jit(f_cache, [table, idx, cache, at])[0].tolist()
        var r = (pos * 2) % 5
        assert_close(
            Tensor(sums).slice(0, pos, pos + 1), [Float64(10 * (9 * r + 3))]
        )
    # Every row was written, by the run, the capture, and four replays.
    var want = List[Float64]()
    for pos in range(6):
        want.append(Float64(10 * (9 * ((pos * 2) % 5) + 3)))
    assert_close(cache.sum(1), want)


def f_host_index(xs: List[Tensor]) raises -> List[Tensor]:
    return [xs[0].take_rows([1, 0])]


def test_host_indices_refused() raises:
    var jit = MiniJit()
    var t = Tensor.ones(3, 2)
    _ = jit(f_host_index, [t])
    var raised = False
    try:
        _ = jit(f_host_index, [t])
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
