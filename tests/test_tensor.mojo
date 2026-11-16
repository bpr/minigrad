"""Tensor operations against hand-computed values."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false

from minigrad.dtype import DT
from minigrad.tensor import Tensor
from helpers import assert_close, assert_shape


def _m23() -> Tensor:
    """[[0, 1, 2], [3, 4, 5]]"""
    var data: List[Float64] = [0, 1, 2, 3, 4, 5]
    return Tensor.from_list(data, [2, 3])


def test_creation() raises:
    assert_close(Tensor.zeros(2, 2), [0, 0, 0, 0])
    assert_close(Tensor.ones(3), [1, 1, 1])
    assert_close(Tensor.full([2], 2.5), [2.5, 2.5])
    assert_close(Tensor.arange(5), [0, 1, 2, 3, 4])
    assert_close(Tensor.arange(2, 8, 3), [2, 5])
    assert_close(Tensor.eye(2), [1, 0, 0, 1])
    assert_equal(Tensor.arange(3).dtype(), DT.int32)
    var rows: List[List[Float64]] = [[1, 2], [3, 4]]
    assert_shape(Tensor(rows), [2, 2])
    assert_close(Tensor(3.5), [3.5])


def test_arithmetic() raises:
    var x = _m23()
    assert_close(x + x, [0, 2, 4, 6, 8, 10])
    assert_close(x - 1.0, [-1, 0, 1, 2, 3, 4])
    assert_close(2.0 * x, [0, 2, 4, 6, 8, 10])
    assert_close(x / 2, [0, 0.5, 1, 1.5, 2, 2.5])
    assert_close(1.0 - x, [1, 0, -1, -2, -3, -4])
    assert_close(-x, [0, -1, -2, -3, -4, -5])
    assert_close(x**2, [0, 1, 4, 9, 16, 25])
    assert_close(
        (x + 1.0) ** 0.5, [1, 1.41421356, 1.7320508, 2, 2.236068, 2.4494897]
    )
    assert_close(
        (x + 1.0) ** 1.5, [1, 2.828427, 5.196152, 8, 11.18034, 14.696938], 1e-4
    )


def test_broadcast() raises:
    var x = _m23()
    var row: List[Float64] = [10, 20, 30]
    var col: List[Float64] = [100, 200]
    assert_close(x + Tensor(row), [10, 21, 32, 13, 24, 35])
    assert_close(x + Tensor(col).reshape(2, 1), [100, 101, 102, 203, 204, 205])
    assert_shape(Tensor(col).reshape(2, 1) + Tensor(row), [2, 3])


def test_int_and_promotion() raises:
    var a = Tensor.arange(4)
    assert_equal((a + 1).dtype(), DT.int32)
    assert_equal((a + 1.5).dtype(), DT.float32)
    assert_equal((a / 2).dtype(), DT.float32)
    assert_close(a / 2, [0, 0.5, 1, 1.5])
    assert_close(a // 2, [0, 0, 1, 1])
    assert_close(a % 3, [0, 1, 2, 0])
    assert_equal((a.float() * a).dtype(), DT.float32)
    assert_close(a.cast(DT.float64) * 0.5, [0, 0.5, 1, 1.5])


def test_math() raises:
    var data: List[Float64] = [0.5, 1.0, 2.0]
    var x = Tensor(data)
    assert_close(x.exp(), [1.6487213, 2.7182818, 7.389056])
    assert_close(x.log(), [-0.6931472, 0, 0.6931472])
    assert_close(x.sqrt(), [0.70710678, 1, 1.41421356])
    assert_close(x.reciprocal(), [2, 1, 0.5])
    assert_close(x.sin(), [0.47942554, 0.84147098, 0.90929743])
    assert_close(x.cos(), [0.87758256, 0.54030231, -0.41614684])
    assert_close(x.sigmoid(), [0.62245933, 0.73105858, 0.88079708])
    assert_close(x.tanh(), [0.46211716, 0.76159416, 0.96402758])
    var y: List[Float64] = [-1.5, -0.2, 0.0, 0.7, 2.5]
    var t = Tensor(y)
    assert_close(t.relu(), [0, 0, 0, 0.7, 2.5])
    assert_close(t.abs(), [1.5, 0.2, 0, 0.7, 2.5])
    assert_close(t.sign(), [-1, -1, 0, 1, 1])
    assert_close(t.trunc(), [-1, 0, 0, 0, 2])
    assert_close(t.floor(), [-2, -1, 0, 0, 2])
    assert_close(t.ceil(), [-1, 0, 0, 1, 3])
    assert_close(t.clip(-1.0, 1.0), [-1, -0.2, 0, 0.7, 1])
    assert_close(t.leaky_relu(0.1), [-0.15, -0.02, 0, 0.7, 2.5])
    assert_close(t.maximum(Tensor(0.5)), [0.5, 0.5, 0.5, 0.7, 2.5])
    assert_close(t.minimum(Tensor(0.5)), [-1.5, -0.2, 0, 0.5, 0.5])


def test_compare_where() raises:
    var x = _m23()
    assert_equal(x.lt(2.0).dtype(), DT.bool)
    assert_close(x.lt(2.0), [1, 1, 0, 0, 0, 0])
    assert_close(x.ge(2.0), [0, 0, 1, 1, 1, 1])
    assert_close(x.eq(3.0), [0, 0, 0, 1, 0, 0])
    assert_close(x.ne(3.0), [1, 1, 1, 0, 1, 1])
    assert_close(x.gt(2.0).where(x, -1.0), [-1, -1, -1, 3, 4, 5])
    assert_close(x.gt(2.0).where(1.0, 0.0), [0, 0, 0, 1, 1, 1])
    assert_close(x.lt(2.0).logical_not(), [0, 0, 1, 1, 1, 1])


def test_reduce() raises:
    var x = _m23()
    assert_close(x.sum(), [15])
    assert_shape(x.sum(), List[Int]())
    assert_close(x.sum(0), [3, 5, 7])
    assert_close(x.sum(1), [3, 12])
    assert_shape(x.sum(1, keepdim=True), [2, 1])
    assert_close(x.sum(-1), [3, 12])
    assert_close(x.max(), [5])
    assert_close(x.max(0), [3, 4, 5])
    assert_close(x.min(1), [0, 3])
    assert_close(x.mean(), [2.5])
    assert_close(x.mean(0), [1.5, 2.5, 3.5])
    assert_close(x.`var`(1), [1, 1])
    assert_close(x.std(), [1.8708287])
    assert_close(x.argmax(1), [2, 2])
    assert_close(x.argmax(0), [1, 1, 1])
    assert_close((-x).argmax(1), [0, 0])
    assert_equal(x.lt(3.0).sum().dtype(), DT.int32)
    assert_close(x.lt(3.0).sum(), [3])


def test_movement() raises:
    var x = _m23()
    assert_close(x.T(), [0, 3, 1, 4, 2, 5])
    assert_close(x.reshape(3, 2), [0, 1, 2, 3, 4, 5])
    assert_shape(x.reshape(-1, 2), [3, 2])
    assert_close(x.T().reshape(6), [0, 3, 1, 4, 2, 5])
    assert_close(x.flip(1), [2, 1, 0, 5, 4, 3])
    assert_close(x.flip(0, 1), [5, 4, 3, 2, 1, 0])
    assert_close(x.pad([1, 0, 0, 1]), [0, 0, 0, 0, 0, 1, 2, 0, 3, 4, 5, 0])
    assert_close(x.shrink([0, 2, 1, 3]), [1, 2, 4, 5])
    assert_close(x.slice(1, 1, 3), [1, 2, 4, 5])
    assert_close(x.slice(1, -1, 3), [2, 5])
    assert_close(x[1], [3, 4, 5])
    assert_close(x[1, 2], [5])
    assert_shape(x.unsqueeze(0), [1, 2, 3])
    assert_shape(x.unsqueeze(-1), [2, 3, 1])
    assert_shape(x.unsqueeze(1).squeeze(1), [2, 3])
    assert_shape(x.flatten(), [6])
    assert_close(x.expand(2, 2, 3), [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
    assert_close(x.repeat([1, 2]), [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5])
    assert_close(
        Tensor.cat([x, x + 10.0], 0), [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
    )
    assert_close(
        Tensor.cat([x, x + 10.0], 1), [0, 1, 2, 10, 11, 12, 3, 4, 5, 13, 14, 15]
    )
    assert_shape(Tensor.stack([x, x], 0), [2, 2, 3])


def test_matmul() raises:
    var x = _m23()
    assert_close(x @ x.T(), [5, 14, 14, 50])
    assert_close(x.T() @ x, [9, 12, 15, 12, 17, 22, 15, 22, 29])
    var v: List[Float64] = [1, 2, 3]
    assert_close(x @ Tensor(v), [8, 26])
    assert_close(Tensor(v) @ Tensor(v), [14])
    var b = Tensor.stack([x, x * 2.0], 0)  # (2, 2, 3)
    assert_shape(b @ x.T(), [2, 2, 2])
    assert_close(b @ x.T(), [5, 14, 14, 50, 10, 28, 28, 100])


def test_softmax() raises:
    var x = _m23()
    var s = x.softmax()
    assert_close(
        s,
        [
            0.09003057,
            0.24472847,
            0.66524096,
            0.09003057,
            0.24472847,
            0.66524096,
        ],
    )
    assert_close(s.sum(1), [1, 1])
    assert_close(
        x.log_softmax(),
        [
            -2.4076059,
            -1.4076059,
            -0.4076059,
            -2.4076059,
            -1.4076059,
            -0.4076059,
        ],
    )
    assert_close(x.logsumexp(1), [2.4076059, 5.4076059])


def test_lazy_and_realize() raises:
    var x = _m23()
    var y = x * 2.0 + 1.0
    assert_false(y.uop().is_realized())
    _ = y.realize()
    assert_true(y.uop().is_realized())
    assert_close(y, [1, 3, 5, 7, 9, 11])
    # A copy of a tensor is the same tensor.
    var z = y
    _ = z.assign(y * 0.0)
    assert_close(y, [0, 0, 0, 0, 0, 0])


def test_assign() raises:
    var w = Tensor.ones(4).contiguous().realize()
    var alias = w
    _ = w.assign(w * 3.0)
    _ = w.assign(w + 1.0)
    assert_close(alias, [4, 4, 4, 4])
    var data: List[Float64] = [1, 2, 3, 4]
    var v = Tensor(data)
    _ = v.assign(v.flip(0))
    assert_close(v, [4, 3, 2, 1])


def test_assign_does_not_alias() raises:
    """Tensors that merely have equal values are separate tensors."""
    var a = Tensor.zeros(3).contiguous().realize()
    var b = Tensor.zeros(3).contiguous().realize()
    var doubled = a * 2.0 + 1.0
    _ = a.assign(a + 5.0)
    assert_close(a, [5, 5, 5])
    assert_close(b, [0, 0, 0])
    # Computed from the old value of `a`, even though realized later.
    assert_close(doubled, [1, 1, 1])


def test_item_and_string() raises:
    var x = _m23()
    assert_equal(x.sum().item(), 15.0)
    assert_equal(Tensor.arange(3).to_string(), "[0, 1, 2]")
    assert_equal(x.to_string(), "[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]")


def test_half_storage() raises:
    var d: List[Float64] = [1.0, -2.5, 0.1, 65504.0, 3.0e38]
    var x = Tensor(d)
    # float16 keeps 11 significant bits and overflows past 65504;
    # bfloat16 keeps 8 bits and float32's range.
    var h = x.half()
    assert_equal(h.dtype(), DT.float16)
    assert_close(h.slice(0, 0, 4), [1.0, -2.5, 0.0999755859375, 65504.0])
    var b = x.cast(DT.bfloat16)
    assert_close(b.slice(0, 0, 4), [1.0, -2.5, 0.10009765625, 65536.0])
    assert_true(abs(b.tolist()[4] / 3.0e38 - 1.0) < 1.0 / 256.0)
    assert_equal((h + h).dtype(), DT.float16)
    assert_equal((h + b).dtype(), DT.float32)
    var s = Tensor.ones(3000).half().sum()
    assert_equal(s.dtype(), DT.float16)
    assert_close(s, [3000.0])  # accumulated in float32: exact


def test_round() raises:
    var d: List[Float64] = [-2.5, -1.5, -0.5, 0.4, 0.5, 1.5, 2.5, 2.6, -2.6]
    assert_close(Tensor(d).round(), [-2, -2, 0, 0, 0, 2, 2, 3, -3])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
