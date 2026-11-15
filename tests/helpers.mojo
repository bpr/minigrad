"""Shared test helpers."""

from std.testing import assert_equal, assert_true

from minigrad.tensor import Tensor


def assert_close(t: Tensor, expect: List[Float64], atol: Float64 = 1e-5) raises:
    """Checks a tensor's elements, in row-major order, against `expect`."""
    var got = t.tolist()
    assert_equal(len(got), len(expect))
    for i in range(len(got)):
        var d = got[i] - expect[i]
        if d < 0:
            d = -d
        if not (
            d <= atol + 1e-5 * (expect[i] if expect[i] > 0 else -expect[i])
        ):
            raise Error(
                t"element {i}: got {got[i]}, expected {expect[i]} (all: {got})"
            )


def assert_shape(t: Tensor, expect: List[Int]) raises:
    """Checks a tensor's shape."""
    assert_equal(t.shape(), expect)
