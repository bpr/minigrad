"""Dense linear algebra against identities: `a x = b`, `l l^T = a`,
`a v = lambda v` with orthonormal `v`, and known determinants."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false

from minigrad.dtype import DT
from minigrad.linalg import cholesky, det, eigh, inv, solve
from minigrad.tensor import Tensor
from helpers import assert_close, assert_shape


def _random(n: Int, m: Int, seed: Int) raises -> Tensor:
    Tensor.manual_seed(seed)
    return Tensor.randn(n, m).cast(DT.float64).realize()


def _spd(n: Int, seed: Int) raises -> Tensor:
    """A symmetric positive definite matrix."""
    var r = _random(n, n, seed)
    return (r @ r.T() + Tensor.eye(n, DT.float64) * Float64(n)).realize()


def test_solve() raises:
    for n in [1, 2, 3, 9, 40]:
        var a = _random(n, n, n)
        var b = _random(n, 3, n + 100)
        var x = solve(a, b)
        assert_shape(x, [n, 3])
        assert_equal(x.dtype(), DT.float64)
        assert_close(a @ x, b.tolist(), 1e-9)
        var v = b.slice(1, 0, 1).reshape(n)
        assert_close(a @ solve(a, v), v.tolist(), 1e-9)


def test_solve_needs_pivoting() raises:
    # A zero first pivot, which elimination without row exchanges fails.
    var rows: List[List[Float64]] = [[0, 1, 2], [1, 0, 3], [4, -3, 8]]
    var a = Tensor(rows)
    var b: List[Float64] = [8, 11, 25]
    assert_close(a @ solve(a, Tensor(b)), b, 1e-5)
    assert_equal(solve(a, Tensor(b)).dtype(), DT.float32)


def test_singular() raises:
    var rows: List[List[Float64]] = [[1, 2], [2, 4]]
    var raised = False
    try:
        _ = solve(Tensor(rows), Tensor.ones(2))
    except:
        raised = True
    assert_true(raised)
    var r3: List[List[Float64]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert_close(det(Tensor(r3)), [0.0], 1e-4)


def test_det_inv() raises:
    var rows: List[List[Float64]] = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
    var a = Tensor(rows)
    assert_close(det(a), [4.0], 1e-5)
    assert_shape(det(a), List[Int]())
    assert_close(a @ inv(a), Tensor.eye(3).tolist(), 1e-5)
    var b = _random(12, 12, 7)
    assert_close(b @ inv(b), Tensor.eye(12).tolist(), 1e-9)
    # det of a product is the product of dets.
    var c = _random(12, 12, 8)
    assert_close(det(b @ c), [det(b).item() * det(c).item()], 1e-6)


def test_cholesky() raises:
    var a = _spd(30, 3)
    var l = cholesky(a)
    assert_close(l @ l.T(), a.tolist(), 1e-9)
    assert_close(l.triu(1), List[Float64](length=900, fill=0.0))
    var rows: List[List[Float64]] = [[1, 2], [2, 1]]
    var raised = False
    try:
        _ = cholesky(Tensor(rows))
    except:
        raised = True
    assert_true(raised)


def test_eigh() raises:
    # The second-difference matrix: eigenvalues 2 - 2 cos(k pi / (n + 1)).
    var rows: List[List[Float64]] = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
    var e = eigh(Tensor(rows).cast(DT.float64))
    assert_close(
        e.values, [2.0 - 1.4142135623730951, 2.0, 3.414213562373095], 1e-12
    )
    for n in [1, 5, 33]:
        var a = _spd(n, n + 7)
        var r = eigh(a)
        var v = r.vectors
        # A V = V diag(values), V^T V = I, ascending.
        assert_close(a @ v, (v * r.values.reshape(1, n)).tolist(), 1e-9)
        assert_close(v.T() @ v, Tensor.eye(n).tolist(), 1e-12)
        var w = r.values.tolist()
        for i in range(1, n):
            assert_true(w[i - 1] <= w[i])


def test_errors() raises:
    var raised = False
    try:
        _ = det(Tensor.ones(2, 3))
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        _ = solve(Tensor.eye(3), Tensor.ones(4))
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
