"""Small dense linear algebra (`minigrad.linalg`).

`solve`, `det`, and `inv` by LU factorization with partial pivoting;
`cholesky`; and `eigh`, the symmetric eigenproblem, by Householder
reduction to tridiagonal form and the implicit QL method (EISPACK's
`tred2` and `tql2`).

These are eager: they realize their input, compute on the host in
`float64`, and return realized tensors of the input's float type
(`float32` for integer input). They are not differentiable. They are
meant for the small matrices of analysis and initialization, not for the
inner loop of a model.

The routines are molina's (`molina.dense`, the linear algebra package
of the same author's MojoDEC); this module converts tensors to and from
its `RealArray`. It imports `molina.dense` and `molina.real_array` only,
never the `molina` package itself, whose sparse solvers need MAX.
"""

from molina.dense import (
    cholesky as _cholesky,
    det as _det,
    eigh as _eigh,
    inv as _inv,
    solve as _solve,
)
from molina.real_array import RealArray

from minigrad.dtype import DT
from minigrad.tensor import Tensor


@fieldwise_init
struct Eigh(Copyable):
    """The eigenvalues and eigenvectors of a symmetric matrix."""

    var values: Tensor
    """The `n` eigenvalues, in ascending order."""
    var vectors: Tensor
    """`n x n`: column `i` is the unit eigenvector of `values[i]`."""


def solve(a: Tensor, b: Tensor) raises -> Tensor:
    """The solution `x` of `a x = b`.

    Args:
        a: A non-singular `n x n` matrix.
        b: A vector of length `n`, or an `n x m` matrix of right-hand
            sides.

    Returns:
        `x`, of `b`'s shape.

    Raises:
        If `a` is not square or is singular (an exactly zero pivot
        column; nearly singular matrices solve without error), or `b`
        does not have `n` rows.
    """
    var n = _square(a, "solve")
    var shape = b.shape()
    if len(shape) < 1 or len(shape) > 2 or shape[0] != n:
        raise Error(t"solve: right-hand side of shape {shape} for {n} x {n}")
    var B = RealArray(values=b.tolist()) if len(shape) == 1 else RealArray(
        n, shape[1], values=b.tolist()
    )
    return _tensor(_solve(_matrix(a, n), B), shape, _out_type(a, b))


def det(a: Tensor) raises -> Tensor:
    """The determinant of a square matrix (0 if it is singular).

    Returns:
        A scalar tensor.
    """
    var n = _square(a, "det")
    var v = 1.0 if n == 0 else _det(_matrix(a, n))
    return Tensor.from_list([v], List[Int](), _out_type(a, a))


def inv(a: Tensor) raises -> Tensor:
    """The inverse of a non-singular square matrix.

    Raises:
        If `a` is not square or is singular.
    """
    var n = _square(a, "inv")
    return _tensor(_inv(_matrix(a, n)), [n, n], _out_type(a, a))


def cholesky(a: Tensor) raises -> Tensor:
    """The lower-triangular `l` with `a = l l^T`.

    Only the lower triangle of `a` is read.

    Args:
        a: A symmetric positive definite `n x n` matrix.

    Returns:
        `l`, `n x n`.

    Raises:
        If `a` is not square or not positive definite (a non-positive
        pivot).
    """
    var n = _square(a, "cholesky")
    return _tensor(_cholesky(_matrix(a, n)), [n, n], _out_type(a, a))


def eigh(a: Tensor) raises -> Eigh:
    """The eigenvalues (ascending) and eigenvectors of a symmetric matrix.

    Householder reduction to tridiagonal form, then the implicit QL
    method: `O(n^3)` with a small constant. Only `a`'s symmetry is
    assumed, not checked.

    Args:
        a: A symmetric `n x n` matrix.

    Returns:
        The eigenvalues, and the eigenvectors as columns.

    Raises:
        If `a` is not square, or the QL iteration does not converge
        within 100 sweeps for some eigenvalue.
    """
    var n = _square(a, "eigh")
    var dt = _out_type(a, a)
    var r = _eigh(_matrix(a, n))
    return Eigh(_tensor(r.values, [n], dt), _tensor(r.vectors, [n, n], dt))


# ---- helpers


def _square(a: Tensor, name: StaticString) raises -> Int:
    var s = a.shape()
    if len(s) != 2 or s[0] != s[1]:
        raise Error(t"{name}: expected a square matrix, got shape {s}")
    return s[0]


def _out_type(a: Tensor, b: Tensor) -> DT:
    """`float64` if either input is, else `float32`."""
    if a.dtype() == DT.float64 or b.dtype() == DT.float64:
        return DT.float64
    return DT.float32


def _matrix(a: Tensor, n: Int) raises -> RealArray:
    """The square tensor `a` as an `n x n` `RealArray`, in `float64`."""
    return RealArray(n, n, values=a.tolist())


def _tensor(x: RealArray, shape: List[Int], dt: DT) raises -> Tensor:
    """`x`'s entries as a realized tensor of `shape` and type `dt`."""
    var values = List[Float64](capacity=x.size)
    for i in range(x.size):
        values.append(x.unsafe_get(i))
    return Tensor.from_list(values, shape, dt)
