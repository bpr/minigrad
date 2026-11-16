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

The routines are copied from MojoDEC (`mojodec/routines/dense.mojo`, the
same author's MIT-licensed project), with NuMojo arrays replaced by flat
lists. A fix to one belongs in the other too.
"""

from std.math import sqrt

from minigrad.dtype import DT
from minigrad.tensor import Tensor

comptime _W = 8
"""SIMD width of the inner loops: 8 doubles."""


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
    var m = 1 if len(shape) == 1 else shape[1]
    var f = _LU(a.tolist(), n)
    var x = f.solve(b.tolist(), m)
    return Tensor.from_list(x, shape, _out_type(a, b))


def det(a: Tensor) raises -> Tensor:
    """The determinant of a square matrix (0 if it is singular).

    Returns:
        A scalar tensor.
    """
    var n = _square(a, "det")
    var d = a.tolist()
    var v: Float64
    if n == 0:
        v = 1.0
    elif n == 1:
        v = d[0]
    elif n == 2:
        v = d[0] * d[3] - d[1] * d[2]
    else:
        try:
            var f = _LU(d^, n)
            v = f.sign
            for i in range(n):
                v *= f.lu[i * n + i]
        except:
            v = 0.0
    return Tensor.from_list([v], List[Int](), _out_type(a, a))


def inv(a: Tensor) raises -> Tensor:
    """The inverse of a non-singular square matrix.

    Raises:
        If `a` is not square or is singular.
    """
    var n = _square(a, "inv")
    var eye = List[Float64](length=n * n, fill=0.0)
    for i in range(n):
        eye[i * n + i] = 1.0
    var f = _LU(a.tolist(), n)
    return Tensor.from_list(f.solve(eye, n), [n, n], _out_type(a, a))


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
    var A = a.tolist()
    var L = List[Float64](length=n * n, fill=0.0)
    for j in range(n):
        var s = A[j * n + j] - _dot_at(L, j * n, L, j * n, j)
        if s <= 0:
            raise Error("cholesky: matrix is not positive definite")
        var d = sqrt(s)
        L[j * n + j] = d
        for i in range(j + 1, n):
            var t = A[i * n + j] - _dot_at(L, i * n, L, j * n, j)
            L[i * n + j] = t / d
    return Tensor.from_list(L, [n, n], _out_type(a, a))


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
    if n == 0:
        return Eigh(
            Tensor.from_list(List[Float64](), [0], dt),
            Tensor.from_list(List[Float64](), [0, 0], dt),
        )
    # V holds the matrix, then the transposed accumulated transformations:
    # row i of V is eigenvector i, so the inner loops run over contiguous
    # memory. The matrix is symmetric, so it is its own transpose.
    var V = a.tolist()
    var d = List[Float64](length=n, fill=0.0)
    var e = List[Float64](length=n, fill=0.0)
    _tred2(n, V, d, e)
    _tql2(n, V, d, e)
    var Z = List[Float64](length=n * n, fill=0.0)
    for i in range(n):
        for j in range(n):
            Z[i * n + j] = V[j * n + i]
    return Eigh(Tensor.from_list(d, [n], dt), Tensor.from_list(Z, [n, n], dt))


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


struct _LU(Movable):
    """`P A = L U`, packed in one array (`L` below the diagonal with an
    implied unit diagonal, `U` on and above), with the row permutation."""

    var lu: List[Float64]
    var perm: List[Int]
    """Row `i` of `P A` is row `perm[i]` of `A`."""
    var sign: Float64
    """The permutation's sign."""
    var n: Int

    def __init__(out self, var lu: List[Float64], n: Int) raises:
        """Factors the row-major `n x n` matrix `lu`.

        Raises:
            If a pivot column is exactly zero.
        """
        self.n = n
        self.perm = List[Int](capacity=n)
        for i in range(n):
            self.perm.append(i)
        self.sign = 1.0
        for k in range(n):
            var p = k
            var best = abs(lu[k * n + k])
            for i in range(k + 1, n):
                var v = abs(lu[i * n + k])
                if v > best:
                    best = v
                    p = i
            if best == 0:
                raise Error("matrix is singular")
            if p != k:
                for j in range(n):
                    var t = lu[k * n + j]
                    lu[k * n + j] = lu[p * n + j]
                    lu[p * n + j] = t
                self.perm.swap_elements(k, p)
                self.sign = -self.sign
            var pivot = lu[k * n + k]
            for i in range(k + 1, n):
                var f = lu[i * n + k] / pivot
                lu[i * n + k] = f
                if f != 0:
                    _row_sub(lu, i * n + k + 1, k * n + k + 1, f, n - k - 1)
        self.lu = lu^

    def solve(self, b: List[Float64], m: Int) -> List[Float64]:
        """`X` with `A X = B`, for the row-major `n x m` matrix `B`."""
        var n = self.n
        var X = List[Float64](length=n * m, fill=0.0)
        if m == 1:
            for i in range(n):
                X[i] = b[self.perm[i]] - _dot_at(self.lu, i * n, X, 0, i)
            for i in range(n - 1, -1, -1):
                var s = X[i] - _dot_at(
                    self.lu, i * n + i + 1, X, i + 1, n - i - 1
                )
                X[i] = s / self.lu[i * n + i]
            return X^
        # Several right-hand sides: whole rows of X, which are contiguous.
        for i in range(n):
            for c in range(m):
                X[i * m + c] = b[self.perm[i] * m + c]
            for j in range(i):
                _row_sub(X, i * m, j * m, self.lu[i * n + j], m)
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                _row_sub(X, i * m, j * m, self.lu[i * n + j], m)
            var pivot = self.lu[i * n + i]
            for c in range(m):
                X[i * m + c] = X[i * m + c] / pivot
        return X^


def _tred2(
    n: Int, mut V: List[Float64], mut d: List[Float64], mut e: List[Float64]
):
    """EISPACK's Householder reduction of the symmetric `V` to tridiagonal
    form (`d`, `e`), accumulating the transformations in `V`."""
    for j in range(n):
        d[j] = V[j * n + (n - 1)]
    for i in range(n - 1, 0, -1):
        var scale: Float64 = 0
        var h: Float64 = 0
        for k in range(i):
            scale += abs(d[k])
        if scale == 0:
            e[i] = d[i - 1]
            for j in range(i):
                d[j] = V[j * n + (i - 1)]
                V[j * n + i] = 0
                V[i * n + j] = 0
        else:
            for k in range(i):
                d[k] = d[k] / scale
                h += d[k] * d[k]
            var f = d[i - 1]
            var g = sqrt(h)
            if f > 0:
                g = -g
            e[i] = scale * g
            h = h - f * g
            d[i - 1] = f - g
            for j in range(i):
                e[j] = 0
            for j in range(i):
                f = d[j]
                V[i * n + j] = f
                g = e[j] + V[j * n + j] * f
                for k in range(j + 1, i):
                    g += V[j * n + k] * d[k]
                    e[k] = e[k] + (V[j * n + k] * f)
                e[j] = g
            f = 0
            for j in range(i):
                e[j] = e[j] / h
                f += e[j] * d[j]
            var hh = f / (h + h)
            for j in range(i):
                e[j] = e[j] - (hh * d[j])
            for j in range(i):
                f = d[j]
                g = e[j]
                for k in range(j, i):
                    V[j * n + k] = V[j * n + k] - (f * e[k] + g * d[k])
                d[j] = V[j * n + (i - 1)]
                V[j * n + i] = 0
        d[i] = h
    for i in range(n - 1):
        V[i * n + (n - 1)] = V[i * n + i]
        V[i * n + i] = 1
        var h = d[i + 1]
        if h != 0:
            for k in range(i + 1):
                d[k] = V[(i + 1) * n + k] / h
            for j in range(i + 1):
                var g: Float64 = 0
                for k in range(i + 1):
                    g += V[(i + 1) * n + k] * V[j * n + k]
                for k in range(i + 1):
                    V[j * n + k] = V[j * n + k] - (g * d[k])
        for k in range(i + 1):
            V[(i + 1) * n + k] = 0
    for j in range(n):
        d[j] = V[j * n + (n - 1)]
        V[j * n + (n - 1)] = 0
    V[(n - 1) * n + (n - 1)] = 1
    e[0] = 0


def _tql2(
    n: Int, mut V: List[Float64], mut d: List[Float64], mut e: List[Float64]
) raises:
    """EISPACK's implicit QL on the tridiagonal (`d`, `e`), `e[i]` coupling
    `i - 1` and `i`, applying the rotations to the rows of `V`; then sorts
    the eigenvalues ascending, with their rows of `V`."""
    for i in range(1, n):
        e[i - 1] = e[i]
    e[n - 1] = 0
    var f: Float64 = 0
    var tst1: Float64 = 0
    var eps: Float64 = 2.0**-52
    for l in range(n):
        tst1 = max(tst1, abs(d[l]) + abs(e[l]))
        var m = l
        while m < n:
            if abs(e[m]) <= eps * tst1:
                break
            m += 1
        if m > l:
            var iters = 0
            while True:
                iters += 1
                if iters > 100:
                    raise Error("eigh: QL iteration did not converge")
                var g = d[l]
                var p = (d[l + 1] - g) / (2 * e[l])
                var r = _hypot(p, 1.0)
                if p < 0:
                    r = -r
                d[l] = e[l] / (p + r)
                d[l + 1] = e[l] * (p + r)
                var dl1 = d[l + 1]
                var h = g - d[l]
                for i in range(l + 2, n):
                    d[i] = d[i] - h
                f += h
                p = d[m]
                var c: Float64 = 1
                var c2 = c
                var c3 = c
                var el1 = e[l + 1]
                var s: Float64 = 0
                var s2: Float64 = 0
                for i in range(m - 1, l - 1, -1):
                    c3 = c2
                    c2 = c
                    s2 = s
                    g = c * e[i]
                    h = c * p
                    r = _hypot(p, e[i])
                    e[i + 1] = s * r
                    s = e[i] / r
                    c = p / r
                    p = c * d[i] - s * g
                    d[i + 1] = h + s * (c * g + s * d[i])
                    _rotate_rows(V, n, i, c, s)
                p = -s * s2 * c3 * el1 * e[l] / dl1
                e[l] = s * p
                d[l] = c * p
                if abs(e[l]) <= eps * tst1:
                    break
        d[l] = d[l] + f
        e[l] = 0
    for i in range(n - 1):
        var k = i
        var p = d[i]
        for j in range(i + 1, n):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            for j in range(n):
                var t = V[i * n + j]
                V[i * n + j] = V[k * n + j]
                V[k * n + j] = t


def _dot_at(
    a: List[Float64], i: Int, b: List[Float64], j: Int, n: Int
) -> Float64:
    """`sum(a[i + t] * b[j + t] for t < n)`."""
    var pa = a.unsafe_ptr()
    var pb = b.unsafe_ptr()
    var acc = SIMD[DType.float64, _W](0)
    var t = 0
    while t + _W <= n:
        acc += pa.unsafe_load[width=_W](i + t) * pb.unsafe_load[width=_W](j + t)
        t += _W
    var s = acc.reduce_add()
    while t < n:
        s += a[i + t] * b[j + t]
        t += 1
    return s


def _row_sub(mut a: List[Float64], dst: Int, src: Int, f: Float64, n: Int):
    """`a[dst + t] -= f * a[src + t]` for `t < n`; the ranges must not
    overlap."""
    var p = a.unsafe_ptr()
    var t = 0
    while t + _W <= n:
        p.unsafe_store(
            dst + t,
            p.unsafe_load[width=_W](dst + t)
            - f * p.unsafe_load[width=_W](src + t),
        )
        t += _W
    while t < n:
        a[dst + t] = a[dst + t] - f * a[src + t]
        t += 1


def _rotate_rows(mut V: List[Float64], n: Int, i: Int, c: Float64, s: Float64):
    """Applies a Givens rotation to rows `i` and `i + 1` of the `n x n`
    `V`: `(v_i, v_i+1) <- (c v_i - s v_i+1, s v_i + c v_i+1)`."""
    var p = V.unsafe_ptr()
    var a = i * n
    var b = a + n
    var k = 0
    while k + _W <= n:
        var x = p.unsafe_load[width=_W](a + k)
        var y = p.unsafe_load[width=_W](b + k)
        p.unsafe_store(b + k, s * x + c * y)
        p.unsafe_store(a + k, c * x - s * y)
        k += _W
    while k < n:
        var x = p[unsafe_offset=a + k]
        var y = p[unsafe_offset=b + k]
        p[unsafe_offset=b + k] = s * x + c * y
        p[unsafe_offset=a + k] = c * x - s * y
        k += 1


def _hypot(a: Float64, b: Float64) -> Float64:
    """`sqrt(a^2 + b^2)` without overflow."""
    var x = abs(a)
    var y = abs(b)
    if x < y:
        var t = x
        x = y
        y = t
    if x == 0:
        return 0
    var r = y / x
    return x * sqrt(1 + r * r)
