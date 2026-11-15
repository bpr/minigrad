"""View algebra against a dense reference: every movement op is applied
both to a `View` and to an explicit table of source indices."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false

from minigrad.view import View, prod, strides_for_shape


struct Rng:
    var state: UInt64

    def __init__(out self, seed: Int):
        self.state = UInt64(seed) * 2862933555777941757 + 3037000493

    def next(mut self, n: Int) -> Int:
        """A value in `[0, n)`."""
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(n))


struct Dense(Copyable):
    """For each index of `shape`, the source element it reads, or -1."""

    var shape: List[Int]
    var idx: List[Int]

    def __init__(out self, var shape: List[Int], var idx: List[Int]):
        self.shape = shape^
        self.idx = idx^

    @staticmethod
    def identity(shape: List[Int]) -> Dense:
        var idx = List[Int]()
        for i in range(prod(shape)):
            idx.append(i)
        return Dense(shape.copy(), idx^)

    def remap(self, new_shape: List[Int], src: List[Int]) -> Dense:
        """`src[i]` is the flat index into self for new flat index `i`,
        or -1."""
        var idx = List[Int]()
        for s in src:
            idx.append(self.idx[s] if s >= 0 else -1)
        return Dense(new_shape.copy(), idx^)


def unravel(flat: Int, shape: List[Int]) -> List[Int]:
    var out = List[Int](length=len(shape), fill=0)
    var f = flat
    for i in range(len(shape)):
        var d = len(shape) - 1 - i
        out[d] = f % shape[d]
        f //= shape[d]
    return out^


def ravel(index: List[Int], shape: List[Int]) -> Int:
    var f = 0
    for d in range(len(shape)):
        f = f * shape[d] + index[d]
    return f


def view_lookup(v: View, index: List[Int]) -> Int:
    var a = v.offset
    for d in range(len(index)):
        if v.has_mask():
            if index[d] < v.mask[2 * d] or index[d] >= v.mask[2 * d + 1]:
                return -1
        a += index[d] * v.strides[d]
    return a


def check(v: View, d: Dense) raises:
    assert_equal(v.shape, d.shape)
    for i in range(prod(d.shape)):
        var index = unravel(i, d.shape)
        assert_equal(view_lookup(v, index), d.idx[i])


def dense_permute(d: Dense, axes: List[Int]) -> Dense:
    var shape = List[Int]()
    for a in axes:
        shape.append(d.shape[a])
    var src = List[Int]()
    for i in range(prod(shape)):
        var index = unravel(i, shape)
        var old = List[Int](length=len(axes), fill=0)
        for k in range(len(axes)):
            old[axes[k]] = index[k]
        src.append(ravel(old, d.shape))
    return d.remap(shape, src)


def dense_expand(d: Dense, shape: List[Int]) -> Dense:
    var src = List[Int]()
    for i in range(prod(shape)):
        var index = unravel(i, shape)
        for k in range(len(shape)):
            if d.shape[k] == 1:
                index[k] = 0
        src.append(ravel(index, d.shape))
    return d.remap(shape, src)


def dense_resize(d: Dense, arg: List[Int]) -> Dense:
    """Pad and shrink: new index i reads old index i + arg[2k]."""
    var shape = List[Int]()
    for k in range(len(d.shape)):
        shape.append(arg[2 * k + 1] - arg[2 * k])
    var src = List[Int]()
    for i in range(prod(shape)):
        var index = unravel(i, shape)
        var valid = True
        for k in range(len(shape)):
            index[k] += arg[2 * k]
            if index[k] < 0 or index[k] >= d.shape[k]:
                valid = False
        src.append(ravel(index, d.shape) if valid else -1)
    return d.remap(shape, src)


def dense_flip(d: Dense, axes: List[Int]) -> Dense:
    var src = List[Int]()
    for i in range(prod(d.shape)):
        var index = unravel(i, d.shape)
        for k in range(len(axes)):
            if axes[k] != 0:
                index[k] = d.shape[k] - 1 - index[k]
        src.append(ravel(index, d.shape))
    return d.remap(d.shape, src)


def dense_window(
    d: Dense, kernel: List[Int], stride: List[Int], dilation: List[Int]
) -> Dense:
    var nk = len(kernel)
    var p = len(d.shape) - nk
    var shape = List[Int]()
    for i in range(p):
        shape.append(d.shape[i])
    for j in range(nk):
        shape.append(
            (d.shape[p + j] - dilation[j] * (kernel[j] - 1) - 1) // stride[j]
            + 1
        )
    for j in range(nk):
        shape.append(kernel[j])
    var src = List[Int]()
    for i in range(prod(shape)):
        var index = unravel(i, shape)
        var old = List[Int]()
        for k in range(p):
            old.append(index[k])
        for j in range(nk):
            old.append(
                index[p + j] * stride[j] + index[p + nk + j] * dilation[j]
            )
        src.append(ravel(old, d.shape))
    return d.remap(shape, src)


def random_shape(mut rng: Rng) -> List[Int]:
    var shape = List[Int]()
    for _ in range(rng.next(4) + 1):
        shape.append(rng.next(4) + 1)
    return shape^


def random_factorization(mut rng: Rng, n: Int) -> List[Int]:
    """A random shape with `n` elements."""
    var shape = List[Int]()
    var rest = n
    while rest > 1 and len(shape) < 4:
        var f = rng.next(rest) + 1
        while rest % f != 0:
            f -= 1
        if f == 1 and rng.next(2) == 0:
            f = rest
        shape.append(f)
        rest //= f
    if rest > 1 or len(shape) == 0:
        shape.append(rest)
    if rng.next(3) == 0:
        shape.insert(rng.next(len(shape) + 1), 1)
    return shape^


def test_identity() raises:
    var shape: List[Int] = [2, 3, 4]
    var v = View.identity(shape)
    assert_true(v.contiguous)
    var expect: List[Int] = [12, 4, 1]
    assert_equal(v.strides, expect)
    check(v, Dense.identity(shape))


def test_transpose_flatten_fails() raises:
    var shape: List[Int] = [2, 3]
    var axes: List[Int] = [1, 0]
    var flat: List[Int] = [6]
    var v = View.identity(shape).permute(axes)
    assert_false(v.contiguous)
    assert_false(Bool(v.reshape(flat)))


def test_unsqueeze_of_permuted() raises:
    var shape: List[Int] = [2, 3]
    var axes: List[Int] = [1, 0]
    var target: List[Int] = [3, 1, 2, 1]
    var v = View.identity(shape).permute(axes)
    var d = dense_permute(Dense.identity(shape), axes)
    var r = v.reshape(target)
    assert_true(Bool(r))
    check(r.value(), Dense(target.copy(), d.idx.copy()))


def test_pad_then_shrink_roundtrip() raises:
    var shape: List[Int] = [3, 3]
    var p: List[Int] = [1, 2, 0, 1]
    var s: List[Int] = [1, 4, 0, 3]
    var v = View.identity(shape).pad(p).shrink(s)
    assert_true(v.contiguous)


def test_random_chains() raises:
    var rng = Rng(1234)
    var reshapes = 0
    var reshape_ok = 0
    var windows = 0
    for _ in range(3000):
        var shape = random_shape(rng)
        var v = View.identity(shape)
        var d = Dense.identity(shape)
        for _ in range(rng.next(6) + 1):
            var n = len(d.shape)
            var op = rng.next(7)
            if op == 6:  # sliding windows over the last one or two dims
                var nk = min(n, rng.next(2) + 1)
                var kernel = List[Int]()
                var stride = List[Int]()
                var dilation = List[Int]()
                for j in range(nk):
                    var size = d.shape[n - nk + j]
                    var k = rng.next(size) + 1
                    var dil = 1
                    if k > 1 and (size - 1) // (k - 1) > 1 and rng.next(2) == 0:
                        dil = 2
                    kernel.append(k)
                    dilation.append(dil)
                    stride.append(rng.next(2) + 1)
                var r = v.window(kernel, stride, dilation)
                if r:
                    windows += 1
                    v = r.value().copy()
                    d = dense_window(d, kernel, stride, dilation)
                else:
                    # Masked: materialize, then take windows of the copy.
                    assert_true(v.has_mask())
                    v = View.identity(d.shape)
                    d = Dense.identity(d.shape)
            elif op == 0:  # permute
                var axes = List[Int]()
                for k in range(n):
                    axes.append(k)
                for k in range(n):
                    var j = rng.next(n)
                    var t = axes[k]
                    axes[k] = axes[j]
                    axes[j] = t
                v = v.permute(axes)
                d = dense_permute(d, axes)
            elif op == 1:  # expand
                var ns = d.shape.copy()
                for k in range(n):
                    if ns[k] == 1 and rng.next(2) == 0:
                        ns[k] = rng.next(3) + 1
                v = v.expand(ns)
                d = dense_expand(d, ns)
            elif op == 2:  # pad
                var arg = List[Int]()
                var rs = List[Int]()
                for k in range(n):
                    var lo = rng.next(3)
                    var hi = rng.next(3)
                    arg.append(lo)
                    arg.append(hi)
                    rs.append(-lo)
                    rs.append(d.shape[k] + hi)
                v = v.pad(arg)
                d = dense_resize(d, rs)
            elif op == 3:  # shrink
                var arg = List[Int]()
                for k in range(n):
                    var lo = rng.next(d.shape[k])
                    var hi = lo + 1 + rng.next(d.shape[k] - lo)
                    arg.append(lo)
                    arg.append(hi)
                v = v.shrink(arg)
                d = dense_resize(d, arg)
            elif op == 4:  # flip
                var axes = List[Int]()
                for _ in range(n):
                    axes.append(rng.next(2))
                v = v.flip(axes)
                d = dense_flip(d, axes)
            else:  # reshape
                var ns = random_factorization(rng, prod(d.shape))
                reshapes += 1
                var r = v.reshape(ns)
                d = Dense(ns.copy(), d.idx.copy())
                if r:
                    reshape_ok += 1
                    v = r.value().copy()
                else:
                    # Materialize: the dense table becomes the new buffer.
                    v = View.identity(ns)
                    d = Dense.identity(ns)
            check(v, d)
    print("reshapes:", reshapes, "expressible as one view:", reshape_ok)
    print("windows:", windows)
    assert_true(windows > 500)
    assert_true(reshape_ok * 2 > reshapes)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
