"""The executor's paths on sizes that cross their boundaries: blocks of
256, flattened dimensions, matrix-multiply tiles, and threads. Expected
values come from plain loops over `tolist()` data."""

from std.math import exp
from std.testing import TestSuite, assert_equal, assert_true, assert_false

from minigrad.dtype import DT
from minigrad.graph import graph, OP_CAST, OP_EXPAND
from minigrad.graph import BytePtr
from minigrad.runtime.gemm import gemm
from minigrad.runtime.kernel import runtime
from minigrad.tensor import Tensor
from helpers import assert_close, assert_shape


def _seq(shape: List[Int], dtype: DT = DT.float32) -> Tensor:
    """Distinct, non-monotone values in about [-2, 4]."""
    var n = 1
    for s in shape:
        n *= s
    var d = List[Float64]()
    for i in range(n):
        d.append(Float64((i * 37) % 23) * 0.25 - 2.0 + Float64(i % 7) * 0.03)
    return Tensor.from_list(d, shape, dtype)


def _reduce_ref(
    data: List[Float64], shape: List[Int], axes: List[Int], is_max: Bool
) -> List[Float64]:
    """Reduces `data` of `shape` over `axes`, dropping them."""
    var n = len(shape)
    var red = List[Bool](length=n, fill=False)
    for a in axes:
        red[a] = True
    var out_shape = List[Int]()
    for d in range(n):
        if not red[d]:
            out_shape.append(shape[d])
    var out_n = 1
    for s in out_shape:
        out_n *= s
    var out = List[Float64](length=out_n, fill=-1e300 if is_max else 0.0)
    var index = List[Int](length=n, fill=0)
    for i in range(len(data)):
        var f = i
        for dd in range(n):
            var d = n - 1 - dd
            index[d] = f % shape[d]
            f //= shape[d]
        var o = 0
        for d in range(n):
            if not red[d]:
                o = o * shape[d] + index[d]
        if is_max:
            if data[i] > out[o]:
                out[o] = data[i]
        else:
            out[o] += data[i]
    return out^


def test_elementwise_block_boundaries() raises:
    for n in [1, 15, 16, 17, 63, 64, 255, 256, 257, 511, 513, 1000]:
        var x = _seq([n])
        var got = (x * 2.0 + 1.0).relu() - x.exp() * 0.01
        var data = x.tolist()
        var expect = List[Float64]()
        for v in data:
            var r = v * 2.0 + 1.0
            if r < 0:
                r = 0
            expect.append(r - exp(v) * 0.01)
        assert_close(got, expect, 1e-4)


def test_reduce_each_axis() raises:
    var shape: List[Int] = [37, 300]
    var x = _seq(shape)
    var data = x.tolist()
    assert_close(x.sum(0), _reduce_ref(data, shape, [0], False), 1e-2)
    assert_close(x.sum(1), _reduce_ref(data, shape, [1], False), 1e-2)
    assert_close(x.max(0), _reduce_ref(data, shape, [0], True))
    assert_close(x.max(1), _reduce_ref(data, shape, [1], True))
    assert_close(x.sum(), _reduce_ref(data, shape, [0, 1], False), 1e-1)
    assert_close(x.max(), _reduce_ref(data, shape, [0, 1], True))


def test_reduce_through_views() raises:
    var shape: List[Int] = [37, 300]
    var x = _seq(shape)
    var t = x.T()
    var tdata = t.tolist()
    assert_close(t.sum(0), _reduce_ref(tdata, [300, 37], [0], False), 1e-2)
    assert_close(t.max(1), _reduce_ref(tdata, [300, 37], [1], True))
    var f = x.flip(1)
    assert_close(f.sum(1), _reduce_ref(f.tolist(), shape, [1], False), 1e-2)
    assert_close(f.max(0), _reduce_ref(f.tolist(), shape, [0], True))
    var p = x.pad([2, 3, 5, 0]).shrink([1, 40, 3, 290])
    var pshape: List[Int] = [39, 287]
    assert_shape(p, pshape)
    assert_close(p.sum(1), _reduce_ref(p.tolist(), pshape, [1], False), 1e-2)
    assert_close(p.sum(0), _reduce_ref(p.tolist(), pshape, [0], False), 1e-2)
    assert_close(
        (p + 1.0).max(0), _reduce_ref((p + 1.0).tolist(), pshape, [0], True)
    )


def test_reduce_trailing_dims_flattened() raises:
    # Small trailing dimensions that views keep from merging: the
    # executor flattens them. Groups below and above a block of 256.
    var shape: List[Int] = [6, 4, 5, 3, 3]
    var x = _seq(shape).flip(4)
    var data = x.tolist()
    assert_close(x.sum(3, 4), _reduce_ref(data, shape, [3, 4], False), 1e-3)
    assert_close(x.max(3, 4), _reduce_ref(data, shape, [3, 4], True))
    assert_close(
        x.sum(2, 3, 4), _reduce_ref(data, shape, [2, 3, 4], False), 1e-3
    )
    var big: List[Int] = [3, 20, 20]
    var y = _seq(big).flip(2)
    assert_close(y.sum(1, 2), _reduce_ref(y.tolist(), big, [1, 2], False), 1e-2)
    assert_close(y.max(1, 2), _reduce_ref(y.tolist(), big, [1, 2], True))
    var z = _seq([5, 30, 30]).permute(0, 2, 1)
    assert_close(
        z.sum(1, 2), _reduce_ref(z.tolist(), [5, 30, 30], [1, 2], False), 1e-2
    )
    # A reduced dimension outside the flattened ones.
    var w = _seq([40, 3, 5]).flip(2)
    assert_close(
        w.sum(0), _reduce_ref(w.tolist(), [40, 3, 5], [0], False), 1e-3
    )
    assert_close(
        w.sum(0, 2), _reduce_ref(w.tolist(), [40, 3, 5], [0, 2], False), 1e-3
    )


def test_elementwise_small_trailing_dims() raises:
    var shape: List[Int] = [50, 7, 3]
    var x = _seq(shape)
    var y = x.flip(2).pad([0, 0, 1, 1, 0, 2]) * 2.0 + 1.0
    var src = x.tolist()
    var expect = List[Float64]()
    for i in range(50):
        for j in range(9):
            for k in range(5):
                if j < 1 or j > 7 or k > 2:
                    expect.append(1.0)
                else:
                    expect.append(
                        src[(i * 7 + j - 1) * 3 + (2 - k)] * 2.0 + 1.0
                    )
    assert_close(y, expect, 1e-4)


def _matmul_ref(
    a: List[Float64], b: List[Float64], n: Int, k: Int, m: Int
) -> List[Float64]:
    var out = List[Float64]()
    for i in range(n):
        for j in range(m):
            var s = 0.0
            for t in range(k):
                s += a[i * k + t] * b[t * m + j]
            out.append(s)
    return out^


def check_matmul(n: Int, k: Int, m: Int) raises:
    var a = _seq([n, k])
    var b = _seq([k, m]) * 0.5
    assert_close(a @ b, _matmul_ref(a.tolist(), b.tolist(), n, k, m), 1e-2)
    # The same product from transposed operands.
    var at = _seq([k, n]).T()
    assert_close(at @ b, _matmul_ref(at.tolist(), b.tolist(), n, k, m), 1e-2)
    var bt = (_seq([m, k]) * 0.5).T()
    assert_close(a @ bt, _matmul_ref(a.tolist(), bt.tolist(), n, k, m), 1e-2)


def test_matmul_sizes() raises:
    var before = graph()[].stats.gemm_kernels
    check_matmul(1, 5, 1)
    check_matmul(7, 3, 5)
    check_matmul(8, 8, 32)
    check_matmul(9, 17, 33)
    check_matmul(33, 70, 45)
    check_matmul(200, 3, 100)
    check_matmul(3, 300, 2)
    check_matmul(65, 257, 97)
    assert_true(graph()[].stats.gemm_kernels > before)


def test_matrix_vector() raises:
    """Few rows: B streamed once, by columns or by rows, threaded past
    a million multiply-adds."""
    var before = graph()[].stats.gemm_kernels
    check_matmul(1, 7, 300)
    check_matmul(2, 300, 9)
    check_matmul(16, 33, 70)
    check_matmul(1, 768, 2304)
    check_matmul(3, 512, 1000)
    assert_true(graph()[].stats.gemm_kernels >= before + 15)


def test_few_rows() raises:
    """Up to two panels of rows (a batch of decode steps): B read in
    place, by rows for an `(in, out)` B and as dot products for an
    `(out, in)` B; partial panels, terms past a vector and a block of
    256, columns past whole tiles, and threads."""
    for n in [4, 5, 8, 9, 13, 16, 17]:
        check_matmul(n, 37, 101)
        check_matmul(n, 300, 70)
    check_matmul(8, 768, 400)  # past the threading threshold
    check_matmul(13, 260, 1000)
    for dt in [DT.int8, DT.float16]:
        for shape in [(8, 300, 1000), (11, 64, 70)]:
            var x = _seq([shape[0], shape[1]])
            var w = (
                (_seq([shape[1], shape[2]]) * 4.0).round().cast(dt).realize()
            )
            var expect = (x @ w.float().contiguous()).tolist()
            assert_close(x @ w, expect, 1e-3)
            var wt = w.T().contiguous().realize()
            assert_close(x @ wt.T(), expect, 1e-3)


def test_batched_contraction_threads() raises:
    """Heads as a batch dimension, small and large per product."""
    for shape in [(12, 1, 64, 40), (12, 30, 64, 40), (4, 200, 64, 300)]:
        var h = shape[0]
        var t = shape[1]
        var d = shape[2]
        var l = shape[3]
        var q = _seq([h, t, d])
        var k = _seq([h, d, l]) * 0.5
        var qd = q.tolist()
        var kd = k.tolist()
        var expect = List[Float64]()
        for i in range(h):
            var pq = List[Float64]()
            var pk = List[Float64]()
            for e in range(t * d):
                pq.append(qd[i * t * d + e])
            for e in range(d * l):
                pk.append(kd[i * d * l + e])
            for v in _matmul_ref(pq, pk, t, d, l):
                expect.append(v)
        assert_close(q @ k, expect, 2e-2)


def test_converted_operand() raises:
    """A float32 product with the other operand stored narrower: gemm
    converts it as it reads, in every path (few rows, tiles, a batch),
    and must equal the product with a float32 copy."""
    var before = graph()[].stats.gemm_kernels
    for shape in [(1, 300, 70), (3, 64, 1000), (40, 100, 90)]:
        var n = shape[0]
        var k = shape[1]
        var m = shape[2]
        var x = _seq([n, k])
        var w = _seq([k, m])
        for dt in [DT.float16, DT.bfloat16, DT.int8]:
            var wq = (w * 4.0).round().cast(dt).realize()
            var expect = (x @ wq.float().contiguous()).tolist()
            assert_close(x @ wq, expect, 1e-3)
            # Stored transposed, as an output projection is.
            var wt = wq.T().contiguous().realize()
            assert_close(x @ wt.T(), expect, 1e-3)
            # Converted on the left too.
            assert_close(
                wt @ x.T(), (wt.float().contiguous() @ x.T()).tolist(), 1e-3
            )
    var heads = _seq([4, 6, 32])
    var keys = _seq([4, 32, 50]).cast(DT.bfloat16).realize()
    assert_close(
        heads @ keys, (heads @ keys.float().contiguous()).tolist(), 1e-3
    )
    assert_true(graph()[].stats.gemm_kernels >= before + 30)


def test_pad_over_ops_that_break_zeros() raises:
    """A pad over `exp`, `log`, a reciprocal, a division, `pow` or an
    equality reads zeros where it pads, not `f(0)`: the padded value
    must equal padding a realized copy."""
    var x = _seq([3, 4]).abs() + 0.5
    var y = _seq([3, 4]) * 2.0 + 3.0
    var vals = List[Tensor]()
    vals.append(x.exp())
    vals.append(x.log())
    vals.append(x.reciprocal())
    vals.append(x / y)
    vals.append(x.pow(y))
    vals.append(x.eq(x).float())
    vals.append(Tensor.eye(3))
    vals.append((x.exp() + 1.0).sqrt())
    for v in vals:
        var want = v.contiguous().realize().pad([1, 2, 0, 3]).tolist()
        assert_close(v.pad([1, 2, 0, 3]), want, 1e-5)
        # Moved by more than the pad: a transpose and a shrink.
        var w = v.contiguous().realize().T().pad([2, 0, 1, 1])
        assert_close(v.T().pad([2, 0, 1, 1]), w.tolist(), 1e-5)


def test_unconverted_operand() raises:
    """An operand cast from a type gemm does not convert (uint8 images,
    int32), or a second converted operand, is materialized first: the
    product still runs on gemm."""
    var x = (_seq([40, 100]) * 3.0).round().abs()
    var w = _seq([100, 90])
    for dt in [DT.uint8, DT.int32, DT.float16]:
        var xq = x.cast(dt).realize()
        var wq = w.cast(dt).realize() if dt == DT.float16 else w
        var expect = xq.float().contiguous() @ wq.float().contiguous()
        var want = expect.tolist()
        var before = graph()[].stats.gemm_kernels
        assert_close(xq.float() @ wq.float(), want, 1e-3)
        assert_true(graph()[].stats.gemm_kernels == before + 1)


def test_cast_below_movements() raises:
    """A cast converts before a broadcast or window, so a materialized
    cast operand (uint8 images into a convolution) is the size of the
    images, not of their windows times the output channels."""
    var x = (_seq([2, 1, 9, 9]) * 3.0).round().abs().cast(DT.uint8).realize()
    var e = x.expand(2, 4, 9, 9).cast(DT.float32)
    assert_equal(e.uop().op(), OP_EXPAND)
    assert_equal(e.uop().src(0).op(), OP_CAST)
    var w = _seq([3, 1, 3, 3])
    var want = x.float().contiguous().conv2d(w).tolist()
    var before = graph()[].stats.gemm_kernels
    assert_close(x.conv2d(w), want, 1e-3)
    assert_true(graph()[].stats.gemm_kernels == before + 1)


def test_schedule_cache() raises:
    """A graph of a structure seen before replays its schedule, on its
    own buffers; a different constant is a different structure."""
    var hits = graph()[].stats.schedule_hits
    for i in range(4):
        var d = List[Float64]()
        for j in range(300):
            d.append(Float64((j * 7 + i) % 11) - 5.0)
        var a = Tensor.from_list(d, [3, 100])
        var b = Tensor.from_list(d, [100, 3])
        var got = ((a * 2.0 + 1.0).relu() @ b).sum(1)
        var expect = List[Float64]()
        for r in range(3):
            var s = 0.0
            for c in range(3):
                for k in range(100):
                    var x = d[r * 100 + k] * 2.0 + 1.0
                    s += (x if x > 0 else 0.0) * d[k * 3 + c]
            expect.append(s)
        assert_close(got, expect, 1e-3)
    assert_equal(graph()[].stats.schedule_hits, hits + 3)
    var a = Tensor.ones(3, 100)
    _ = ((a * 3.0 + 1.0).relu() @ Tensor.ones(100, 3)).sum(1).realize()
    assert_equal(graph()[].stats.schedule_hits, hits + 3)


def test_matmul_float64_and_ints() raises:
    var a = _seq([9, 11], DT.float64)
    var b = _seq([11, 6], DT.float64)
    assert_close(a @ b, _matmul_ref(a.tolist(), b.tolist(), 9, 11, 6), 1e-9)
    var ia = Tensor.arange(12).reshape(3, 4)
    var ib = Tensor.arange(8).reshape(4, 2)
    assert_equal((ia @ ib).dtype(), DT.int32)
    assert_close(ia @ ib, _matmul_ref(ia.tolist(), ib.tolist(), 3, 4, 2))


def test_matmul_threads() raises:
    # Enough work for gemm to use every core.
    var n = 300
    var k = 400
    var m = 350
    var a = _seq([n, k])
    var b = _seq([k, m]) * 0.5
    assert_close(a @ b, _matmul_ref(a.tolist(), b.tolist(), n, k, m), 2e-1)
    # A transposed (a weight gradient's shape): packed, one chunk per
    # thread.
    var at = _seq([k, n]).T()
    assert_close(at @ b, _matmul_ref(at.tolist(), b.tolist(), n, k, m), 2e-1)


def _gemm_check(
    ci: List[Int],
    cj: List[Int],
    ai: List[Int],
    ak: List[Int],
    bk: List[Int],
    bj: List[Int],
    a_size: Int,
    b_size: Int,
    c_size: Int,
    threads: Int,
) raises:
    """`gemm` on these tables against a plain loop, with B exactly
    `b_size` elements, so that its last windows end at its last element."""
    var a = List[Float32](capacity=a_size)
    for i in range(a_size):
        a.append(Float32((i * 37) % 23) * 0.25 - 2.0)
    var b = List[Float32](capacity=b_size)
    for i in range(b_size):
        b.append(Float32((i * 11) % 19) * 0.125 - 1.0)
    var c = List[Float32](length=c_size, fill=0.0)
    var expect = List[Float64](length=c_size, fill=0.0)
    for i in range(len(ci)):
        for j in range(len(cj)):
            var sum = 0.0
            for k in range(len(ak)):
                sum += Float64(a[ai[i] + ak[k]]) * Float64(b[bk[k] + bj[j]])
            expect[ci[i] + cj[j]] = sum
    gemm[DType.float32](
        BytePtr(unsafe_from_address=Int(c.unsafe_ptr())),
        ci,
        cj,
        BytePtr(unsafe_from_address=Int(a.unsafe_ptr())),
        ai,
        ak,
        BytePtr(unsafe_from_address=Int(b.unsafe_ptr())),
        bk,
        bj,
        threads,
    )
    _ = a^
    _ = b^
    for i in range(c_size):
        var d = abs(Float64(c[i]) - expect[i])
        if d > 1e-3 * (1.0 + abs(expect[i])):
            raise Error(t"gemm: element {i} is {c[i]}, expected {expect[i]}")


def _arange(n: Int, step: Int) -> List[Int]:
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(i * step)
    return out^


def test_gemm_convolution_tables() raises:
    """A convolution's tables, as the planner builds them: B's panels are
    runs of an image's rows (shorter than a vector, longer, and mixed,
    packed by runs), and a weight gradient's A has runs of one image's
    positions (too short to cut blocks at, one block, and across
    blocks)."""
    var n = 3
    var cin = 2
    for shape in [(9, 5, 5), (17, 13, 5), (24, 20, 5), (5, 3, 3)]:
        var size = shape[0]
        var out_size = shape[1]
        var kernel = shape[2]
        var image = size * size
        var positions = out_size * out_size
        # Positions (n, y, x) in the input and the output; window
        # offsets (c, ky, kx) in the input.
        var x_at = List[Int]()
        var y_at = List[Int]()
        for b in range(n):
            for y in range(out_size):
                for x in range(out_size):
                    x_at.append(b * cin * image + y * size + x)
                    y_at.append(b * positions + y * out_size + x)
        var window = List[Int]()
        for c in range(cin):
            for ky in range(kernel):
                for kx in range(kernel):
                    window.append(c * image + ky * size + kx)
        var nk = cin * kernel * kernel
        for cout in [20, 32]:
            for threads in [1, 4]:
                # Forward: C[o, (n, y, x)] = sum_w W[o, w] X[(n, y, x) + w].
                var cj = List[Int]()
                for b in range(n):
                    for p in range(positions):
                        cj.append(b * cout * positions + p)
                _gemm_check(
                    _arange(cout, positions),
                    cj,
                    _arange(cout, nk),
                    _arange(nk, 1),
                    window,
                    x_at,
                    cout * nk,
                    n * cin * image,
                    n * cout * positions,
                    threads,
                )
                # Weight gradient: C[o, w] = sum_(n, y, x) dY[n, o, y, x]
                # X[(n, y, x) + w].
                var ak = List[Int]()
                for b in range(n):
                    for p in range(positions):
                        ak.append(b * cout * positions + p)
                _gemm_check(
                    _arange(cout, nk),
                    _arange(nk, 1),
                    _arange(cout, positions),
                    ak,
                    x_at,
                    window,
                    n * cout * positions,
                    n * cin * image,
                    cout * nk,
                    threads,
                )


def test_batched_matmul() raises:
    # (4, 9, 7) @ (7, 5): the batch joins the rows.
    var a = _seq([4, 9, 7])
    var b = _seq([7, 5])
    var ad = a.tolist()
    var expect = List[Float64]()
    for i in range(4):
        var part = List[Float64]()
        for t in range(63):
            part.append(ad[i * 63 + t])
        for v in _matmul_ref(part, b.tolist(), 9, 7, 5):
            expect.append(v)
    assert_close(a @ b, expect, 1e-3)
    # (4, 9, 7) @ (4, 7, 5): a batch dimension shared by both operands.
    var c = _seq([4, 7, 5]) * 0.5
    var cd = c.tolist()
    var expect2 = List[Float64]()
    for i in range(4):
        var pa = List[Float64]()
        var pc = List[Float64]()
        for t in range(63):
            pa.append(ad[i * 63 + t])
        for t in range(35):
            pc.append(cd[i * 35 + t])
        for v in _matmul_ref(pa, pc, 9, 7, 5):
            expect2.append(v)
    assert_close(a @ c, expect2, 1e-3)


def test_int_and_bool_reductions() raises:
    var x = Tensor.arange(1000)
    assert_equal(x.sum().item(), 499500.0)
    assert_equal(x.max().item(), 999.0)
    assert_equal(x.reshape(10, 100).sum(1).tolist()[9], 94950.0)
    var even = (x % 2).eq(Tensor.zeros(1, dtype=DT.int32))
    assert_equal(even.sum().item(), 500.0)
    assert_equal(even.max().item(), 1.0)
    assert_equal(x.lt(0.0).max().item(), 0.0)
    var wide = Tensor.arange(300).cast(DT.int64) * 100000000
    assert_equal(wide.sum().item(), 4485000000000.0)


def test_where_and_cast_across_blocks() raises:
    var x = _seq([700])
    var got = x.gt(0.5).where(x.int().float(), x * -1.0)
    var expect = List[Float64]()
    for v in x.tolist():
        if v > 0.5:
            expect.append(Float64(Int(v)))
        else:
            expect.append(-v)
    assert_close(got, expect)


def _block_kernels(x: Tensor, y: Tensor) raises -> List[Tensor]:
    """Kernels of every shape the block executor splits: flat and
    strided elementwise, broadcast and padded loads, flattened offset
    tables, `where` and casts, and reductions over leading, middle,
    trailing, and several axes."""
    var out = List[Tensor]()
    out.append((x * y + 1.0).relu())
    out.append(x.permute(2, 0, 1) + y.permute(2, 0, 1) * 2.0)
    out.append(x + y.sum(0, keepdim=True))
    out.append(x.pad([0, 0, 1, 2, 3, 0]).exp())
    out.append(x.permute(0, 2, 1) * 3.0)
    out.append(x.gt(0.5).where(x.int().float(), y))
    out.append(x.sum(0))
    out.append(x.sum(1))
    out.append(x.sum(2))
    out.append(x.max(1, 2))
    out.append(x.permute(1, 0, 2).sum(0))
    out.append((x * y).sum(0, 2))
    for i in range(len(out)):
        _ = out[i].realize()
    return out^


def test_blocks_split_over_threads() raises:
    """Every block kernel shape on all threads equals the same kernel on
    one thread, bit for bit: chunks split outputs, never an output's
    order of summation."""
    var rt = runtime()
    var threads = rt[].threads
    var split = rt[].kernels_split
    for shape in [(64, 40, 30), (3, 7, 4000), (5000, 3, 5)]:
        var sh: List[Int] = [shape[0], shape[1], shape[2]]
        rt[].threads = 1
        var one = _block_kernels(_seq(sh), _seq(sh) * 0.5)
        rt[].threads = max(threads, 4)
        var many = _block_kernels(_seq(sh), _seq(sh) * 0.5)
        rt[].threads = threads
        for i in range(len(one)):
            var a = one[i].tolist()
            var b = many[i].tolist()
            assert_equal(len(a), len(b))
            for e in range(len(a)):
                assert_true(
                    a[e] == b[e], String(t"kernel {i} of {sh}, element {e}")
                )
    assert_true(rt[].kernels_split >= split + 30)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
