"""Typed SIMD loops over one block of elements
(`minigrad.runtime.blockops`).

The CPU executor evaluates a kernel one block (at most `BLOCK` elements)
at a time: each instruction of the kernel is one call here that reads its
operand blocks and writes its result block. Blocks stay in the L1 cache
between instructions, so a fused expression never touches a full-size
temporary.

Functions here are generic over the element type and dispatched on the
runtime `DT` by the `*_block` entry points. Loops process whole SIMD
vectors and may read or write up to one vector past `n` elements: callers
give every block that much room (buffers are padded, temporaries are
`BLOCK` elements plus padding). The exceptions, which touch exactly `n`
elements, are the integer division loops and the `store_*`/`accumulate_*`
functions that write the kernel's output.
"""

from std.math import exp, log, sin, sqrt, trunc
from std.memory import bitcast, unsafe_memcpy
from std.sys import simd_width_of

from minigrad.dtype import DT
from minigrad.graph import (
    BytePtr,
    OP_ADD,
    OP_AND,
    OP_CMPEQ,
    OP_CMPLT,
    OP_CMPNE,
    OP_DIV,
    OP_EXP,
    OP_LOG,
    OP_MAX,
    OP_MOD,
    OP_MUL,
    OP_NEG,
    OP_OR,
    OP_POW,
    OP_RECIP,
    OP_SIN,
    OP_SQRT,
    OP_SUB,
    OP_TRUNC,
    OP_XOR,
    REDUCE_MAX,
    REDUCE_SUM,
)

comptime BLOCK = 256
"""Elements per block. A multiple of every SIMD width."""

comptime MASKED = -(1 << 62)
"""The entry of an offset table for an element that reads as zero.
Offsets themselves can be negative (a flipped view walks backward), so
the marker is a value no offset takes."""

comptime PAD_BYTES = 64
"""Bytes readable and writable past the end of every block and buffer:
one SIMD vector of any element type."""


def mojo_dtype(code: Int) -> DType:
    """The Mojo `DType` that stores elements of the `DT` with this code.
    `bool` is stored as `uint8` holding 0 or 1."""
    if code == 0 or code == 2:
        return DType.uint8
    if code == 1:
        return DType.int8
    if code == 3:
        return DType.int16
    if code == 4:
        return DType.uint16
    if code == 5:
        return DType.int32
    if code == 6:
        return DType.uint32
    if code == 7:
        return DType.int64
    if code == 8:
        return DType.uint64
    if code == 9:
        return DType.float32
    return DType.float64


# ---- unary


def _unary[dt: DType](op: Int, a: BytePtr, o: BytePtr, n: Int):
    comptime W = simd_width_of[dt]()
    var pa = a.unsafe_bitcast[Scalar[dt]]()
    var po = o.unsafe_bitcast[Scalar[dt]]()
    var i = 0
    if op == OP_NEG:
        while i < n:
            po.unsafe_store(i, -pa.unsafe_load[width=W, alignment=1](i))
            i += W
        return
    comptime if dt.is_floating_point():
        if op == OP_EXP:
            while i < n:
                po.unsafe_store(i, exp(pa.unsafe_load[width=W, alignment=1](i)))
                i += W
        elif op == OP_LOG:
            while i < n:
                po.unsafe_store(i, log(pa.unsafe_load[width=W, alignment=1](i)))
                i += W
        elif op == OP_SIN:
            while i < n:
                po.unsafe_store(i, sin(pa.unsafe_load[width=W, alignment=1](i)))
                i += W
        elif op == OP_SQRT:
            while i < n:
                po.unsafe_store(
                    i, sqrt(pa.unsafe_load[width=W, alignment=1](i))
                )
                i += W
        elif op == OP_RECIP:
            var one = SIMD[dt, W](1)
            while i < n:
                po.unsafe_store(
                    i, one / pa.unsafe_load[width=W, alignment=1](i)
                )
                i += W
        elif op == OP_TRUNC:
            while i < n:
                po.unsafe_store(
                    i, trunc(pa.unsafe_load[width=W, alignment=1](i))
                )
                i += W


def unary_block(dt: DT, op: Int, a: BytePtr, o: BytePtr, n: Int):
    """Writes `op(a[i])` to `o[i]` for a block of `n` elements of `dt`."""
    comptime for c in range(1, DT.COUNT):
        if dt.code == c:
            _unary[mojo_dtype(c)](op, a, o, n)
            return


# ---- binary


def _binary[dt: DType](op: Int, a: BytePtr, b: BytePtr, o: BytePtr, n: Int):
    comptime W = simd_width_of[dt]()
    var pa = a.unsafe_bitcast[Scalar[dt]]()
    var pb = b.unsafe_bitcast[Scalar[dt]]()
    var po = o.unsafe_bitcast[Scalar[dt]]()
    var i = 0
    if op == OP_ADD:
        while i < n:
            po.unsafe_store(
                i,
                pa.unsafe_load[width=W, alignment=1](i)
                + pb.unsafe_load[width=W, alignment=1](i),
            )
            i += W
    elif op == OP_SUB:
        while i < n:
            po.unsafe_store(
                i,
                pa.unsafe_load[width=W, alignment=1](i)
                - pb.unsafe_load[width=W, alignment=1](i),
            )
            i += W
    elif op == OP_MUL:
        while i < n:
            po.unsafe_store(
                i,
                pa.unsafe_load[width=W, alignment=1](i)
                * pb.unsafe_load[width=W, alignment=1](i),
            )
            i += W
    elif op == OP_MAX:
        while i < n:
            po.unsafe_store(
                i,
                max(
                    pa.unsafe_load[width=W, alignment=1](i),
                    pb.unsafe_load[width=W, alignment=1](i),
                ),
            )
            i += W
    elif op == OP_DIV:
        comptime if dt.is_floating_point():
            while i < n:
                po.unsafe_store(
                    i,
                    pa.unsafe_load[width=W, alignment=1](i)
                    / pb.unsafe_load[width=W, alignment=1](i),
                )
                i += W
        else:
            # Truncating division, as in C; x / 0 is 0. Exactly n
            # elements: a stray lane must not divide by zero.
            while i < n:
                var d = pb[unsafe_offset=i]
                var x = pa[unsafe_offset=i]
                if d == 0:
                    po[unsafe_offset=i] = 0
                else:
                    var q = x // d
                    if (x % d != 0) and ((x < 0) != (d < 0)):
                        q += 1
                    po[unsafe_offset=i] = q
                i += 1
    elif op == OP_MOD:
        comptime if dt.is_floating_point():
            while i < n:
                var x = pa.unsafe_load[width=W, alignment=1](i)
                var d = pb.unsafe_load[width=W, alignment=1](i)
                po.unsafe_store(i, x - trunc(x / d) * d)
                i += W
        else:
            # The remainder of truncating division, as in C; x % 0 is 0.
            while i < n:
                var d = pb[unsafe_offset=i]
                var x = pa[unsafe_offset=i]
                if d == 0:
                    po[unsafe_offset=i] = 0
                else:
                    var q = x // d
                    if (x % d != 0) and ((x < 0) != (d < 0)):
                        q += 1
                    po[unsafe_offset=i] = x - q * d
                i += 1
    elif op == OP_POW:
        comptime if dt.is_floating_point():
            while i < n:
                po.unsafe_store(
                    i,
                    pa.unsafe_load[width=W, alignment=1](i)
                    ** pb.unsafe_load[width=W, alignment=1](i),
                )
                i += W
    else:
        comptime if dt.is_integral():
            if op == OP_AND:
                while i < n:
                    po.unsafe_store(
                        i,
                        pa.unsafe_load[width=W, alignment=1](i)
                        & pb.unsafe_load[width=W, alignment=1](i),
                    )
                    i += W
            elif op == OP_OR:
                while i < n:
                    po.unsafe_store(
                        i,
                        pa.unsafe_load[width=W, alignment=1](i)
                        | pb.unsafe_load[width=W, alignment=1](i),
                    )
                    i += W
            elif op == OP_XOR:
                while i < n:
                    po.unsafe_store(
                        i,
                        pa.unsafe_load[width=W, alignment=1](i)
                        ^ pb.unsafe_load[width=W, alignment=1](i),
                    )
                    i += W


def binary_block(dt: DT, op: Int, a: BytePtr, b: BytePtr, o: BytePtr, n: Int):
    """Writes `op(a[i], b[i])` to `o[i]` for a block of `n` elements of
    `dt`. Comparisons are `compare_block`."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            _binary[mojo_dtype(c)](op, a, b, o, n)
            return


def _compare[dt: DType](op: Int, a: BytePtr, b: BytePtr, o: BytePtr, n: Int):
    comptime W = simd_width_of[dt]()
    var pa = a.unsafe_bitcast[Scalar[dt]]()
    var pb = b.unsafe_bitcast[Scalar[dt]]()
    var po = o.unsafe_bitcast[UInt8]()
    var t = SIMD[DType.uint8, W](1)
    var f = SIMD[DType.uint8, W](0)
    var i = 0
    if op == OP_CMPLT:
        while i < n:
            var m = pa.unsafe_load[width=W, alignment=1](i).lt(
                pb.unsafe_load[width=W, alignment=1](i)
            )
            po.unsafe_store(i, m.select(t, f))
            i += W
    elif op == OP_CMPNE:
        while i < n:
            var m = pa.unsafe_load[width=W, alignment=1](i).ne(
                pb.unsafe_load[width=W, alignment=1](i)
            )
            po.unsafe_store(i, m.select(t, f))
            i += W
    else:
        while i < n:
            var m = pa.unsafe_load[width=W, alignment=1](i).eq(
                pb.unsafe_load[width=W, alignment=1](i)
            )
            po.unsafe_store(i, m.select(t, f))
            i += W


def compare_block(dt: DT, op: Int, a: BytePtr, b: BytePtr, o: BytePtr, n: Int):
    """Writes the boolean `op(a[i], b[i])` to the byte `o[i]`, where the
    operands are blocks of `n` elements of `dt`."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            _compare[mojo_dtype(c)](op, a, b, o, n)
            return


# ---- where and cast


def _where[dt: DType](c: BytePtr, a: BytePtr, b: BytePtr, o: BytePtr, n: Int):
    comptime W = simd_width_of[dt]()
    var pc = c.unsafe_bitcast[UInt8]()
    var pa = a.unsafe_bitcast[Scalar[dt]]()
    var pb = b.unsafe_bitcast[Scalar[dt]]()
    var po = o.unsafe_bitcast[Scalar[dt]]()
    var zero = SIMD[DType.uint8, W](0)
    var i = 0
    while i < n:
        var m = pc.unsafe_load[width=W, alignment=1](i).ne(zero)
        po.unsafe_store(
            i,
            m.select(
                pa.unsafe_load[width=W, alignment=1](i),
                pb.unsafe_load[width=W, alignment=1](i),
            ),
        )
        i += W


def where_block(dt: DT, c: BytePtr, a: BytePtr, b: BytePtr, o: BytePtr, n: Int):
    """Writes `a[i] if c[i] else b[i]` to `o[i]`; `c` is a block of
    bytes, the others blocks of `n` elements of `dt`."""
    comptime for k in range(DT.COUNT):
        if dt.code == k:
            _where[mojo_dtype(k)](c, a, b, o, n)
            return


def _cast[src: DType, dst: DType](a: BytePtr, o: BytePtr, n: Int):
    comptime W = min(simd_width_of[src](), simd_width_of[dst]())
    var pa = a.unsafe_bitcast[Scalar[src]]()
    var po = o.unsafe_bitcast[Scalar[dst]]()
    var i = 0
    while i < n:
        po.unsafe_store(i, pa.unsafe_load[width=W, alignment=1](i).cast[dst]())
        i += W


def _to_bool[src: DType](a: BytePtr, o: BytePtr, n: Int):
    comptime W = simd_width_of[src]()
    var pa = a.unsafe_bitcast[Scalar[src]]()
    var po = o.unsafe_bitcast[UInt8]()
    var zero = SIMD[src, W](0)
    var t = SIMD[DType.uint8, W](1)
    var f = SIMD[DType.uint8, W](0)
    var i = 0
    while i < n:
        var m = pa.unsafe_load[width=W, alignment=1](i).ne(zero)
        po.unsafe_store(i, m.select(t, f))
        i += W


def _cast_from[src: DType](dst: DT, a: BytePtr, o: BytePtr, n: Int):
    if dst.code == 0:
        _to_bool[src](a, o, n)
        return
    comptime for c in range(1, DT.COUNT):
        if dst.code == c:
            _cast[src, mojo_dtype(c)](a, o, n)
            return


def cast_block(src: DT, dst: DT, a: BytePtr, o: BytePtr, n: Int):
    """Converts a block of `n` elements of `src` at `a` to `dst` at `o`.
    Float to integer truncates; anything to `bool` tests for nonzero."""
    comptime for c in range(DT.COUNT):
        if src.code == c:
            _cast_from[mojo_dtype(c)](dst, a, o, n)
            return


# ---- loads


def _fill[dt: DType](o: BytePtr, bits: Int, is_float: Bool, n: Int):
    comptime W = simd_width_of[dt]()
    var po = o.unsafe_bitcast[Scalar[dt]]()
    var v: Scalar[dt]
    comptime if dt.is_floating_point():
        v = Scalar[dt](_bits_to_f64(bits))
    else:
        v = Scalar[dt](bits)
    var vec = SIMD[dt, W](v)
    var i = 0
    while i < n:
        po.unsafe_store(i, vec)
        i += W


def _bits_to_f64(bits: Int) -> Float64:
    return bitcast[DType.float64](Int64(bits))


def fill_block(dt: DT, o: BytePtr, bits: Int, n: Int):
    """Fills a block of `n` elements of `dt` with a constant given as a
    `CONST` node argument (float bits, or an integer value)."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            _fill[mojo_dtype(c)](o, bits, dt.is_float(), n)
            return


def _splat[dt: DType](a: BytePtr, o: BytePtr, n: Int):
    comptime W = simd_width_of[dt]()
    var po = o.unsafe_bitcast[Scalar[dt]]()
    var vec = SIMD[dt, W](a.unsafe_bitcast[Scalar[dt]]()[])
    var i = 0
    while i < n:
        po.unsafe_store(i, vec)
        i += W


def splat_block(dt: DT, a: BytePtr, o: BytePtr, n: Int):
    """Fills a block of `n` elements of `dt` with the element at `a`."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            _splat[mojo_dtype(c)](a, o, n)
            return


def _gather[dt: DType](a: BytePtr, stride: Int, o: BytePtr, n: Int):
    var pa = a.unsafe_bitcast[Scalar[dt]]()
    var po = o.unsafe_bitcast[Scalar[dt]]()
    for i in range(n):
        po[unsafe_offset=i] = pa[unsafe_offset=i * stride]


def gather_block(dt: DT, a: BytePtr, stride: Int, o: BytePtr, n: Int):
    """Copies `n` elements of `dt`, `stride` elements apart starting at
    `a`, to the block `o`. Touches exactly `n` elements."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            _gather[mojo_dtype(c)](a, stride, o, n)
            return


def _gather_table[
    dt: DType
](a: BytePtr, table: Pointer[Int, MutUntrackedOrigin], o: BytePtr, n: Int):
    var pa = a.unsafe_bitcast[Scalar[dt]]()
    var po = o.unsafe_bitcast[Scalar[dt]]()
    for i in range(n):
        var t = table[unsafe_offset=i]
        if t != MASKED:
            po[unsafe_offset=i] = pa[unsafe_offset=t]
        else:
            po[unsafe_offset=i] = 0


def gather_table_block(
    dt: DT,
    a: BytePtr,
    table: Pointer[Int, MutUntrackedOrigin],
    o: BytePtr,
    n: Int,
):
    """Copies the elements of `dt` at offsets `table[0..n]` from `a` to
    the block `o`; an offset of `MASKED` reads as zero. Touches exactly
    `n` elements."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            _gather_table[mojo_dtype(c)](a, table, o, n)
            return


# ---- output


def _accumulate[dt: DType](rop: Int, a: BytePtr, o: BytePtr, n: Int):
    comptime W = simd_width_of[dt]()
    var pa = a.unsafe_bitcast[Scalar[dt]]()
    var po = o.unsafe_bitcast[Scalar[dt]]()
    var i = 0
    if rop == REDUCE_SUM:
        while i + W <= n:
            po.unsafe_store(
                i,
                po.unsafe_load[width=W, alignment=1](i)
                + pa.unsafe_load[width=W, alignment=1](i),
            )
            i += W
        while i < n:
            po[unsafe_offset=i] += pa[unsafe_offset=i]
            i += 1
    else:
        while i + W <= n:
            po.unsafe_store(
                i,
                max(
                    po.unsafe_load[width=W, alignment=1](i),
                    pa.unsafe_load[width=W, alignment=1](i),
                ),
            )
            i += W
        while i < n:
            po[unsafe_offset=i] = max(po[unsafe_offset=i], pa[unsafe_offset=i])
            i += 1


def accumulate_block(dt: DT, rop: Int, a: BytePtr, o: BytePtr, n: Int):
    """Combines a block into the output: `o[i] = rop(o[i], a[i])` for
    exactly `n` elements of `dt`."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            _accumulate[mojo_dtype(c)](rop, a, o, n)
            return


def _reduce_into[dt: DType](rop: Int, a: BytePtr, o: BytePtr, n: Int):
    comptime W = simd_width_of[dt]()
    var pa = a.unsafe_bitcast[Scalar[dt]]()
    var po = o.unsafe_bitcast[Scalar[dt]]()
    var i = 0
    var acc = po[]
    if rop == REDUCE_SUM:
        var vacc = SIMD[dt, W](0)
        while i + W <= n:
            vacc += pa.unsafe_load[width=W, alignment=1](i)
            i += W
        acc += vacc.reduce_add()
        while i < n:
            acc += pa[unsafe_offset=i]
            i += 1
    else:
        if n >= W:
            var vacc = pa.unsafe_load[width=W, alignment=1](0)
            i = W
            while i + W <= n:
                vacc = max(vacc, pa.unsafe_load[width=W, alignment=1](i))
                i += W
            acc = max(acc, vacc.reduce_max())
        while i < n:
            acc = max(acc, pa[unsafe_offset=i])
            i += 1
    po[] = acc


def reduce_into_block(dt: DT, rop: Int, a: BytePtr, o: BytePtr, n: Int):
    """Reduces a block into one output element:
    `o[0] = rop(o[0], a[0], ..., a[n-1])`, elements of `dt`."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            _reduce_into[mojo_dtype(c)](rop, a, o, n)
            return


def _segment_reduce[
    dt: DType
](rop: Int, a: BytePtr, group: Int, o: BytePtr, groups: Int):
    var pa = a.unsafe_bitcast[Scalar[dt]]()
    var po = o.unsafe_bitcast[Scalar[dt]]()
    if rop == REDUCE_SUM:
        for q in range(groups):
            var acc = po[unsafe_offset=q]
            var row = pa.unsafe_offset(q * group)
            for r in range(group):
                acc += row[unsafe_offset=r]
            po[unsafe_offset=q] = acc
    else:
        for q in range(groups):
            var acc = po[unsafe_offset=q]
            var row = pa.unsafe_offset(q * group)
            for r in range(group):
                acc = max(acc, row[unsafe_offset=r])
            po[unsafe_offset=q] = acc


def segment_reduce_block(
    dt: DT, rop: Int, a: BytePtr, group: Int, o: BytePtr, groups: Int
):
    """Reduces consecutive runs of a block into the output:
    `o[q] = rop(o[q], a[q * group], ..., a[q * group + group - 1])` for
    `groups` output elements of `dt`."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            _segment_reduce[mojo_dtype(c)](rop, a, group, o, groups)
            return


def _fill_identity[dt: DType](rop: Int, o: BytePtr, n: Int):
    var po = o.unsafe_bitcast[Scalar[dt]]()
    var v = Scalar[dt](0)
    if rop == REDUCE_MAX:
        comptime if dt.is_floating_point():
            v = Scalar[dt](-1.0) / Scalar[dt](0.0)
        else:
            v = Scalar[dt].MIN
    for i in range(n):
        po[unsafe_offset=i] = v


def fill_identity(dt: DT, rop: Int, o: BytePtr, n: Int):
    """Sets `n` output elements of `dt` to the identity of `rop`: 0 for
    a sum, the least value for a max."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            _fill_identity[mojo_dtype(c)](rop, o, n)
            return


def copy_block(dt: DT, a: BytePtr, o: BytePtr, n: Int):
    """Copies exactly `n` elements of `dt` from `a` to `o`."""
    unsafe_memcpy(dest=o, src=a, count=n * dt.itemsize())
