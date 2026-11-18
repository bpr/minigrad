"""Handles to graph nodes and the operations that build them
(`minigrad.uop`).

A `UOp` is a counted reference to one node of the `Graph`. Its methods
append nodes: they check shapes and element types, and simplify as they
build (operations on constants fold, identities such as `x * 1` and
no-op movements return their input), so the graph a gradient produces
stays small. `Tensor` is a thin layer of broadcasting and type promotion
over these.
"""

from std.math import exp, log, sin, sqrt, trunc
from std.memory import bitcast

from minigrad.dtype import DT
from minigrad.graph import (
    graph,
    is_movement,
    OP_ADD,
    OP_AND,
    OP_BUFFER,
    OP_CAST,
    OP_CMPEQ,
    OP_CMPLT,
    OP_CMPNE,
    OP_CONST,
    OP_CONTIGUOUS,
    OP_CONV,
    OP_DETACH,
    OP_DIV,
    OP_EXP,
    OP_EXPAND,
    OP_FLIP,
    OP_LOG,
    OP_MAX,
    OP_MOD,
    OP_MUL,
    OP_NEG,
    OP_OR,
    OP_PAD,
    OP_PERMUTE,
    OP_POW,
    OP_RECIP,
    OP_REDUCE,
    OP_RESHAPE,
    OP_SHRINK,
    OP_SIN,
    OP_SQRT,
    OP_SUB,
    OP_TRUNC,
    OP_WHERE,
    OP_WINDOW,
    OP_XOR,
    REDUCE_MAX,
    REDUCE_SUM,
)
from minigrad.view import prod


def float_bits(v: Float64) -> Int:
    """The bit pattern of `v` as an `Int`."""
    return Int(bitcast[DType.int64](v))


def bits_float(b: Int) -> Float64:
    """The `Float64` whose bit pattern is `b`."""
    return bitcast[DType.float64](Int64(b))


def const_bits(dt: DT, v: Float64) -> Int:
    """The canonical argument of a `CONST` node of type `dt` and value `v`.

    Floats are stored as `Float64` bits (after rounding to `float32` for
    that type), integers as their value, and booleans as 0 or 1.
    """
    if dt == DT.float32:
        return float_bits(Float64(Float32(v)))
    if dt.is_float():
        return float_bits(v)
    if dt.is_bool():
        return 1 if v != 0 else 0
    return Int(v)


def bits_value(dt: DT, b: Int) -> Float64:
    """The value of a `CONST` argument, as a `Float64`."""
    if dt.is_float():
        return bits_float(b)
    return Float64(b)


def broadcast_shapes(a: List[Int], b: List[Int]) raises -> List[Int]:
    """The shape two shapes broadcast to, by NumPy's rules.

    Args:
        a: A shape.
        b: A shape.

    Returns:
        The common shape.

    Raises:
        If a pair of dimensions differ and neither is 1.
    """
    var n = max(len(a), len(b))
    var out = List[Int](length=n, fill=1)
    for i in range(n):
        var da = a[len(a) - n + i] if i >= n - len(a) else 1
        var db = b[len(b) - n + i] if i >= n - len(b) else 1
        if da != db and da != 1 and db != 1:
            raise Error(t"cannot broadcast shapes {a} and {b}")
        out[i] = db if da == 1 else da
    return out^


struct UOp(Equatable, ImplicitlyCopyable):
    """A counted reference to a node of the graph."""

    var id: Int
    """The node's index in the graph."""

    def __init__(out self, *, id: Int):
        """Takes a new reference to node `id`."""
        self.id = id
        graph()[].nodes[id].rc += 1

    def __init__(out self, *, copy: Self):
        self.id = copy.id
        graph()[].nodes[copy.id].rc += 1

    def __deinit__(deinit self):
        graph()[].decref(self.id)

    def __eq__(self, other: Self) -> Bool:
        """Whether both refer to the same node."""
        return self.id == other.id

    def __ne__(self, other: Self) -> Bool:
        return self.id != other.id

    # ---- accessors

    def op(self) -> Int:
        """The node's operation code."""
        return graph()[].nodes[self.id].op

    def dtype(self) -> DT:
        """The element type."""
        return graph()[].nodes[self.id].dtype

    def shape(self) -> List[Int]:
        """A copy of the shape."""
        return graph()[].nodes[self.id].shape.copy()

    def ndim(self) -> Int:
        """The number of dimensions."""
        return len(graph()[].nodes[self.id].shape)

    def size(self) -> Int:
        """The number of elements."""
        return prod(graph()[].nodes[self.id].shape)

    def arg(self) -> List[Int]:
        """A copy of the node's argument."""
        return graph()[].nodes[self.id].arg.copy()

    def src(self, i: Int) -> UOp:
        """Input `i` (0, 1, or 2) of the node."""
        var g = graph()
        if i == 0:
            return UOp(id=g[].nodes[self.id].s0)
        if i == 1:
            return UOp(id=g[].nodes[self.id].s1)
        return UOp(id=g[].nodes[self.id].s2)

    def is_realized(self) -> Bool:
        """Whether the node is a buffer, or movement operations over one:
        its elements exist in memory."""
        var g = graph()
        var n = self.id
        while is_movement(g[].nodes[n].op):
            n = g[].nodes[n].s0
        return g[].nodes[n].op == OP_BUFFER

    def const_value(self) -> Optional[Float64]:
        """The value, if every element is the same constant: a `CONST`
        under movement operations other than padding."""
        var g = graph()
        var n = self.id
        while True:
            var op = g[].nodes[n].op
            if op == OP_CONST:
                return bits_value(g[].nodes[n].dtype, g[].nodes[n].arg[0])
            if is_movement(op) and op != OP_PAD:
                n = g[].nodes[n].s0
            else:
                return None

    # ---- construction

    # Inputs are passed as handles, never as bare ids: a handle is
    # destroyed right after its last use, so `f(x.id)` could free node
    # `x` before `f` links it. A borrowed handle lives across the call.

    @staticmethod
    def _make0(
        op: Int, dtype: DT, var shape: List[Int], var arg: List[Int]
    ) -> UOp:
        return UOp(id=graph()[].add(op, dtype, shape^, -1, -1, -1, arg^))

    @staticmethod
    def _make1(
        op: Int, dtype: DT, var shape: List[Int], var arg: List[Int], a: UOp
    ) -> UOp:
        return UOp(id=graph()[].add(op, dtype, shape^, a.id, -1, -1, arg^))

    @staticmethod
    def _make2(
        op: Int,
        dtype: DT,
        var shape: List[Int],
        var arg: List[Int],
        a: UOp,
        b: UOp,
    ) -> UOp:
        return UOp(id=graph()[].add(op, dtype, shape^, a.id, b.id, -1, arg^))

    @staticmethod
    def _make3(
        op: Int,
        dtype: DT,
        var shape: List[Int],
        var arg: List[Int],
        a: UOp,
        b: UOp,
        c: UOp,
    ) -> UOp:
        return UOp(id=graph()[].add(op, dtype, shape^, a.id, b.id, c.id, arg^))

    @staticmethod
    def const(dt: DT, v: Float64, shape: List[Int]) -> UOp:
        """An array of `shape` whose elements all equal `v`.

        Args:
            dt: The element type.
            v: The value, converted to `dt`.
            shape: The shape.

        Returns:
            A `CONST` node reshaped and expanded to `shape`.
        """
        var c = UOp._make0(OP_CONST, dt, List[Int](), [const_bits(dt, v)])
        if len(shape) == 0:
            return c
        var ones = List[Int](length=len(shape), fill=1)
        var r = UOp._make1(OP_RESHAPE, dt, ones.copy(), ones^, c)
        if prod(shape) == 1:
            return r
        return UOp._make1(OP_EXPAND, dt, shape.copy(), shape.copy(), r)

    def const_like(self, v: Float64) -> UOp:
        """A constant with this node's shape and element type."""
        return UOp.const(self.dtype(), v, self.shape())

    @staticmethod
    def new_buffer(dt: DT, shape: List[Int]) -> UOp:
        """A realized array of uninitialized memory.

        Args:
            dt: The element type.
            shape: The shape.

        Returns:
            A `BUFFER` node owning new memory.
        """
        var g = graph()
        var b = g[].alloc_buffer(prod(shape) * dt.itemsize())
        g[].buffers[b].rc += 1
        return UOp._make0(OP_BUFFER, dt, shape.copy(), [b])

    @staticmethod
    def on_buffer(buffer: Int, dt: DT, shape: List[Int]) -> UOp:
        """A `BUFFER` node for the existing buffer `buffer`, sharing its
        memory."""
        graph()[].buffers[buffer].rc += 1
        return UOp._make0(OP_BUFFER, dt, shape.copy(), [buffer])

    # ---- movement

    def reshape(self, new_shape: List[Int]) raises -> UOp:
        """Reinterprets the elements, in row-major order, as `new_shape`."""
        var g = graph()
        if g[].nodes[self.id].shape == new_shape:
            return self
        for s in new_shape:
            if s < 0:
                raise Error(t"reshape to negative dimension: {new_shape}")
        if prod(new_shape) != self.size():
            raise Error(
                t"cannot reshape {self.shape()} to {new_shape}: sizes differ"
            )
        var c = self.const_value()
        if c:
            return UOp.const(self.dtype(), c.value(), new_shape)
        var base = self
        if g[].nodes[self.id].op == OP_RESHAPE:
            base = self.src(0)
            if g[].nodes[base.id].shape == new_shape:
                return base
        return UOp._make1(
            OP_RESHAPE, self.dtype(), new_shape.copy(), new_shape.copy(), base
        )

    def permute(self, axes: List[Int]) raises -> UOp:
        """Reorders dimensions: dimension `i` of the result is `axes[i]`."""
        var n = self.ndim()
        if len(axes) != n:
            raise Error(t"permute {axes} of a {n}-dimensional array")
        var seen = List[Bool](length=n, fill=False)
        var identity = True
        for i in range(n):
            var a = axes[i]
            if a < 0 or a >= n or seen[a]:
                raise Error(t"invalid permutation {axes}")
            seen[a] = True
            if a != i:
                identity = False
        if identity:
            return self
        var shape = List[Int](capacity=n)
        for a in axes:
            shape.append(graph()[].nodes[self.id].shape[a])
        var c = self.const_value()
        if c:
            return UOp.const(self.dtype(), c.value(), shape)
        return UOp._make1(OP_PERMUTE, self.dtype(), shape^, axes.copy(), self)

    def expand(self, new_shape: List[Int]) raises -> UOp:
        """Broadcasts size-1 dimensions to the sizes in `new_shape`."""
        var g = graph()
        if g[].nodes[self.id].shape == new_shape:
            return self
        var n = self.ndim()
        if len(new_shape) != n:
            raise Error(t"cannot expand {self.shape()} to {new_shape}")
        for d in range(n):
            var s = g[].nodes[self.id].shape[d]
            if s != new_shape[d] and s != 1:
                raise Error(t"cannot expand {self.shape()} to {new_shape}")
        var c = self.const_value()
        if c:
            return UOp.const(self.dtype(), c.value(), new_shape)
        return UOp._make1(
            OP_EXPAND, self.dtype(), new_shape.copy(), new_shape.copy(), self
        )

    def broadcast_to(self, shape: List[Int]) raises -> UOp:
        """Broadcasts to `shape`, adding leading dimensions as needed."""
        var g = graph()
        if g[].nodes[self.id].shape == shape:
            return self
        var n = self.ndim()
        if n > len(shape):
            raise Error(t"cannot broadcast {self.shape()} to {shape}")
        var padded = List[Int](length=len(shape) - n, fill=1)
        for s in g[].nodes[self.id].shape:
            padded.append(s)
        return self.reshape(padded).expand(shape)

    def pad(self, arg: List[Int]) raises -> UOp:
        """Pads with zeros: `arg[2d]` before and `arg[2d+1]` after
        dimension `d`."""
        var n = self.ndim()
        if len(arg) != 2 * n:
            raise Error(t"pad {arg} of a {n}-dimensional array")
        var any = False
        var shape = List[Int](capacity=n)
        for d in range(n):
            if arg[2 * d] < 0 or arg[2 * d + 1] < 0:
                raise Error(t"negative padding {arg}")
            if arg[2 * d] != 0 or arg[2 * d + 1] != 0:
                any = True
            shape.append(
                graph()[].nodes[self.id].shape[d] + arg[2 * d] + arg[2 * d + 1]
            )
        if not any:
            return self
        return UOp._make1(OP_PAD, self.dtype(), shape^, arg.copy(), self)

    def shrink(self, arg: List[Int]) raises -> UOp:
        """Keeps indices `[arg[2d], arg[2d+1])` of each dimension `d`."""
        var n = self.ndim()
        if len(arg) != 2 * n:
            raise Error(t"shrink {arg} of a {n}-dimensional array")
        var full = True
        var shape = List[Int](capacity=n)
        for d in range(n):
            var s = graph()[].nodes[self.id].shape[d]
            var lo = arg[2 * d]
            var hi = arg[2 * d + 1]
            if lo < 0 or hi > s or lo > hi:
                raise Error(t"shrink {arg} out of bounds for {self.shape()}")
            if lo != 0 or hi != s:
                full = False
            shape.append(hi - lo)
        if full:
            return self
        var c = self.const_value()
        if c:
            return UOp.const(self.dtype(), c.value(), shape)
        return UOp._make1(OP_SHRINK, self.dtype(), shape^, arg.copy(), self)

    def flip(self, axes: List[Int]) raises -> UOp:
        """Reverses each dimension `d` with `axes[d] != 0`."""
        var n = self.ndim()
        if len(axes) != n:
            raise Error(t"flip {axes} of a {n}-dimensional array")
        var any = False
        for d in range(n):
            if axes[d] != 0 and graph()[].nodes[self.id].shape[d] > 1:
                any = True
        if not any or Bool(self.const_value()):
            return self
        return UOp._make1(
            OP_FLIP, self.dtype(), self.shape(), axes.copy(), self
        )

    def window(
        self, kernel: List[Int], stride: List[Int], dilation: List[Int]
    ) raises -> UOp:
        """Sliding windows over the last `len(kernel)` dimensions; see
        `View.window`. The result has the leading dimensions, then the
        window positions, then the offsets within a window."""
        var nk = len(kernel)
        var n = self.ndim()
        if nk > n or len(stride) != nk or len(dilation) != nk:
            raise Error(t"window {kernel} of a {n}-dimensional array")
        var shape = List[Int]()
        var mine = self.shape()
        var arg = List[Int]()
        arg.append(nk)
        for i in range(n - nk):
            shape.append(mine[i])
        for j in range(nk):
            if kernel[j] < 1 or stride[j] < 1 or dilation[j] < 1:
                raise Error("window sizes, strides, and dilations must be >= 1")
            var span = dilation[j] * (kernel[j] - 1) + 1
            if span > mine[n - nk + j]:
                raise Error(
                    t"window of extent {span} over a dimension of size"
                    t" {mine[n - nk + j]}"
                )
            shape.append((mine[n - nk + j] - span) // stride[j] + 1)
        for j in range(nk):
            shape.append(kernel[j])
        for j in range(nk):
            arg.append(kernel[j])
        for j in range(nk):
            arg.append(stride[j])
        for j in range(nk):
            arg.append(dilation[j])
        var c = self.const_value()
        if c:
            return UOp.const(self.dtype(), c.value(), shape)
        return UOp._make1(OP_WINDOW, self.dtype(), shape^, arg^, self)

    # ---- elementwise

    def cast(self, dt: DT) -> UOp:
        """Converts the elements to `dt`."""
        if self.dtype() == dt:
            return self
        var c = self.const_value()
        if c:
            var v = c.value()
            if not dt.is_float() and self.dtype().is_float():
                v = trunc(v)
            return UOp.const(dt, v, self.shape())
        # Convert before a movement that does not drop elements: if the
        # cast is materialized (a contraction operand gemm cannot
        # convert), it is then the size of its source, not of a window
        # or a broadcast. Zeros a pad adds convert to zero either way.
        var op = self.op()
        if is_movement(op) and op != OP_SHRINK:
            return UOp._make1(
                op, dt, self.shape(), self.arg(), self.src(0).cast(dt)
            )
        return UOp._make1(OP_CAST, dt, self.shape(), List[Int](), self)

    def alu1(self, op: Int) raises -> UOp:
        """Applies the unary operation `op` to every element."""
        var dt = self.dtype()
        if op != OP_NEG and not dt.is_float():
            raise Error(t"unary op {op} needs a float input, got {dt}")
        if op == OP_NEG and dt.is_bool():
            raise Error("cannot negate a bool array")
        var c = self.const_value()
        if c:
            var v = c.value()
            var r = v
            if op == OP_NEG:
                r = -v
            elif op == OP_EXP:
                r = exp(v)
            elif op == OP_LOG:
                r = log(v)
            elif op == OP_SIN:
                r = sin(v)
            elif op == OP_SQRT:
                r = sqrt(v)
            elif op == OP_RECIP:
                r = 1.0 / v
            elif op == OP_TRUNC:
                r = trunc(v)
            return self.const_like(r)
        var g = graph()
        if op == OP_NEG and g[].nodes[self.id].op == OP_NEG:
            return self.src(0)
        return UOp._make1(op, dt, self.shape(), List[Int](), self)

    def alu2(self, op: Int, other: UOp) raises -> UOp:
        """Applies the binary operation `op` elementwise.

        Args:
            op: An `OP_ADD` ... `OP_XOR` code.
            other: The right operand, of the same shape and element type.

        Returns:
            The result; of type `bool` for comparisons.

        Raises:
            If the shapes or element types differ, or the operation is
            not defined for the element type.
        """
        var g = graph()
        var dt = self.dtype()
        if dt != other.dtype():
            raise Error(
                t"binary op on different dtypes {dt} and {other.dtype()}"
            )
        if g[].nodes[self.id].shape != g[].nodes[other.id].shape:
            raise Error(
                t"binary op on shapes {self.shape()} and {other.shape()}"
            )
        var is_cmp = op == OP_CMPLT or op == OP_CMPNE or op == OP_CMPEQ
        var is_bitwise = op == OP_AND or op == OP_OR or op == OP_XOR
        if is_bitwise and dt.is_float():
            raise Error("bitwise op on a float array")
        if dt.is_bool() and not (is_cmp or is_bitwise or op == OP_MAX):
            if op == OP_ADD:
                return self.alu2(OP_OR, other)
            if op == OP_MUL:
                return self.alu2(OP_AND, other)
            raise Error(t"arithmetic op {op} on a bool array")
        var out_dt = DT.bool if is_cmp else dt

        var cx = self.const_value()
        var cy = other.const_value()
        if cx and cy and (dt.is_float() or dt.is_int()):
            var a = cx.value()
            var b = cy.value()
            var folded = True
            var r = 0.0
            if op == OP_ADD:
                r = a + b
            elif op == OP_SUB:
                r = a - b
            elif op == OP_MUL:
                r = a * b
            elif op == OP_DIV and dt.is_float():
                r = a / b
            elif op == OP_MAX:
                r = max(a, b)
            elif op == OP_CMPLT:
                r = 1.0 if a < b else 0.0
            elif op == OP_CMPNE:
                r = 1.0 if a != b else 0.0
            elif op == OP_CMPEQ:
                r = 1.0 if a == b else 0.0
            else:
                folded = False
            if folded:
                return UOp.const(out_dt, r, self.shape())
        if cy:
            var b = cy.value()
            if b == 0 and (op == OP_ADD or op == OP_SUB):
                return self
            if b == 1 and (op == OP_MUL or op == OP_DIV):
                return self
            if b == 0 and op == OP_MUL:
                return other
        if cx:
            var a = cx.value()
            if a == 0 and op == OP_ADD:
                return other
            if a == 1 and op == OP_MUL:
                return other
            if a == 0 and op == OP_MUL:
                return self
        return UOp._make2(op, out_dt, self.shape(), List[Int](), self, other)

    def where(self, a: UOp, b: UOp) raises -> UOp:
        """Selects `a` where this (boolean) array is true, else `b`."""
        var g = graph()
        if not self.dtype().is_bool():
            raise Error("where condition must be bool")
        if a.dtype() != b.dtype():
            raise Error(
                t"where branches have dtypes {a.dtype()} and {b.dtype()}"
            )
        if (
            g[].nodes[self.id].shape != g[].nodes[a.id].shape
            or g[].nodes[self.id].shape != g[].nodes[b.id].shape
        ):
            raise Error(
                t"where on shapes {self.shape()}, {a.shape()}, {b.shape()}"
            )
        var c = self.const_value()
        if c:
            return a if c.value() != 0 else b
        if a == b:
            return a
        return UOp._make3(
            OP_WHERE, a.dtype(), self.shape(), List[Int](), self, a, b
        )

    # ---- reduce

    def reduce(self, rop: Int, axes: List[Int]) raises -> UOp:
        """Reduces over `axes`, keeping them as size-1 dimensions.

        Args:
            rop: `REDUCE_SUM` or `REDUCE_MAX`.
            axes: The distinct dimensions to reduce, in increasing order.

        Returns:
            The reduced array, of the same element type.
        """
        var n = self.ndim()
        var shape = self.shape()
        var kept = List[Int]()
        var count = 1
        for a in axes:
            if a < 0 or a >= n:
                raise Error(t"reduce axis {a} out of range for {shape}")
            if shape[a] != 1:
                kept.append(a)
                count *= shape[a]
                shape[a] = 1
        if len(kept) == 0:
            return self
        if count == 0:
            raise Error("reduce over an empty dimension")
        var c = self.const_value()
        if c:
            var v = c.value()
            if rop == REDUCE_SUM:
                v *= Float64(count)
            return UOp.const(self.dtype(), v, shape)
        var arg = List[Int](capacity=1 + len(kept))
        arg.append(rop)
        for a in kept:
            arg.append(a)
        return UOp._make1(OP_REDUCE, self.dtype(), shape^, arg^, self)

    # ---- other

    def contiguous(self) -> UOp:
        """Marks the array to be materialized in its own memory. The
        result is a new node every time, unless this already is a
        buffer or the result of `contiguous`."""
        var op = self.op()
        if op == OP_BUFFER or op == OP_CONTIGUOUS:
            return self
        return self.fresh()

    def fresh(self) -> UOp:
        """A node with this node's value that no other handle refers to:
        a copy, when realized."""
        return UOp._make1(
            OP_CONTIGUOUS, self.dtype(), self.shape(), List[Int](), self
        )

    def as_conv(self, x: UOp, w: UOp, arg: List[Int]) -> UOp:
        """Records that this node is the convolution of `x` by `w`.

        Args:
            x: The input, before padding.
            w: The weight.
            arg: The groups, stride, dilation, and padding.

        Returns:
            A node with this node's value that `gradient` differentiates
            as a convolution.
        """
        return UOp._make3(
            OP_CONV, self.dtype(), self.shape(), arg.copy(), self, x, w
        )

    def detach(self) -> UOp:
        """The same values, treated as a constant by `gradient`."""
        var op = self.op()
        if op == OP_BUFFER or op == OP_DETACH or op == OP_CONST:
            return self
        return UOp._make1(
            OP_DETACH, self.dtype(), self.shape(), List[Int](), self
        )
