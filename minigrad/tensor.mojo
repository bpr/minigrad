"""The tensor API (`minigrad.tensor`).

A `Tensor` is a handle to a lazy node of the graph. Operations build
nodes and return at once; `realize()` (or reading data with `tolist()` or
`item()`) schedules and runs the kernels that compute the values.

Handles behave like Python references, as tinygrad's tensors do: a copy
of a `Tensor` is the same tensor, so assigning to one (`assign`, `+=`) or
storing a gradient in it is seen through every copy. Values, on the other
hand, are immutable: assigning points the tensor at a new value and never
overwrites memory another tensor reads. The API follows tinygrad's names
and semantics.
"""

from std.math import cos, log, pi, sqrt
from std.ffi import _Global
from std.memory import ArcPointer, unsafe_memcpy
from std.os import abort
from std.utils.numerics import inf

from minigrad.dtype import DT, promote, sum_acc_dtype
from minigrad.gradient import compute_gradients
from minigrad.graph import (
    BytePtr,
    graph,
    OP_ADD,
    OP_AND,
    OP_BUFFER,
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
from minigrad.runtime.blockops import mojo_dtype
from minigrad.runtime.kernel import (
    Instr,
    JitStep,
    STEP_COPY,
    STEP_GATHER,
    run_copy,
    run_gather,
    runtime,
)
from minigrad.schedule import realize
from minigrad.uop import UOp, broadcast_shapes
from minigrad.view import prod


def _read[dt: DType](p: BytePtr, i: Int) -> Float64:
    return Float64(p.unsafe_bitcast[Scalar[dt]]()[unsafe_offset=i])


def read_element(dt: DT, p: BytePtr, i: Int) -> Float64:
    """Element `i` of a buffer of `dt` at `p`, as a `Float64`."""
    comptime for c in range(DT.COUNT):
        if dt.code == c:
            return _read[mojo_dtype(c)](p, i)
    return 0.0


def _record(var s: JitStep):
    """Appends an eager operation to a capturing `TinyJit`'s tape."""
    var rt = runtime()
    if not rt[].recording:
        return
    var g = graph()
    for b in s.ins:
        g[].hold(b)
    g[].hold(s.out)
    rt[].tape.append(s^)


struct JitState(Movable):
    """The tensors a capturing `TinyJit` has seen assigned, with their
    values before the step."""

    var tensors: List[Tensor]
    var before: List[UOp]

    def __init__(out self):
        self.tensors = List[Tensor]()
        self.before = List[UOp]()


def _init_jit_state() -> JitState:
    return JitState()


comptime _JIT_STATE = _Global["minigrad_jit_state", _init_jit_state]


def jit_state() -> Pointer[JitState, MutUntrackedOrigin]:
    """The assignments of the current `TinyJit` capture."""
    try:
        return _JIT_STATE.get_or_create_ptr()
    except:
        abort("minigrad: cannot create the jit state")


def _read_all[dt: DType](p: BytePtr, n: Int, mut out: List[Float64]):
    var q = p.unsafe_bitcast[Scalar[dt]]()
    for i in range(n):
        out.append(Float64(q[unsafe_offset=i]))


def _write[dt: DType](p: BytePtr, i: Int, v: Float64):
    p.unsafe_bitcast[Scalar[dt]]()[unsafe_offset=i] = Scalar[dt](v)


def write_element(dt: DT, p: BytePtr, i: Int, v: Float64):
    """Stores `v`, converted to `dt`, as element `i` of a buffer at `p`."""
    if dt.is_bool():
        p[unsafe_offset=i] = 1 if v != 0 else 0
        return
    comptime for c in range(1, DT.COUNT):
        if dt.code == c:
            _write[mojo_dtype(c)](p, i, v)
            return


def _next_u64() -> UInt64:
    """The next value of the global generator (splitmix64)."""
    var g = graph()
    g[].rng_state += 0x9E3779B97F4A7C15
    var z = g[].rng_state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _next_uniform() -> Float64:
    """A uniform value in [0, 1) with 53 random bits."""
    return Float64(_next_u64() >> 11) * (1.0 / 9007199254740992.0)


struct TensorCell(Movable):
    """The state shared by every copy of a `Tensor`."""

    var uop: UOp
    """The node holding the tensor's value."""
    var requires_grad: Int
    """1 if `backward` computes a gradient for the tensor, 0 if not, -1
    if unset (an optimizer then turns it on)."""
    var slot: Int
    """The tensor's gradient slot in the graph, or -1."""

    def __init__(out self, uop: UOp):
        self.uop = uop
        self.requires_grad = -1
        self.slot = -1

    def __deinit__(deinit self):
        if self.slot >= 0:
            graph()[].free_grad_slot(self.slot)


struct Tensor(ImplicitlyCopyable, Writable):
    """A lazy n-dimensional array with automatic differentiation."""

    var cell: ArcPointer[TensorCell]

    # ---- construction

    def __init__(out self, uop: UOp):
        """Wraps a graph node."""
        self.cell = ArcPointer(TensorCell(uop))

    def __init__(out self, value: Float64):
        """A scalar `float32` tensor."""
        self.cell = ArcPointer(
            TensorCell(UOp.const(DT.default_float, value, List[Int]()))
        )

    def __init__(out self, data: List[Float64]):
        """A 1-dimensional `float32` tensor holding `data`."""
        self = Tensor.from_list(data, [len(data)], DT.default_float)

    def __init__(out self, data: List[List[Float64]]) raises:
        """A 2-dimensional `float32` tensor; `data` holds its rows."""
        var flat = List[Float64]()
        var cols = len(data[0]) if len(data) > 0 else 0
        for row in data:
            if len(row) != cols:
                raise Error("rows of a 2-d tensor have different lengths")
            for v in row:
                flat.append(v)
        self = Tensor.from_list(flat, [len(data), cols], DT.default_float)

    def __init__(out self, data: List[Int]):
        """A 1-dimensional `int32` tensor holding `data`."""
        var flat = List[Float64](capacity=len(data))
        for v in data:
            flat.append(Float64(v))
        self = Tensor.from_list(flat, [len(data)], DT.default_int)

    @staticmethod
    def from_list(
        data: List[Float64], shape: List[Int], dtype: DT = DT.default_float
    ) -> Tensor:
        """A realized tensor of `shape` holding `data` in row-major order.

        Args:
            data: The elements; `prod(shape)` of them are read.
            shape: The shape.
            dtype: The element type the values are converted to.

        Returns:
            The tensor.
        """
        var u = UOp.new_buffer(dtype, shape)
        var p = graph()[].buffer_ptr(u.id)
        for i in range(min(len(data), prod(shape))):
            write_element(dtype, p, i, data[i])
        return Tensor(u)

    @staticmethod
    def empty(shape: List[Int], dtype: DT = DT.default_float) -> Tensor:
        """A realized tensor of uninitialized memory; fill it through
        `data_ptr()`."""
        return Tensor(UOp.new_buffer(dtype, shape))

    @staticmethod
    def full(
        shape: List[Int], value: Float64, dtype: DT = DT.default_float
    ) -> Tensor:
        """A tensor of `shape` whose elements all equal `value`."""
        return Tensor(UOp.const(dtype, value, shape))

    @staticmethod
    def zeros(*shape: Int, dtype: DT = DT.default_float) -> Tensor:
        """A tensor of zeros."""
        return Tensor(UOp.const(dtype, 0.0, _list(shape)))

    @staticmethod
    def ones(*shape: Int, dtype: DT = DT.default_float) -> Tensor:
        """A tensor of ones."""
        return Tensor(UOp.const(dtype, 1.0, _list(shape)))

    def zeros_like(self) -> Tensor:
        """Zeros with this tensor's shape and element type."""
        return Tensor(self.uop().const_like(0.0))

    def ones_like(self) -> Tensor:
        """Ones with this tensor's shape and element type."""
        return Tensor(self.uop().const_like(1.0))

    def full_like(self, value: Float64) -> Tensor:
        """A constant with this tensor's shape and element type."""
        return Tensor(self.uop().const_like(value))

    @staticmethod
    def arange(stop: Int) -> Tensor:
        """The `int32` values `0, 1, ..., stop - 1`."""
        return Tensor.arange(0, stop, 1)

    @staticmethod
    def arange(start: Int, stop: Int, step: Int = 1) -> Tensor:
        """The `int32` values from `start` up to (not including) `stop`,
        `step` apart."""
        var n = max(0, (stop - start + step - (1 if step > 0 else -1)) // step)
        var u = UOp.new_buffer(DT.default_int, [n])
        var p = graph()[].buffer_ptr(u.id).unsafe_bitcast[Int32]()
        for i in range(n):
            p[unsafe_offset=i] = Int32(start + i * step)
        return Tensor(u)

    @staticmethod
    def eye(n: Int, dtype: DT = DT.default_float) raises -> Tensor:
        """The `n` by `n` identity matrix."""
        var r = Tensor.arange(n)
        return (r.reshape(n, 1).eq(r.reshape(1, n))).cast(dtype)

    # ---- random

    @staticmethod
    def manual_seed(seed: Int):
        """Seeds the generator behind `rand`, `randn`, and the others."""
        graph()[].rng_state = UInt64(seed) * 0x9E3779B97F4A7C15 + 1

    @staticmethod
    def rand(*shape: Int) -> Tensor:
        """Uniform `float32` samples in [0, 1)."""
        return Tensor.uniform(_list(shape), 0.0, 1.0)

    @staticmethod
    def uniform(
        shape: List[Int], low: Float64 = 0.0, high: Float64 = 1.0
    ) -> Tensor:
        """Uniform `float32` samples in [`low`, `high`)."""
        var u = UOp.new_buffer(DT.float32, shape)
        var p = graph()[].buffer_ptr(u.id).unsafe_bitcast[Float32]()
        for i in range(prod(shape)):
            var x = Float64(_next_u64() >> 40) * (1.0 / 16777216.0)
            p[unsafe_offset=i] = Float32(low + (high - low) * x)
        return Tensor(u)

    @staticmethod
    def randn(*shape: Int) -> Tensor:
        """Standard normal `float32` samples."""
        var sh = _list(shape)
        var u = UOp.new_buffer(DT.float32, sh)
        var p = graph()[].buffer_ptr(u.id).unsafe_bitcast[Float32]()
        var n = prod(sh)
        var i = 0
        while i < n:
            # Box-Muller: two normals from two uniforms.
            var u1 = 1.0 - _next_uniform()
            var u2 = _next_uniform()
            var r = sqrt(-2.0 * log(u1))
            p[unsafe_offset=i] = Float32(r * cos(2.0 * pi * u2))
            if i + 1 < n:
                p[unsafe_offset=i + 1] = Float32(
                    r * cos(2.0 * pi * u2 - pi / 2.0)
                )
            i += 2
        return Tensor(u)

    @staticmethod
    def randint(shape: List[Int], low: Int, high: Int) -> Tensor:
        """Uniform `int32` samples in [`low`, `high`)."""
        var u = UOp.new_buffer(DT.int32, shape)
        var p = graph()[].buffer_ptr(u.id).unsafe_bitcast[Int32]()
        var span = UInt64(high - low)
        for i in range(prod(shape)):
            p[unsafe_offset=i] = Int32(low + Int(_next_u64() % span))
        return Tensor(u)

    @staticmethod
    def kaiming_uniform(shape: List[Int], a: Float64 = 0.01) -> Tensor:
        """Uniform samples scaled for layers followed by a leaky ReLU of
        slope `a` (He initialization)."""
        var fan_in = 1
        for i in range(1, len(shape)):
            fan_in *= shape[i]
        var bound = (
            sqrt(3.0) * sqrt(2.0 / (1.0 + a * a)) / sqrt(Float64(fan_in))
        )
        return Tensor.uniform(shape, -bound, bound)

    @staticmethod
    def glorot_uniform(shape: List[Int]) -> Tensor:
        """Uniform samples scaled by fan-in plus fan-out (Xavier
        initialization)."""
        var fan = shape[0]
        var rest = 1
        for i in range(1, len(shape)):
            rest *= shape[i]
        var bound = sqrt(6.0 / Float64(fan + rest))
        return Tensor.uniform(shape, -bound, bound)

    # ---- properties

    def uop(self) -> UOp:
        """The graph node holding the tensor's value."""
        return self.cell[].uop

    def shape(self) -> List[Int]:
        """The size of each dimension."""
        return self.cell[].uop.shape()

    def dtype(self) -> DT:
        """The element type."""
        return self.cell[].uop.dtype()

    def ndim(self) -> Int:
        """The number of dimensions."""
        return self.cell[].uop.ndim()

    def numel(self) -> Int:
        """The number of elements."""
        return self.cell[].uop.size()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "<Tensor shape=",
            self.shape(),
            " dtype=",
            self.dtype(),
            " realized" if self.uop().is_realized() else " lazy",
            ">",
        )

    def _replace(self, u: UOp):
        """Points every copy of this tensor at node `u`."""
        self.cell[].uop = u
        if self.cell[].slot >= 0:
            graph()[].grad_slots[self.cell[].slot].node = u.id

    # ---- realization and data

    def realize(self) raises -> Tensor:
        """Computes the tensor's value now.

        Returns:
            This tensor.
        """
        realize([self.uop()])
        return self

    @staticmethod
    def realize_all(tensors: List[Tensor]) raises:
        """Computes the values of several tensors in one schedule, so
        work they share is done once."""
        var uops = List[UOp](capacity=len(tensors))
        for t in tensors:
            uops.append(t.uop())
        realize(uops)

    def _contiguous_uop(self) raises -> UOp:
        """A realized node holding the elements in row-major order."""
        var u = self.uop()
        if u.op() != OP_BUFFER:
            realize([u])
        if u.op() != OP_BUFFER:
            # Movement operations over a buffer: copy into order.
            u = u.contiguous()
            realize([u])
        return u

    def tolist(self) raises -> List[Float64]:
        """The elements in row-major order, realizing the tensor.

        Returns:
            Every element converted to `Float64`.
        """
        var u = self._contiguous_uop()
        var p = graph()[].buffer_ptr(u.id)
        var dt = u.dtype()
        var out = List[Float64](capacity=u.size())
        comptime for c in range(DT.COUNT):
            if dt.code == c:
                _read_all[mojo_dtype(c)](p, u.size(), out)
        return out^

    def item(self) raises -> Float64:
        """The single element of a one-element tensor.

        Raises:
            If the tensor does not have exactly one element.
        """
        if self.numel() != 1:
            raise Error(
                t"item() needs one element, the shape is {self.shape()}"
            )
        var u = self._contiguous_uop()
        return read_element(u.dtype(), graph()[].buffer_ptr(u.id), 0)

    def data_ptr(self) raises -> BytePtr:
        """The tensor's memory, realizing it in place.

        The pointer is valid while this tensor is alive and not
        reassigned. Write through it only to fill a tensor made by
        `Tensor.empty`: any other tensor's memory may be shared with
        tensors that have the same value.

        Raises:
            If the tensor is a view that is not its own buffer.
        """
        var u = self.uop()
        if u.op() != OP_BUFFER:
            realize([u])
        if u.op() != OP_BUFFER:
            raise Error("data_ptr() of a view; call contiguous() first")
        return graph()[].buffer_ptr(u.id)

    # ---- eager operations (outside the lazy graph)

    def take_rows(self, rows: List[Int]) raises -> Tensor:
        """The given indices of dimension 0, gathered now: an embedding
        lookup. See the overload taking a tensor of indices, which a
        `TinyJit` can replay with new indices; this one cannot be
        captured.

        Raises:
            If an index is out of range, or a `TinyJit` is capturing.
        """
        if runtime()[].recording:
            raise Error(
                "take_rows() with host indices inside a TinyJit: pass the"
                " indices as a tensor"
            )
        var idx = List[Float64](capacity=len(rows))
        for r in rows:
            idx.append(Float64(r))
        return self.take_rows(Tensor.from_list(idx, [len(rows)], DT.int32))

    def take_rows(self, rows: Tensor) raises -> Tensor:
        """The given indices of dimension 0, gathered now: an embedding
        lookup.

        This tensor and `rows` are realized first. The result is a new
        realized tensor; it is not differentiable with respect to this
        one.

        Args:
            rows: A 1-dimensional `int32` or `int64` tensor of indices
                into dimension 0.

        Returns:
            A tensor of shape `(len(rows), shape[1:]...)`.

        Raises:
            If an index is out of range.
        """
        var u = self._contiguous_uop()
        var shape = u.shape()
        if len(shape) == 0:
            raise Error("take_rows() of a scalar")
        var idx = rows._contiguous_uop()
        if idx.ndim() != 1:
            raise Error(t"take_rows(): indices of shape {idx.shape()}")
        var out_shape = shape.copy()
        out_shape[0] = idx.size()
        var res = UOp.new_buffer(u.dtype(), out_shape)
        var g = graph()
        var s = JitStep(
            STEP_GATHER,
            -1,
            List[Instr](),
            [g[].nodes[u.id].arg[0], g[].nodes[idx.id].arg[0]],
            g[].nodes[res.id].arg[0],
        )
        s.index_dt = idx.dtype()
        s.n = idx.size()
        s.row = u.size() // max(shape[0], 1) * u.dtype().itemsize()
        s.limit = shape[0]
        run_gather(
            g[].buffer_ptr(u.id),
            g[].buffer_ptr(idx.id),
            g[].buffer_ptr(res.id),
            s,
        )
        _record(s^)
        _ = u^  # keep the source handles alive past the gather
        _ = idx^
        return Tensor(res)

    def write_(self, dim: Int, start: Int, value: Tensor) raises:
        """Overwrites indices `[start, start + n)` of dimension `dim` with
        `value`, now. See the overload taking `start` as a tensor, which a
        `TinyJit` can replay at new positions; this one cannot be
        captured.

        Raises:
            If the shapes do not fit, the tensor does not own its buffer,
            or a `TinyJit` is capturing.
        """
        if runtime()[].recording:
            raise Error(
                "write_() at a host position inside a TinyJit: pass the"
                " position as a tensor"
            )
        var at: List[Float64] = [Float64(start)]
        self.write_(dim, Tensor.from_list(at, [1], DT.int32), value)

    def write_(self, dim: Int, start: Tensor, value: Tensor) raises:
        """Overwrites indices `[start, start + n)` of dimension `dim` with
        `value`, now.

        This is the one operation that changes memory in place, for
        state such as a key-value cache. The tensor must own its buffer:
        make it with `Tensor.empty` (or `zeros(...).contiguous()` and
        `realize()`). Views of it (slices, reshapes, ...) share its
        memory and see the new values; so do lazy tensors computed from
        it that are realized afterwards. Take a copy with
        `contiguous().realize()` to keep the old values.

        Args:
            dim: The dimension to write along.
            start: The first index written: an `int32` or `int64` tensor
                of one element.
            value: The new elements: this tensor's shape, with `n` in
                dimension `dim`. It is realized and converted to this
                tensor's element type.

        Raises:
            If the tensor does not own a buffer, or the shapes do not
            fit.
        """
        var g = graph()
        var u = self.uop()
        if u.op() != OP_BUFFER:
            raise Error("write_() needs a realized tensor with its own buffer")
        var buffer = g[].nodes[u.id].arg[0]
        if g[].buffers[buffer].rc - g[].buffers[buffer].held != 1:
            raise Error("write_() on a buffer shared with another tensor")
        var d = self._axis(dim)
        var shape = u.shape()
        var vshape = value.shape()
        if len(vshape) != len(shape):
            raise Error(t"write_() of shape {vshape} into {shape}")
        for i in range(len(shape)):
            if i != d and vshape[i] != shape[i]:
                raise Error(t"write_() of shape {vshape} into {shape}")
        var v = value.cast(u.dtype())._contiguous_uop()
        var at = start._contiguous_uop()
        if at.size() != 1:
            raise Error("write_(): the start must be a one-element tensor")
        var st = JitStep(
            STEP_COPY,
            -1,
            List[Instr](),
            [g[].nodes[v.id].arg[0], g[].nodes[at.id].arg[0]],
            buffer,
        )
        st.index_dt = at.dtype()
        st.n = vshape[d]
        st.limit = shape[d]
        st.outer = 1
        for i in range(d):
            st.outer *= shape[i]
        st.row = u.dtype().itemsize()
        for i in range(d + 1, len(shape)):
            st.row *= shape[i]
        run_copy(
            g[].buffer_ptr(v.id),
            g[].buffer_ptr(at.id),
            g[].buffer_ptr(u.id),
            st,
        )
        _record(st^)
        # `x.id` is not a use of `x`: keep the handles (and so their
        # buffers) alive past the copy.
        _ = v^
        _ = at^

    def to_string(self) raises -> String:
        """The elements as nested brackets, realizing the tensor."""
        var data = self.tolist()
        var shape = self.shape()
        var out = String("")
        var is_float = self.dtype().is_float()
        _format(out, data, shape, 0, 0, is_float)
        return out

    # ---- gradients

    def requires_grad_(self, value: Bool = True) -> Tensor:
        """Sets whether `backward` computes a gradient for this tensor.

        Returns:
            This tensor.
        """
        var g = graph()
        self.cell[].requires_grad = 1 if value else 0
        if value and self.cell[].slot < 0:
            # The tensor needs a node of its own. A constant would be
            # folded out of whatever is computed from it, leaving nothing
            # to differentiate with respect to; and two tensors built
            # from equal expressions are one node, so if another tensor
            # already takes gradients here each would get the sum.
            var id = self.cell[].uop.id
            var shared = Bool(self.cell[].uop.const_value())
            for s in range(len(g[].grad_slots)):
                if g[].grad_slots[s].node == id:
                    shared = True
            if shared:
                var own = self.uop().fresh()
                self.cell[].uop = own
            self.cell[].slot = g[].new_grad_slot(self.cell[].uop.id)
        elif not value and self.cell[].slot >= 0:
            g[].free_grad_slot(self.cell[].slot)
            self.cell[].slot = -1
        return self

    def requires_grad(self) -> Bool:
        """Whether `backward` computes a gradient for this tensor."""
        return self.cell[].requires_grad == 1

    def backward(self) raises:
        """Differentiates this scalar with respect to every tensor that
        requires a gradient, adding the results to their gradients.

        Gradients are lazy: they are computed when realized (for example
        by an optimizer's `step`).

        Raises:
            If the tensor is not a scalar, or is already realized: a
            realized tensor is a buffer, with no record of how it was
            computed.
        """
        var g = graph()
        var root = self.uop()
        if root.size() != 1:
            raise Error(
                t"backward() needs a scalar, the shape is {self.shape()}"
            )
        if root.is_realized():
            raise Error(
                "backward() on a realized tensor: call it before realize(),"
                " item(), or tolist()"
            )
        var targets = List[Int]()
        for s in range(len(g[].grad_slots)):
            var node = g[].grad_slots[s].node
            if node >= 0 and node not in targets:
                targets.append(node)
        var grads = compute_gradients(root, root.const_like(1.0), targets)
        for s in range(len(g[].grad_slots)):
            var node = g[].grad_slots[s].node
            if node < 0 or node not in grads:
                continue
            var new = grads[node]
            var old = g[].grad_slots[s].grad
            if old >= 0:
                new = UOp(id=old).alu2(OP_ADD, new)
            g[].incref(new.id)
            g[].grad_slots[s].grad = new.id
            if old >= 0:
                g[].decref(old)

    def grad(self) raises -> Tensor:
        """The gradient accumulated by `backward`.

        Raises:
            If the tensor has no gradient.
        """
        var slot = self.cell[].slot
        if slot < 0 or graph()[].grad_slots[slot].grad < 0:
            raise Error("tensor has no gradient")
        return Tensor(UOp(id=graph()[].grad_slots[slot].grad))

    def has_grad(self) -> Bool:
        """Whether `backward` has stored a gradient in this tensor."""
        var slot = self.cell[].slot
        return slot >= 0 and graph()[].grad_slots[slot].grad >= 0

    def zero_grad(self):
        """Discards the accumulated gradient."""
        var g = graph()
        var slot = self.cell[].slot
        if slot >= 0 and g[].grad_slots[slot].grad >= 0:
            var old = g[].grad_slots[slot].grad
            g[].grad_slots[slot].grad = -1
            g[].decref(old)

    def detach(self) -> Tensor:
        """The same values, treated as a constant by `backward`."""
        return Tensor(self.uop().detach())

    def gradient(self, targets: List[Tensor]) raises -> List[Tensor]:
        """Differentiates this scalar with respect to `targets`, without
        touching their stored gradients.

        Args:
            targets: The tensors to differentiate with respect to.

        Returns:
            One gradient per target.

        Raises:
            If this tensor does not depend on some target.
        """
        var root = self.uop()
        if root.size() != 1:
            raise Error(
                t"gradient() needs a scalar, the shape is {self.shape()}"
            )
        var ids = List[Int]()
        for t in targets:
            ids.append(t.uop().id)
        var grads = compute_gradients(root, root.const_like(1.0), ids)
        var out = List[Tensor]()
        for i in ids:
            if i not in grads:
                raise Error(
                    "gradient(): the tensor does not depend on a target"
                )
            out.append(Tensor(grads[i]))
        return out^

    # ---- assignment

    def assign(self, value: Tensor) raises -> Tensor:
        """Gives this tensor (every copy of it) a new value.

        The tensor is pointed at `value`'s node; no memory is
        overwritten, so tensors computed from the old value, realized or
        not, are unaffected. (tinygrad writes into a realized tensor's
        buffer instead; the values every tensor reads are the same.)

        Args:
            value: The new value, of the same shape and element type.

        Returns:
            This tensor.
        """
        var u = self.uop()
        var v = value.uop()
        if u.shape() != v.shape():
            raise Error(t"assign of shape {v.shape()} to {u.shape()}")
        if u.dtype() != v.dtype():
            raise Error(t"assign of dtype {v.dtype()} to {u.dtype()}")
        if runtime()[].recording:
            # A TinyJit capture remembers the value each tensor had
            # before the step, to copy the new value back into it.
            var st = jit_state()
            var seen = False
            for t in st[].tensors:
                if t.cell is self.cell:
                    seen = True
            if not seen:
                st[].tensors.append(self)
                st[].before.append(u)
        self._replace(v)
        return self

    # ---- elementwise helpers

    def _binary(self, op: Int, other: Tensor) raises -> Tensor:
        """Broadcasts and promotes both operands, then applies `op`."""
        var a = self.uop()
        var b = other.uop()
        var dt = promote(a.dtype(), b.dtype())
        var shape = broadcast_shapes(a.shape(), b.shape())
        return Tensor(
            a.cast(dt)
            .broadcast_to(shape)
            .alu2(op, b.cast(dt).broadcast_to(shape))
        )

    def _scalar(self, v: Float64, is_int: Bool) -> Tensor:
        """The scalar `v` as a tensor to combine with this one: of this
        tensor's type, unless that would lose the scalar's kind."""
        var dt = self.dtype()
        if dt.is_float():
            return Tensor(UOp.const(dt, v, List[Int]()))
        if is_int:
            if dt.is_bool():
                dt = DT.default_int
            return Tensor(UOp.const(dt, v, List[Int]()))
        return Tensor(UOp.const(DT.default_float, v, List[Int]()))

    def _unary(self, op: Int) raises -> Tensor:
        """Applies a float operation, converting integers to `float32`."""
        var u = self.uop()
        if not u.dtype().is_float():
            u = u.cast(DT.default_float)
        return Tensor(u.alu1(op))

    # ---- arithmetic

    def add(self, other: Tensor) raises -> Tensor:
        """Elementwise sum, with broadcasting."""
        return self._binary(OP_ADD, other)

    def sub(self, other: Tensor) raises -> Tensor:
        """Elementwise difference, with broadcasting."""
        return self._binary(OP_SUB, other)

    def mul(self, other: Tensor) raises -> Tensor:
        """Elementwise product, with broadcasting."""
        return self._binary(OP_MUL, other)

    def div(self, other: Tensor) raises -> Tensor:
        """Elementwise true division; integers divide as `float32`."""
        var a = self
        var b = other
        if not promote(a.dtype(), b.dtype()).is_float():
            a = a.cast(DT.default_float)
            b = b.cast(DT.default_float)
        return a._binary(OP_DIV, b)

    def idiv(self, other: Tensor) raises -> Tensor:
        """Elementwise integer division, truncating toward zero."""
        if promote(self.dtype(), other.dtype()).is_float():
            return self._binary(OP_DIV, other).trunc()
        return self._binary(OP_DIV, other)

    def mod(self, other: Tensor) raises -> Tensor:
        """Elementwise remainder of truncating division."""
        return self._binary(OP_MOD, other)

    def maximum(self, other: Tensor) raises -> Tensor:
        """Elementwise maximum, with broadcasting."""
        return self._binary(OP_MAX, other)

    def minimum(self, other: Tensor) raises -> Tensor:
        """Elementwise minimum, with broadcasting."""
        return self.neg().maximum(other.neg()).neg()

    def neg(self) raises -> Tensor:
        """Elementwise negation."""
        var u = self.uop()
        if u.dtype().is_bool():
            return self.logical_not()
        return Tensor(u.alu1(OP_NEG))

    def __add__(self, rhs: Tensor) raises -> Tensor:
        return self._binary(OP_ADD, rhs)

    def __add__(self, rhs: Float64) raises -> Tensor:
        return self._binary(OP_ADD, self._scalar(rhs, False))

    def __add__(self, rhs: Int) raises -> Tensor:
        return self._binary(OP_ADD, self._scalar(Float64(rhs), True))

    def __radd__(self, lhs: Float64) raises -> Tensor:
        return self._scalar(lhs, False)._binary(OP_ADD, self)

    def __radd__(self, lhs: Int) raises -> Tensor:
        return self._scalar(Float64(lhs), True)._binary(OP_ADD, self)

    def __sub__(self, rhs: Tensor) raises -> Tensor:
        return self._binary(OP_SUB, rhs)

    def __sub__(self, rhs: Float64) raises -> Tensor:
        return self._binary(OP_SUB, self._scalar(rhs, False))

    def __sub__(self, rhs: Int) raises -> Tensor:
        return self._binary(OP_SUB, self._scalar(Float64(rhs), True))

    def __rsub__(self, lhs: Float64) raises -> Tensor:
        return self._scalar(lhs, False)._binary(OP_SUB, self)

    def __rsub__(self, lhs: Int) raises -> Tensor:
        return self._scalar(Float64(lhs), True)._binary(OP_SUB, self)

    def __mul__(self, rhs: Tensor) raises -> Tensor:
        return self._binary(OP_MUL, rhs)

    def __mul__(self, rhs: Float64) raises -> Tensor:
        return self._binary(OP_MUL, self._scalar(rhs, False))

    def __mul__(self, rhs: Int) raises -> Tensor:
        return self._binary(OP_MUL, self._scalar(Float64(rhs), True))

    def __rmul__(self, lhs: Float64) raises -> Tensor:
        return self._scalar(lhs, False)._binary(OP_MUL, self)

    def __rmul__(self, lhs: Int) raises -> Tensor:
        return self._scalar(Float64(lhs), True)._binary(OP_MUL, self)

    def __truediv__(self, rhs: Tensor) raises -> Tensor:
        return self.div(rhs)

    def __truediv__(self, rhs: Float64) raises -> Tensor:
        return self.div(self._scalar(rhs, False))

    def __truediv__(self, rhs: Int) raises -> Tensor:
        return self.div(self._scalar(Float64(rhs), True))

    def __rtruediv__(self, lhs: Float64) raises -> Tensor:
        return self._scalar(lhs, False).div(self)

    def __rtruediv__(self, lhs: Int) raises -> Tensor:
        return self._scalar(Float64(lhs), True).div(self)

    def __floordiv__(self, rhs: Tensor) raises -> Tensor:
        return self.idiv(rhs)

    def __floordiv__(self, rhs: Int) raises -> Tensor:
        return self.idiv(self._scalar(Float64(rhs), True))

    def __mod__(self, rhs: Tensor) raises -> Tensor:
        return self.mod(rhs)

    def __mod__(self, rhs: Int) raises -> Tensor:
        return self.mod(self._scalar(Float64(rhs), True))

    def __neg__(self) raises -> Tensor:
        return self.neg()

    def __matmul__(self, rhs: Tensor) raises -> Tensor:
        return self.dot(rhs)

    def __iadd__(mut self, rhs: Tensor) raises:
        self._replace((self + rhs).uop())

    def __iadd__(mut self, rhs: Float64) raises:
        self._replace((self + rhs).uop())

    def __isub__(mut self, rhs: Tensor) raises:
        self._replace((self - rhs).uop())

    def __isub__(mut self, rhs: Float64) raises:
        self._replace((self - rhs).uop())

    def __imul__(mut self, rhs: Tensor) raises:
        self._replace((self * rhs).uop())

    def __imul__(mut self, rhs: Float64) raises:
        self._replace((self * rhs).uop())

    def __pow__(self, e: Float64) raises -> Tensor:
        return self.pow(e)

    def __pow__(self, e: Int) raises -> Tensor:
        return self.pow(Float64(e))

    def __pow__(self, e: Tensor) raises -> Tensor:
        return self.pow(e)

    def pow(self, e: Float64) raises -> Tensor:
        """Raises every element to the constant power `e`.

        Small integer powers multiply, and `0.5`, `-0.5`, and `-1` use
        `sqrt` and `reciprocal`, as in tinygrad.
        """
        if e == 0.0:
            return self.ones_like()
        if e == 1.0:
            return self
        if e == 2.0:
            return self * self
        if e == 3.0:
            return self * self * self
        if e == 0.5:
            return self.sqrt()
        if e == -0.5:
            return self.rsqrt()
        if e == -1.0:
            return self.reciprocal()
        if e == -2.0:
            return (self * self).reciprocal()
        var x = self
        if not x.dtype().is_float():
            x = x.cast(DT.default_float)
        return x._binary(OP_POW, x._scalar(e, False))

    def pow(self, e: Tensor) raises -> Tensor:
        """Elementwise power, with broadcasting; computed in floating
        point."""
        var x = self
        var y = e
        if not promote(x.dtype(), y.dtype()).is_float():
            x = x.cast(DT.default_float)
        return x._binary(OP_POW, y)

    # ---- comparison and logic

    def lt(self, other: Tensor) raises -> Tensor:
        """Elementwise `self < other`, a `bool` tensor."""
        return self._binary(OP_CMPLT, other)

    def gt(self, other: Tensor) raises -> Tensor:
        """Elementwise `self > other`, a `bool` tensor."""
        return other._binary(OP_CMPLT, self)

    def le(self, other: Tensor) raises -> Tensor:
        """Elementwise `self <= other`, a `bool` tensor."""
        return self.gt(other).logical_not()

    def ge(self, other: Tensor) raises -> Tensor:
        """Elementwise `self >= other`, a `bool` tensor."""
        return self.lt(other).logical_not()

    def eq(self, other: Tensor) raises -> Tensor:
        """Elementwise `self == other`, a `bool` tensor."""
        return self._binary(OP_CMPEQ, other)

    def ne(self, other: Tensor) raises -> Tensor:
        """Elementwise `self != other`, a `bool` tensor."""
        return self._binary(OP_CMPNE, other)

    def lt(self, other: Float64) raises -> Tensor:
        return self.lt(self._scalar(other, False))

    def gt(self, other: Float64) raises -> Tensor:
        return self.gt(self._scalar(other, False))

    def le(self, other: Float64) raises -> Tensor:
        return self.le(self._scalar(other, False))

    def ge(self, other: Float64) raises -> Tensor:
        return self.ge(self._scalar(other, False))

    def eq(self, other: Float64) raises -> Tensor:
        return self.eq(self._scalar(other, False))

    def ne(self, other: Float64) raises -> Tensor:
        return self.ne(self._scalar(other, False))

    def logical_not(self) raises -> Tensor:
        """Elementwise `not`: true where the element is zero."""
        var u = self.uop()
        if u.dtype().is_bool():
            return Tensor(u.alu2(OP_XOR, u.const_like(1.0)))
        return Tensor(u.alu2(OP_CMPEQ, u.const_like(0.0)))

    def bool(self) raises -> Tensor:
        """True where the element is nonzero."""
        return self.cast(DT.bool)

    def where(self, a: Tensor, b: Tensor) raises -> Tensor:
        """Selects from `a` where this tensor is true (nonzero) and from
        `b` elsewhere, broadcasting all three."""
        var c = self.uop().cast(DT.bool)
        var x = a.uop()
        var y = b.uop()
        var dt = promote(x.dtype(), y.dtype())
        var shape = broadcast_shapes(
            c.shape(), broadcast_shapes(x.shape(), y.shape())
        )
        return Tensor(
            c.broadcast_to(shape).where(
                x.cast(dt).broadcast_to(shape), y.cast(dt).broadcast_to(shape)
            )
        )

    def where(self, a: Tensor, b: Float64) raises -> Tensor:
        return self.where(a, a._scalar(b, False))

    def where(self, a: Float64, b: Tensor) raises -> Tensor:
        return self.where(b._scalar(a, False), b)

    def where(self, a: Float64, b: Float64) raises -> Tensor:
        return self.where(Tensor(a), Tensor(b))

    # ---- math

    def cast(self, dtype: DT) -> Tensor:
        """Converts the elements to `dtype`."""
        if self.dtype() == dtype:
            return self
        return Tensor(self.uop().cast(dtype))

    def float(self) -> Tensor:
        """Converts the elements to `float32`."""
        return self.cast(DT.float32)

    def half(self) -> Tensor:
        """Converts the elements to `float16`."""
        return self.cast(DT.float16)

    def int(self) -> Tensor:
        """Converts the elements to `int32`, truncating."""
        return self.cast(DT.int32)

    def exp(self) raises -> Tensor:
        """Elementwise `e ** x`."""
        return self._unary(OP_EXP)

    def log(self) raises -> Tensor:
        """Elementwise natural logarithm."""
        return self._unary(OP_LOG)

    def sqrt(self) raises -> Tensor:
        """Elementwise square root."""
        return self._unary(OP_SQRT)

    def rsqrt(self) raises -> Tensor:
        """Elementwise `1 / sqrt(x)`."""
        return self._unary(OP_SQRT).reciprocal()

    def reciprocal(self) raises -> Tensor:
        """Elementwise `1 / x`."""
        return self._unary(OP_RECIP)

    def sin(self) raises -> Tensor:
        """Elementwise sine."""
        return self._unary(OP_SIN)

    def cos(self) raises -> Tensor:
        """Elementwise cosine."""
        return (pi / 2.0 - self).sin()

    def tan(self) raises -> Tensor:
        """Elementwise tangent."""
        return self.sin() / self.cos()

    def trunc(self) raises -> Tensor:
        """Rounds every element toward zero."""
        if not self.dtype().is_float():
            return self
        return self._unary(OP_TRUNC)

    def floor(self) raises -> Tensor:
        """Rounds every element down."""
        if not self.dtype().is_float():
            return self
        var b = self.trunc()
        return self.lt(b).where(b - 1.0, b)

    def round(self) raises -> Tensor:
        """Rounds every element to the nearest integer, ties to even."""
        if not self.dtype().is_float():
            return self
        var b = self.floor()
        var d = self - b
        var odd = (b - (b * 0.5).floor() * 2.0).ne(0.0)
        var up = d.gt(0.5).logical_or(d.eq(0.5).logical_and(odd))
        return up.where(b + 1.0, b)

    def logical_and(self, other: Tensor) raises -> Tensor:
        """Elementwise `and` of truth values."""
        return Tensor(self.bool().uop().alu2(OP_AND, other.bool().uop()))

    def logical_or(self, other: Tensor) raises -> Tensor:
        """Elementwise `or` of truth values."""
        return Tensor(self.bool().uop().alu2(OP_OR, other.bool().uop()))

    def ceil(self) raises -> Tensor:
        """Rounds every element up."""
        if not self.dtype().is_float():
            return self
        var b = self.trunc()
        return self.gt(b).where(b + 1.0, b)

    def square(self) raises -> Tensor:
        """Elementwise `x * x`."""
        return self * self

    def abs(self) raises -> Tensor:
        """Elementwise absolute value."""
        return self.lt(0.0).where(self.neg(), self)

    def sign(self) raises -> Tensor:
        """Elementwise sign: -1, 0, or 1."""
        return self.gt(0.0).cast(self.dtype()) - self.lt(0.0).cast(self.dtype())

    def relu(self) raises -> Tensor:
        """Elementwise `max(x, 0)`."""
        return self.maximum(self._scalar(0.0, True))

    def sigmoid(self) raises -> Tensor:
        """Elementwise logistic function `1 / (1 + exp(-x))`."""
        return (1.0 + self.neg().exp()).reciprocal()

    def tanh(self) raises -> Tensor:
        """Elementwise hyperbolic tangent."""
        return 2.0 * (2.0 * self).sigmoid() - 1.0

    def gelu(self) raises -> Tensor:
        """The Gaussian error linear unit (tanh approximation)."""
        var inner = sqrt(2.0 / pi) * (self + 0.044715 * self * self * self)
        return 0.5 * self * (1.0 + inner.tanh())

    def silu(self) raises -> Tensor:
        """The sigmoid linear unit `x * sigmoid(x)` (swish)."""
        return self * self.sigmoid()

    def leaky_relu(self, neg_slope: Float64 = 0.01) raises -> Tensor:
        """`x` where positive, `neg_slope * x` elsewhere."""
        return self.lt(0.0).where(self * neg_slope, self)

    def softplus(self, beta: Float64 = 1.0) raises -> Tensor:
        """Elementwise `log(1 + exp(beta * x)) / beta`."""
        return (1.0 + (self * beta).exp()).log() * (1.0 / beta)

    def clip(self, lo: Float64, hi: Float64) raises -> Tensor:
        """Limits every element to [`lo`, `hi`]."""
        return self.maximum(self._scalar(lo, False)).minimum(
            self._scalar(hi, False)
        )

    # ---- movement

    def _axis(self, axis: Int) raises -> Int:
        """`axis` as a nonnegative dimension index."""
        var n = self.ndim()
        var a = axis + n if axis < 0 else axis
        if a < 0 or a >= n:
            raise Error(t"axis {axis} out of range for {n} dimensions")
        return a

    def reshape(self, shape: List[Int]) raises -> Tensor:
        """Reinterprets the elements as `shape`; one entry may be -1 and
        is inferred."""
        var sh = shape.copy()
        var known = 1
        var infer = -1
        for i in range(len(sh)):
            if sh[i] == -1:
                if infer >= 0:
                    raise Error("only one dimension of a reshape can be -1")
                infer = i
            else:
                known *= sh[i]
        if infer >= 0:
            if known == 0 or self.numel() % known != 0:
                raise Error(t"cannot reshape {self.shape()} to {shape}")
            sh[infer] = self.numel() // known
        return Tensor(self.uop().reshape(sh))

    def reshape(self, *shape: Int) raises -> Tensor:
        return self.reshape(_list(shape))

    def flatten(self, start: Int = 0, end: Int = -1) raises -> Tensor:
        """Merges dimensions `start` through `end` into one."""
        if self.ndim() == 0:
            return self.reshape(1)
        var s = self._axis(start)
        var e = self._axis(end)
        var shape = self.shape()
        var out = List[Int]()
        var merged = 1
        for d in range(len(shape)):
            if d < s or d > e:
                out.append(shape[d])
            else:
                merged *= shape[d]
                if d == e:
                    out.append(merged)
        return Tensor(self.uop().reshape(out))

    def unsqueeze(self, dim: Int) raises -> Tensor:
        """Inserts a size-1 dimension at `dim`."""
        var n = self.ndim()
        var d = dim + n + 1 if dim < 0 else dim
        if d < 0 or d > n:
            raise Error(t"unsqueeze dim {dim} out of range")
        var shape = self.shape()
        shape.insert(d, 1)
        return Tensor(self.uop().reshape(shape))

    def squeeze(self, dim: Int) raises -> Tensor:
        """Removes dimension `dim` if its size is 1."""
        var d = self._axis(dim)
        var shape = self.shape()
        if shape[d] != 1:
            return self
        _ = shape.pop(d)
        return Tensor(self.uop().reshape(shape))

    def squeeze(self) raises -> Tensor:
        """Removes every size-1 dimension."""
        var out = List[Int]()
        for s in self.shape():
            if s != 1:
                out.append(s)
        return Tensor(self.uop().reshape(out))

    def permute(self, axes: List[Int]) raises -> Tensor:
        """Reorders dimensions: dimension `i` of the result is
        `axes[i]`."""
        var a = List[Int]()
        for x in axes:
            a.append(self._axis(x))
        return Tensor(self.uop().permute(a))

    def permute(self, *axes: Int) raises -> Tensor:
        return self.permute(_list(axes))

    def transpose(self, dim0: Int = 1, dim1: Int = 0) raises -> Tensor:
        """Swaps two dimensions."""
        var a = self._axis(dim0)
        var b = self._axis(dim1)
        var axes = List[Int]()
        for d in range(self.ndim()):
            axes.append(d)
        axes[a] = b
        axes[b] = a
        return Tensor(self.uop().permute(axes))

    def T(self) raises -> Tensor:
        """The transpose: dimensions in reverse order."""
        var axes = List[Int]()
        var n = self.ndim()
        for d in range(n):
            axes.append(n - 1 - d)
        return Tensor(self.uop().permute(axes))

    def expand(self, shape: List[Int]) raises -> Tensor:
        """Broadcasts to `shape`; an entry of -1 keeps that dimension."""
        var mine = self.shape()
        var sh = shape.copy()
        var lead = len(sh) - len(mine)
        for i in range(len(sh)):
            if sh[i] == -1:
                if i < lead:
                    raise Error("expand: -1 for a new leading dimension")
                sh[i] = mine[i - lead]
        return Tensor(self.uop().broadcast_to(sh))

    def expand(self, *shape: Int) raises -> Tensor:
        return self.expand(_list(shape))

    def pad(self, padding: List[Int]) raises -> Tensor:
        """Pads with zeros. `padding` holds `(before, after)` for each
        dimension in order, flattened."""
        return Tensor(self.uop().pad(padding))

    def shrink(self, bounds: List[Int]) raises -> Tensor:
        """Keeps `[lo, hi)` of each dimension. `bounds` holds `(lo, hi)`
        for each dimension in order, flattened."""
        return Tensor(self.uop().shrink(bounds))

    def slice(self, dim: Int, start: Int, end: Int) raises -> Tensor:
        """Keeps indices `[start, end)` of dimension `dim`; negative
        bounds count from the end."""
        var d = self._axis(dim)
        var shape = self.shape()
        var lo = start + shape[d] if start < 0 else start
        var hi = end + shape[d] if end < 0 else end
        hi = min(hi, shape[d])
        var arg = List[Int]()
        for i in range(len(shape)):
            arg.append(lo if i == d else 0)
            arg.append(hi if i == d else shape[i])
        return Tensor(self.uop().shrink(arg))

    def select(self, dim: Int, index: Int) raises -> Tensor:
        """Index `index` of dimension `dim`, dropping the dimension."""
        var d = self._axis(dim)
        var shape = self.shape()
        var i = index + shape[d] if index < 0 else index
        if i < 0 or i >= shape[d]:
            raise Error(t"index {index} out of range for size {shape[d]}")
        var s = self.slice(d, i, i + 1)
        _ = shape.pop(d)
        return s.reshape(shape)

    def __getitem__(self, *index: Int) raises -> Tensor:
        """Indexes the leading dimensions, dropping them."""
        var out = self
        for i in index:
            out = out.select(0, i)
        return out

    def flip(self, *axes: Int) raises -> Tensor:
        """Reverses the given dimensions."""
        var flags = List[Int](length=self.ndim(), fill=0)
        for a in axes:
            flags[self._axis(a)] = 1
        return Tensor(self.uop().flip(flags))

    def contiguous(self) -> Tensor:
        """Marks the tensor to be materialized in its own memory."""
        return Tensor(self.uop().contiguous())

    def repeat(self, repeats: List[Int]) raises -> Tensor:
        """Tiles the tensor `repeats[d]` times along each dimension."""
        var shape = self.shape()
        var lead = len(repeats) - len(shape)
        if lead < 0:
            raise Error("repeat needs a count for every dimension")
        var base = List[Int](length=lead, fill=1)
        for s in shape:
            base.append(s)
        var unsq = List[Int]()
        var exp = List[Int]()
        var final = List[Int]()
        for d in range(len(base)):
            unsq.append(1)
            unsq.append(base[d])
            exp.append(repeats[d])
            exp.append(base[d])
            final.append(repeats[d] * base[d])
        return self.reshape(unsq).expand(exp).reshape(final)

    @staticmethod
    def cat(tensors: List[Tensor], dim: Int = 0) raises -> Tensor:
        """Joins tensors along dimension `dim`."""
        if len(tensors) == 0:
            raise Error("cat of no tensors")
        var d = tensors[0]._axis(dim)
        var total = 0
        for t in tensors:
            total += t.shape()[d]
        # Pad each to the full size at its position; the sum is the join.
        var out = tensors[0]
        var before = 0
        for i in range(len(tensors)):
            var shape = tensors[i].shape()
            var arg = List[Int](length=2 * len(shape), fill=0)
            arg[2 * d] = before
            arg[2 * d + 1] = total - before - shape[d]
            var padded = tensors[i].pad(arg)
            out = padded if i == 0 else out + padded
            before += shape[d]
        return out

    @staticmethod
    def stack(tensors: List[Tensor], dim: Int = 0) raises -> Tensor:
        """Joins tensors along a new dimension `dim`."""
        var parts = List[Tensor]()
        for t in tensors:
            parts.append(t.unsqueeze(dim))
        return Tensor.cat(parts, dim)

    # ---- reduce

    def _reduce(
        self, rop: Int, axes: List[Int], keepdim: Bool
    ) raises -> Tensor:
        """Reduces over `axes` (all dimensions when empty)."""
        var n = self.ndim()
        var flags = List[Bool](length=n, fill=False)
        if len(axes) == 0:
            for d in range(n):
                flags[d] = True
        for a in axes:
            flags[self._axis(a)] = True
        var ax = List[Int]()
        var out_shape = List[Int]()
        var shape = self.shape()
        for d in range(n):
            if flags[d]:
                ax.append(d)
            else:
                out_shape.append(shape[d])
        var r = self.uop().reduce(rop, ax)
        if not keepdim:
            r = r.reshape(out_shape)
        return Tensor(r)

    def sum(self, *axis: Int, keepdim: Bool = False) raises -> Tensor:
        """The sum over the given dimensions (all when none are given).
        Narrow integers and booleans are summed as 32-bit integers, and
        16-bit floats in `float32` and converted back, as in tinygrad."""
        var x = self.cast(sum_acc_dtype(self.dtype()))
        var s = x._reduce(REDUCE_SUM, _list(axis), keepdim)
        if self.dtype().is_half():
            return s.cast(self.dtype())
        return s

    def max(self, *axis: Int, keepdim: Bool = False) raises -> Tensor:
        """The maximum over the given dimensions (all when none)."""
        return self._reduce(REDUCE_MAX, _list(axis), keepdim)

    def min(self, *axis: Int, keepdim: Bool = False) raises -> Tensor:
        """The minimum over the given dimensions (all when none)."""
        return self.neg()._reduce(REDUCE_MAX, _list(axis), keepdim).neg()

    def _count(self, axes: List[Int]) raises -> Int:
        """The number of elements reduced over `axes`."""
        if len(axes) == 0:
            return self.numel()
        var shape = self.shape()
        var flags = List[Bool](length=len(shape), fill=False)
        for a in axes:
            flags[self._axis(a)] = True
        var c = 1
        for d in range(len(shape)):
            if flags[d]:
                c *= shape[d]
        return c

    def mean(self, *axis: Int, keepdim: Bool = False) raises -> Tensor:
        """The mean over the given dimensions (all when none)."""
        var axes = _list(axis)
        var x = self
        if not x.dtype().is_float():
            x = x.cast(DT.default_float)
        var s = x._reduce(REDUCE_SUM, axes, keepdim)
        return s * (1.0 / Float64(self._count(axes)))

    def `var`(
        self, *axis: Int, keepdim: Bool = False, correction: Int = 1
    ) raises -> Tensor:
        """The variance over the given dimensions, dividing by the count
        minus `correction`."""
        var axes = _list(axis)
        var x = self
        if not x.dtype().is_float():
            x = x.cast(DT.default_float)
        var m = x._reduce(REDUCE_SUM, axes, True) * (
            1.0 / Float64(self._count(axes))
        )
        var d = x - m
        var n = max(1, self._count(axes) - correction)
        return (d * d)._reduce(REDUCE_SUM, axes, keepdim) * (1.0 / Float64(n))

    def std(
        self, *axis: Int, keepdim: Bool = False, correction: Int = 1
    ) raises -> Tensor:
        """The standard deviation over the given dimensions."""
        var axes = _list(axis)
        var x = self
        if not x.dtype().is_float():
            x = x.cast(DT.default_float)
        var m = x._reduce(REDUCE_SUM, axes, True) * (
            1.0 / Float64(self._count(axes))
        )
        var d = x - m
        var n = max(1, self._count(axes) - correction)
        return (
            (d * d)._reduce(REDUCE_SUM, axes, keepdim) * (1.0 / Float64(n))
        ).sqrt()

    def argmax(self, axis: Int = -1) raises -> Tensor:
        """The index of the first maximum along `axis`, as `int32`."""
        var a = self._axis(axis)
        var n = self.shape()[a]
        var m = self.eq(self._reduce(REDUCE_MAX, [a], True))
        # Weight position i by n - i: the largest weight is the first hit.
        var shape = List[Int](length=self.ndim(), fill=1)
        shape[a] = n
        var w = Tensor.arange(n, 0, -1).reshape(shape)
        var hit = (m.cast(DT.int32) * w)._reduce(REDUCE_MAX, [a], False)
        return n - hit

    def argmin(self, axis: Int = -1) raises -> Tensor:
        """The index of the first minimum along `axis`, as `int32`."""
        return self.neg().argmax(axis)

    def logsumexp(self, axis: Int = -1, keepdim: Bool = False) raises -> Tensor:
        """`log(sum(exp(x)))` along `axis`, computed stably."""
        var m = self._reduce(REDUCE_MAX, [axis], True).detach()
        var s = (self - m).exp()._reduce(REDUCE_SUM, [axis], True).log() + m
        if keepdim:
            return s
        return s.squeeze(axis)

    def softmax(self, axis: Int = -1) raises -> Tensor:
        """Exponentials normalized to sum to 1 along `axis`."""
        var m = self - self._reduce(REDUCE_MAX, [axis], True).detach()
        var e = m.exp()
        return e / e._reduce(REDUCE_SUM, [axis], True)

    def log_softmax(self, axis: Int = -1) raises -> Tensor:
        """The logarithm of `softmax`, computed stably."""
        var m = self - self._reduce(REDUCE_MAX, [axis], True).detach()
        return m - m.exp()._reduce(REDUCE_SUM, [axis], True).log()

    # ---- linear algebra

    def dot(self, w: Tensor) raises -> Tensor:
        """The matrix product over the last dimension of this tensor and
        the second-to-last of `w` (the last, if `w` is a vector), with
        broadcasting over the others."""
        var n1 = self.ndim()
        var n2 = w.ndim()
        if n1 == 0 or n2 == 0:
            raise Error("dot needs tensors of at least 1 dimension")
        var sx = self.shape()
        var sw = w.shape()
        var k = min(n2, 2)
        if sx[n1 - 1] != sw[n2 - k]:
            raise Error(t"cannot multiply shapes {sx} and {sw}")
        var ones = min(min(n1 - 1, n2 - 1), 1)
        var xs = List[Int]()
        for i in range(n1 - 1):
            xs.append(sx[i])
        for _ in range(ones):
            xs.append(1)
        xs.append(sx[n1 - 1])
        var ws = List[Int]()
        for i in range(n2 - 2):
            ws.append(sw[i])
        for _ in range(ones):
            ws.append(1)
        for i in range(n2 - k, n2):
            ws.append(sw[i])
        var x = self.reshape(xs)
        var y = w.reshape(ws)
        y = y.transpose(-1, -k)
        return (x * y).sum(-1)

    def matmul(self, w: Tensor) raises -> Tensor:
        """The matrix product; see `dot`."""
        return self.dot(w)

    def linear(
        self, weight: Tensor, bias: Optional[Tensor] = None
    ) raises -> Tensor:
        """`self @ weight + bias`, with `weight` of shape (in, out)."""
        var y = self.dot(weight)
        if bias:
            return y + bias.value()
        return y

    # ---- convolution and pooling

    def _pool(
        self, kernel: List[Int], stride: List[Int], dilation: List[Int]
    ) raises -> Tensor:
        """Sliding windows over the last `len(kernel)` dimensions: the
        leading dimensions, then the window positions, then the offsets
        within a window. Trailing input that does not fill a window is
        dropped."""
        var nk = len(kernel)
        var shape = self.shape()
        var p = len(shape) - nk
        var tiles = True
        for j in range(nk):
            if kernel[j] > stride[j] or dilation[j] != 1:
                tiles = False
        if not tiles:
            return Tensor(self.uop().window(kernel, stride, dilation))
        # Windows that do not overlap are a reshape: (o s) -> (o, s),
        # keep the first k of each s, and move the offsets last.
        var crop = List[Int]()
        var split = List[Int]()
        var keep = List[Int]()
        var axes = List[Int]()
        for i in range(p):
            crop.append(0)
            crop.append(shape[i])
            split.append(shape[i])
            keep.append(0)
            keep.append(shape[i])
            axes.append(i)
        for j in range(nk):
            if kernel[j] > shape[p + j]:
                raise Error(
                    t"pool window {kernel} is larger than the input {shape}"
                )
            var o = (shape[p + j] - kernel[j]) // stride[j] + 1
            # The last window may end past o s - (s - k): pad to o s.
            crop.append(0)
            crop.append(min(shape[p + j], o * stride[j]))
            split.append(o)
            split.append(stride[j])
            keep.append(0)
            keep.append(o)
            keep.append(0)
            keep.append(kernel[j])
            axes.append(p + 2 * j)
        for j in range(nk):
            axes.append(p + 2 * j + 1)
        var x = self.shrink(crop)
        var pad = List[Int](length=2 * len(shape), fill=0)
        var padded = False
        for j in range(nk):
            var o = (shape[p + j] - kernel[j]) // stride[j] + 1
            var short = o * stride[j] - x.shape()[p + j]
            if short > 0:
                pad[2 * (p + j) + 1] = short
                padded = True
        if padded:
            x = x.pad(pad)
        return x.reshape(split).shrink(keep).permute(axes)

    def conv2d(
        self,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        groups: Int = 1,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Int = 0,
    ) raises -> Tensor:
        """Convolves this (batch, channels, spatial...) tensor with
        `weight` of shape (out channels, channels / groups, kernel...).

        Works for any number of spatial dimensions. As in tinygrad, the
        convolution is windows of the input times the broadcast weight,
        summed; the scheduler runs that as a matrix product.

        Args:
            weight: The filters.
            bias: One value per output channel, or `None`.
            groups: The number of groups the channels are split into.
            stride: The step between windows.
            dilation: The step between kernel elements.
            padding: Zeros added on every side of each spatial dimension.

        Returns:
            A tensor of shape (batch, out channels, out spatial...).
        """
        var xs = self.shape()
        var ws = weight.shape()
        var nd = len(ws) - 2
        if len(xs) != len(ws) or nd < 1:
            raise Error(t"conv of input {xs} with weight {ws}")
        var bs = xs[0]
        var cout = ws[0]
        var cin = ws[1]
        if groups * cin != xs[1] or cout % groups != 0:
            raise Error(
                t"conv of input {xs} with weight {ws} and {groups} groups"
            )
        var kernel = List[Int]()
        var strides = List[Int]()
        var dilations = List[Int]()
        var pad = List[Int](length=4, fill=0)
        for j in range(nd):
            kernel.append(ws[2 + j])
            strides.append(stride)
            dilations.append(dilation)
            pad.append(padding)
            pad.append(padding)
        # (bs, groups * cin, o..., k...)
        var x = self.pad(pad)._pool(kernel, strides, dilations)
        var rcout = cout // groups
        var o = List[Int]()
        var xsh = x.shape()
        for j in range(nd):
            o.append(xsh[2 + j])
        # x: (bs, groups, cin, 1, o..., k...) expanded over rcout, then
        # permuted to (bs, groups, rcout, o..., cin, k...).
        var r1: List[Int] = [bs, groups, cin, 1]
        var e1: List[Int] = [bs, groups, cin, rcout]
        var perm: List[Int] = [0, 1, 3]
        for j in range(nd):
            r1.append(o[j])
            e1.append(o[j])
            perm.append(4 + j)
        perm.append(2)
        for j in range(nd):
            r1.append(kernel[j])
            e1.append(kernel[j])
            perm.append(4 + nd + j)
        x = x.reshape(r1).expand(e1).permute(perm)
        # weight: (1, groups, rcout, 1..., cin, k...)
        var wsh: List[Int] = [1, groups, rcout]
        for _ in range(nd):
            wsh.append(1)
        wsh.append(cin)
        for j in range(nd):
            wsh.append(kernel[j])
        var prod = x * weight.reshape(wsh)
        var axes = List[Int]()
        for j in range(1 + nd):
            axes.append(3 + nd + j)
        var out_shape: List[Int] = [bs, cout]
        for j in range(nd):
            out_shape.append(o[j])
        var ret = Tensor(
            prod._reduce(REDUCE_SUM, axes, True)
            .reshape(out_shape)
            .uop()
            .as_conv(
                self.uop(), weight.uop(), [groups, stride, dilation, padding]
            )
        )
        if bias:
            var bsh: List[Int] = [1, cout]
            for _ in range(nd):
                bsh.append(1)
            return ret + bias.value().reshape(bsh)
        return ret

    def max_pool2d(self, kernel: Int = 2, stride: Int = -1) raises -> Tensor:
        """The maximum over `kernel` by `kernel` windows of the last two
        dimensions, `stride` apart (default: the kernel size)."""
        var s = kernel if stride < 0 else stride
        var k: List[Int] = [kernel, kernel]
        var st: List[Int] = [s, s]
        var n = self.ndim()
        return self._pool(k, st, [1, 1])._reduce(REDUCE_MAX, [n, n + 1], False)

    def avg_pool2d(self, kernel: Int = 2, stride: Int = -1) raises -> Tensor:
        """The mean over `kernel` by `kernel` windows of the last two
        dimensions, `stride` apart (default: the kernel size)."""
        var s = kernel if stride < 0 else stride
        var k: List[Int] = [kernel, kernel]
        var st: List[Int] = [s, s]
        var n = self.ndim()
        return self._pool(k, st, [1, 1])._reduce(
            REDUCE_SUM, [n, n + 1], False
        ) * (1.0 / Float64(kernel * kernel))

    def batchnorm(
        self, weight: Tensor, bias: Tensor, mean: Tensor, invstd: Tensor
    ) raises -> Tensor:
        """`(self - mean) * invstd * weight + bias`, with the four
        per-channel vectors broadcast along dimension 1."""
        var shape = List[Int](length=self.ndim(), fill=1)
        shape[1] = self.shape()[1]
        var x = self - mean.reshape(shape)
        return x * (weight * invstd).reshape(shape) + bias.reshape(shape)

    # ---- attention

    def scaled_dot_product_attention(
        self,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: Bool = False,
    ) raises -> Tensor:
        """`softmax(self @ key.T / sqrt(d) + mask) @ value`, over the last
        two dimensions.

        Args:
            key: Keys, `(..., L, d)`.
            value: Values, `(..., L, dv)`.
            attn_mask: A `bool` tensor (true where a query may attend to
                a key) or a float tensor added to the scores; broadcast
                to `(..., T, L)`.
            is_causal: Whether query `i` attends only to keys up to
                `i`, aligned at the end (`L - T + i`), as in a decoder
                with a cache.

        Returns:
            `(..., T, dv)`.
        """
        var d = self.shape()[self.ndim() - 1]
        var scores = self.dot(key.transpose(-1, -2)) * (1.0 / sqrt(Float64(d)))
        if is_causal:
            var t = self.shape()[self.ndim() - 2]
            var l = key.shape()[key.ndim() - 2]
            var allowed = (
                Tensor.arange(l)
                .reshape(1, l)
                .le((Tensor.arange(t) + (l - t)).reshape(t, 1))
            )
            scores = allowed.where(scores, -inf[DType.float64]())
        if attn_mask:
            var m = attn_mask.value()
            if m.dtype().is_bool():
                scores = m.where(scores, -inf[DType.float64]())
            else:
                scores = scores + m
        return scores.softmax(-1).dot(value)

    def tril(self, diagonal: Int = 0) raises -> Tensor:
        """Zeros above diagonal `diagonal` of the last two dimensions."""
        var r = self.shape()[self.ndim() - 2]
        var c = self.shape()[self.ndim() - 1]
        var keep = (
            Tensor.arange(c)
            .reshape(1, c)
            .le((Tensor.arange(r) + diagonal).reshape(r, 1))
        )
        return keep.where(self, self._scalar(0.0, True))

    def triu(self, diagonal: Int = 0) raises -> Tensor:
        """Zeros below diagonal `diagonal` of the last two dimensions."""
        var r = self.shape()[self.ndim() - 2]
        var c = self.shape()[self.ndim() - 1]
        var keep = (
            Tensor.arange(c)
            .reshape(1, c)
            .ge((Tensor.arange(r) + diagonal).reshape(r, 1))
        )
        return keep.where(self, self._scalar(0.0, True))

    # ---- functional

    def one_hot(self, num_classes: Int) raises -> Tensor:
        """A new last dimension of size `num_classes` holding 1 at each
        element's index and 0 elsewhere, as `bool`."""
        return self.unsqueeze(-1).eq(Tensor.arange(num_classes))

    def sparse_categorical_crossentropy(self, target: Tensor) raises -> Tensor:
        """The mean cross-entropy between these logits (classes along the
        last dimension) and integer class labels `target`."""
        var classes = self.shape()[self.ndim() - 1]
        var logp = self.log_softmax(-1)
        var y = target.one_hot(classes)
        var picked = y.where(logp, 0.0)
        return picked.sum().neg() * (1.0 / Float64(target.numel()))

    def dropout(self, p: Float64 = 0.5) raises -> Tensor:
        """In training mode, zeros each element with probability `p` and
        scales the rest by `1 / (1 - p)`; otherwise the identity."""
        if not graph()[].training or p == 0.0:
            return self
        var keep = Tensor.uniform(self.shape(), 0.0, 1.0).ge(p)
        return keep.where(self, 0.0) * (1.0 / (1.0 - p))

    def layernorm(self, axis: Int = -1, eps: Float64 = 1e-5) raises -> Tensor:
        """Normalizes to zero mean and unit variance along `axis`."""
        var y = self - self.mean(axis, keepdim=True)
        return y * ((y * y).mean(axis, keepdim=True) + eps).rsqrt()

    @staticmethod
    def train(mode: Bool = True):
        """Sets training mode, which `dropout` and batch normalization
        consult."""
        graph()[].training = mode

    @staticmethod
    def training() -> Bool:
        """Whether training mode is on."""
        return graph()[].training


def _list(xs: VariadicList[Int, False]) -> List[Int]:
    """A variadic argument list as a `List`."""
    var out = List[Int](capacity=len(xs))
    for x in xs:
        out.append(x)
    return out^


def _format(
    mut out: String,
    data: List[Float64],
    shape: List[Int],
    dim: Int,
    offset: Int,
    is_float: Bool,
):
    """Appends dimension `dim` of `data` at `offset` as nested brackets."""
    if dim == len(shape):
        if is_float:
            out += String(data[offset])
        else:
            out += String(Int(data[offset]))
        return
    var stride = 1
    for d in range(dim + 1, len(shape)):
        stride *= shape[d]
    out += "["
    for i in range(shape[dim]):
        if i > 0:
            out += ", "
        _format(out, data, shape, dim + 1, offset + i * stride, is_float)
    out += "]"
