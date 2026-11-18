"""The Python extension module `minigrad._ext`: the engine's entry points
for the Python package around it (`python/minigrad/__init__.py`).

Each function takes and returns plain Python values and `Handle`s, a
Python object holding one `Tensor`. The Python package does the API work
(argument forms, broadcasting scalars, operators); this module only
converts and dispatches. Operations of one signature share an entry
point and are picked by name, so a new operation is one line here.

Build: `pixi run python-build` (writes `python/minigrad/_ext.so`).
"""

from std.memory import unsafe_memcpy
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from minigrad.dtype import DT
from minigrad.graph import BytePtr, graph
from minigrad.jit import MiniJit
from minigrad.tensor import Tensor


struct Handle(Movable, Writable):
    """A tensor as a Python object."""

    var t: Tensor

    def __init__(out self, t: Tensor):
        self.t = t

    def write_to(self, mut writer: Some[Writer]):
        self.t.write_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        self.t.write_to(writer)


struct Jit(Movable, Writable):
    """A `MiniJit` as a Python object."""

    var jit: MiniJit

    def __init__(out self):
        self.jit = MiniJit()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("<MiniJit>")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<MiniJit>")


# ---- conversions


def _wrap(t: Tensor) raises -> PythonObject:
    return PythonObject(alloc=Handle(t))


def _t(o: PythonObject) raises -> Tensor:
    return o.downcast_value_ptr[Handle]()[].t


def _maybe(o: PythonObject) raises -> Optional[Tensor]:
    if o is Python.none():
        return None
    return _t(o)


def _ints(o: PythonObject) raises -> List[Int]:
    var n = Int(py=o.__len__())
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(Int(py=o[i]))
    return out^


def _tensors(o: PythonObject) raises -> List[Tensor]:
    var n = Int(py=o.__len__())
    var out = List[Tensor](capacity=n)
    for i in range(n):
        out.append(_t(o[i]))
    return out^


def _wrap_all(ts: List[Tensor]) raises -> PythonObject:
    var out = Python.list()
    for t in ts:
        _ = out.append(_wrap(t))
    return out


def _f(o: PythonObject) raises -> Float64:
    return Float64(py=o)


def _dt(o: PythonObject) raises -> DT:
    return DT(Int(py=o))


# ---- creation


def from_memory(
    address: PythonObject, shape: PythonObject, dtype: PythonObject
) raises -> PythonObject:
    """A new realized tensor holding a copy of row-major data at
    `address` (from NumPy or `array.array`)."""
    var t = Tensor.empty(_ints(shape), _dt(dtype))
    var n = t.numel() * t.dtype().itemsize()
    if n > 0:
        unsafe_memcpy(
            dest=t.data_ptr(),
            src=BytePtr(unsafe_from_address=Int(py=address)),
            count=n,
        )
    return _wrap(t)


def full(
    shape: PythonObject, value: PythonObject, dtype: PythonObject
) raises -> PythonObject:
    return _wrap(Tensor.full(_ints(shape), _f(value), _dt(dtype)))


def arange(
    start: PythonObject, stop: PythonObject, step: PythonObject
) raises -> PythonObject:
    return _wrap(Tensor.arange(Int(py=start), Int(py=stop), Int(py=step)))


def eye(n: PythonObject, dtype: PythonObject) raises -> PythonObject:
    return _wrap(Tensor.eye(Int(py=n), _dt(dtype)))


def manual_seed(seed: PythonObject) raises:
    Tensor.manual_seed(Int(py=seed))


def random(
    name: PythonObject, shape: PythonObject, a: PythonObject, b: PythonObject
) raises -> PythonObject:
    """`rand`, `randn`, `uniform(a, b)`, `randint(a, b)`,
    `kaiming_uniform(a)`, `glorot_uniform`."""
    var n = String(py=name)
    var s = _ints(shape)
    if n == "rand":
        return _wrap(Tensor.uniform(s))
    if n == "uniform":
        return _wrap(Tensor.uniform(s, _f(a), _f(b)))
    if n == "randn":
        return _wrap(Tensor.randn(s))
    if n == "randint":
        return _wrap(Tensor.randint(s, Int(py=a), Int(py=b)))
    if n == "kaiming_uniform":
        return _wrap(Tensor.kaiming_uniform(s, _f(a)))
    if n == "glorot_uniform":
        return _wrap(Tensor.glorot_uniform(s))
    raise Error(t"unknown random distribution '{n}'")


# ---- properties and data


def shape(h: PythonObject) raises -> PythonObject:
    """The shape, as a list."""
    var out = Python.list()
    for d in _t(h).shape():
        _ = out.append(PythonObject(d))
    return out


def dtype(h: PythonObject) raises -> PythonObject:
    return PythonObject(_t(h).dtype().code)


def is_realized(h: PythonObject) raises -> PythonObject:
    return PythonObject(_t(h).uop().is_realized())


def memory(h: PythonObject) raises -> PythonObject:
    """`(handle, address)`: the tensor's elements in row-major order, in
    a buffer that stays alive (and unchanged) while `handle` does."""
    var u = _t(h)._contiguous_uop()
    var addr = Int(graph()[].buffer_ptr(u.id))
    return Python.tuple(_wrap(Tensor(u)), PythonObject(addr))


def item(h: PythonObject) raises -> PythonObject:
    var t = _t(h)
    var v = t.item()
    if t.dtype().is_bool():
        return PythonObject(v != 0.0)
    if not t.dtype().is_float():
        return PythonObject(Int(v))
    return PythonObject(v)


def realize(hs: PythonObject) raises:
    Tensor.realize_all(_tensors(hs))


# ---- elementwise


def unary(name: PythonObject, h: PythonObject) raises -> PythonObject:
    var n = String(py=name)
    var x = _t(h)
    var r: Tensor
    if n == "neg":
        r = x.neg()
    elif n == "relu":
        r = x.relu()
    elif n == "exp":
        r = x.exp()
    elif n == "log":
        r = x.log()
    elif n == "sqrt":
        r = x.sqrt()
    elif n == "rsqrt":
        r = x.rsqrt()
    elif n == "reciprocal":
        r = x.reciprocal()
    elif n == "sigmoid":
        r = x.sigmoid()
    elif n == "tanh":
        r = x.tanh()
    elif n == "gelu":
        r = x.gelu()
    elif n == "silu":
        r = x.silu()
    elif n == "sin":
        r = x.sin()
    elif n == "cos":
        r = x.cos()
    elif n == "tan":
        r = x.tan()
    elif n == "trunc":
        r = x.trunc()
    elif n == "floor":
        r = x.floor()
    elif n == "ceil":
        r = x.ceil()
    elif n == "round":
        r = x.round()
    elif n == "square":
        r = x.square()
    elif n == "abs":
        r = x.abs()
    elif n == "sign":
        r = x.sign()
    elif n == "logical_not":
        r = x.logical_not()
    elif n == "contiguous":
        r = x.contiguous()
    elif n == "detach":
        r = x.detach()
    else:
        raise Error(t"unknown unary operation '{n}'")
    return _wrap(r)


def binary(
    name: PythonObject, a: PythonObject, b: PythonObject
) raises -> PythonObject:
    var n = String(py=name)
    var x = _t(a)
    var y = _t(b)
    var r: Tensor
    if n == "add":
        r = x.add(y)
    elif n == "sub":
        r = x.sub(y)
    elif n == "mul":
        r = x.mul(y)
    elif n == "div":
        r = x.div(y)
    elif n == "idiv":
        r = x.idiv(y)
    elif n == "mod":
        r = x.mod(y)
    elif n == "pow":
        r = x.pow(y)
    elif n == "maximum":
        r = x.maximum(y)
    elif n == "minimum":
        r = x.minimum(y)
    elif n == "lt":
        r = x.lt(y)
    elif n == "gt":
        r = x.gt(y)
    elif n == "le":
        r = x.le(y)
    elif n == "ge":
        r = x.ge(y)
    elif n == "eq":
        r = x.eq(y)
    elif n == "ne":
        r = x.ne(y)
    elif n == "logical_and":
        r = x.logical_and(y)
    elif n == "logical_or":
        r = x.logical_or(y)
    elif n == "dot":
        r = x.dot(y)
    elif n == "sparse_categorical_crossentropy":
        r = x.sparse_categorical_crossentropy(y)
    else:
        raise Error(t"unknown binary operation '{n}'")
    return _wrap(r)


def scalar(
    h: PythonObject, value: PythonObject, is_int: PythonObject
) raises -> PythonObject:
    """`value` as a tensor to combine with `h`: of `h`'s type unless that
    loses the value's kind (a float with an integer tensor)."""
    return _wrap(_t(h)._scalar(_f(value), Bool(py=is_int)))


def scalar_pow(h: PythonObject, e: PythonObject) raises -> PythonObject:
    """`h ** e` for a number `e` (exact for small integer exponents)."""
    return _wrap(_t(h).pow(_f(e)))


def with_number(
    name: PythonObject, h: PythonObject, a: PythonObject, b: PythonObject
) raises -> PythonObject:
    """Operations with number arguments: `leaky_relu(a)`, `softplus(a)`,
    `clip(a, b)`, `dropout(a)`, `layernorm(axis a, eps b)`,
    `one_hot(a)`, `tril(a)`, `triu(a)`."""
    var n = String(py=name)
    var x = _t(h)
    if n == "leaky_relu":
        return _wrap(x.leaky_relu(_f(a)))
    if n == "softplus":
        return _wrap(x.softplus(_f(a)))
    if n == "clip":
        return _wrap(x.clip(_f(a), _f(b)))
    if n == "dropout":
        return _wrap(x.dropout(_f(a)))
    if n == "layernorm":
        return _wrap(x.layernorm(Int(py=a), _f(b)))
    if n == "one_hot":
        return _wrap(x.one_hot(Int(py=a)))
    if n == "tril":
        return _wrap(x.tril(Int(py=a)))
    if n == "triu":
        return _wrap(x.triu(Int(py=a)))
    raise Error(t"unknown operation '{n}'")


def where_(
    c: PythonObject, a: PythonObject, b: PythonObject
) raises -> PythonObject:
    return _wrap(_t(c).where(_t(a), _t(b)))


def cast(h: PythonObject, dtype: PythonObject) raises -> PythonObject:
    return _wrap(_t(h).cast(_dt(dtype)))


# ---- reductions


def reduce(
    name: PythonObject,
    h: PythonObject,
    axes: PythonObject,
    keepdim: PythonObject,
    correction: PythonObject,
) raises -> PythonObject:
    """`sum`, `max`, `min`, `mean`, `var`, `std` over `axes` (all when
    empty)."""
    var n = String(py=name)
    var x = _t(h)
    var ax = _ints(axes)
    var k = Bool(py=keepdim)
    if n == "sum":
        return _wrap(x.sum(ax, k))
    if n == "max":
        return _wrap(x.max(ax, k))
    if n == "min":
        return _wrap(x.min(ax, k))
    if n == "mean":
        return _wrap(x.mean(ax, k))
    if n == "var":
        return _wrap(x.`var`(ax, k, Int(py=correction)))
    if n == "std":
        return _wrap(x.std(ax, k, Int(py=correction)))
    raise Error(t"unknown reduction '{n}'")


def along(
    name: PythonObject,
    h: PythonObject,
    axis: PythonObject,
    keepdim: PythonObject,
) raises -> PythonObject:
    """Operations along one axis: `softmax`, `log_softmax`, `argmax`,
    `argmin`, `logsumexp`."""
    var n = String(py=name)
    var x = _t(h)
    var a = Int(py=axis)
    if n == "softmax":
        return _wrap(x.softmax(a))
    if n == "log_softmax":
        return _wrap(x.log_softmax(a))
    if n == "argmax":
        return _wrap(x.argmax(a))
    if n == "argmin":
        return _wrap(x.argmin(a))
    if n == "logsumexp":
        return _wrap(x.logsumexp(a, Bool(py=keepdim)))
    raise Error(t"unknown operation '{n}'")


# ---- movement


def movement(
    name: PythonObject, h: PythonObject, arg: PythonObject
) raises -> PythonObject:
    """`reshape`, `permute`, `expand`, `pad` and `shrink` (flattened
    pairs), `flip` (axes), `repeat`, each with a list of integers."""
    var n = String(py=name)
    var x = _t(h)
    var a = _ints(arg)
    if n == "reshape":
        return _wrap(x.reshape(a))
    if n == "permute":
        return _wrap(x.permute(a))
    if n == "expand":
        return _wrap(x.expand(a))
    if n == "pad":
        return _wrap(x.pad(a))
    if n == "shrink":
        return _wrap(x.shrink(a))
    if n == "repeat":
        return _wrap(x.repeat(a))
    if n == "flip":
        var r = x
        for d in a:
            r = r.flip(d)
        return _wrap(r)
    raise Error(t"unknown movement '{n}'")


def join(
    name: PythonObject, hs: PythonObject, dim: PythonObject
) raises -> PythonObject:
    """`cat` or `stack` of a list of tensors."""
    var n = String(py=name)
    if n == "cat":
        return _wrap(Tensor.cat(_tensors(hs), Int(py=dim)))
    if n == "stack":
        return _wrap(Tensor.stack(_tensors(hs), Int(py=dim)))
    raise Error(t"unknown join '{n}'")


# ---- layers


def conv2d(
    x: PythonObject,
    w: PythonObject,
    b: PythonObject,
    groups: PythonObject,
    stride: PythonObject,
    dilation: PythonObject,
    padding: PythonObject,
) raises -> PythonObject:
    return _wrap(
        _t(x).conv2d(
            _t(w),
            _maybe(b),
            Int(py=groups),
            Int(py=stride),
            Int(py=dilation),
            Int(py=padding),
        )
    )


def pool(
    name: PythonObject,
    h: PythonObject,
    kernel: PythonObject,
    stride: PythonObject,
) raises -> PythonObject:
    """`max_pool2d` or `avg_pool2d`."""
    var n = String(py=name)
    if n == "max_pool2d":
        return _wrap(_t(h).max_pool2d(Int(py=kernel), Int(py=stride)))
    if n == "avg_pool2d":
        return _wrap(_t(h).avg_pool2d(Int(py=kernel), Int(py=stride)))
    raise Error(t"unknown pooling '{n}'")


def linear(
    x: PythonObject, w: PythonObject, b: PythonObject
) raises -> PythonObject:
    return _wrap(_t(x).linear(_t(w), _maybe(b)))


def batchnorm(
    x: PythonObject,
    w: PythonObject,
    b: PythonObject,
    mean: PythonObject,
    invstd: PythonObject,
) raises -> PythonObject:
    return _wrap(_t(x).batchnorm(_t(w), _t(b), _t(mean), _t(invstd)))


def attention(
    q: PythonObject,
    k: PythonObject,
    v: PythonObject,
    mask: PythonObject,
    is_causal: PythonObject,
) raises -> PythonObject:
    return _wrap(
        _t(q).scaled_dot_product_attention(
            _t(k), _t(v), _maybe(mask), Bool(py=is_causal)
        )
    )


def train(mode: PythonObject) raises:
    Tensor.train(Bool(py=mode))


def training() raises -> PythonObject:
    return PythonObject(Tensor.training())


# ---- gradients and state


def requires_grad_(h: PythonObject, value: PythonObject) raises:
    _ = _t(h).requires_grad_(Bool(py=value))


def requires_grad(h: PythonObject) raises -> PythonObject:
    """True, False, or None when unset (an optimizer then turns it
    on)."""
    var r = _t(h).cell[].requires_grad
    if r < 0:
        return Python.none()
    return PythonObject(r == 1)


def backward(h: PythonObject) raises:
    _t(h).backward()


def grad(h: PythonObject) raises -> PythonObject:
    var t = _t(h)
    if not t.has_grad():
        return Python.none()
    return _wrap(t.grad())


def zero_grad(h: PythonObject) raises:
    _t(h).zero_grad()


def gradient(h: PythonObject, targets: PythonObject) raises -> PythonObject:
    return _wrap_all(_t(h).gradient(_tensors(targets)))


def assign(h: PythonObject, value: PythonObject) raises:
    _ = _t(h).assign(_t(value))


def same(a: PythonObject, b: PythonObject) raises -> PythonObject:
    """Whether two handles hold the same tensor (not just equal
    values)."""
    return PythonObject(_t(a).cell is _t(b).cell)


# ---- eager operations


def take_rows(h: PythonObject, rows: PythonObject) raises -> PythonObject:
    return _wrap(_t(h).take_rows(_t(rows)))


def write_(
    h: PythonObject, dim: PythonObject, start: PythonObject, value: PythonObject
) raises:
    _t(h).write_(Int(py=dim), _t(start), _t(value))


def put_rows_(h: PythonObject, rows: PythonObject, value: PythonObject) raises:
    _t(h).put_rows_(_t(rows), _t(value))


# ---- MiniJit


def jit_new() raises -> PythonObject:
    return PythonObject(alloc=Jit())


def jit_begin(j: PythonObject, inputs: PythonObject) raises:
    j.downcast_value_ptr[Jit]()[].jit.begin_capture(_tensors(inputs))


def jit_abort(j: PythonObject) raises:
    j.downcast_value_ptr[Jit]()[].jit.abort_capture()


def jit_end(j: PythonObject, outputs: PythonObject) raises -> PythonObject:
    return _wrap_all(
        j.downcast_value_ptr[Jit]()[].jit.end_capture(_tensors(outputs))
    )


def jit_replay(j: PythonObject, inputs: PythonObject) raises -> PythonObject:
    return _wrap_all(j.downcast_value_ptr[Jit]()[].jit.replay(_tensors(inputs)))


@export
def PyInit__ext() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_ext")
        _ = m.add_type[Handle]("Handle")
        _ = m.add_type[Jit]("Jit")
        m.def_function[from_memory]("from_memory")
        m.def_function[full]("full")
        m.def_function[arange]("arange")
        m.def_function[eye]("eye")
        m.def_function[manual_seed]("manual_seed")
        m.def_function[random]("random")
        m.def_function[shape]("shape")
        m.def_function[dtype]("dtype")
        m.def_function[is_realized]("is_realized")
        m.def_function[memory]("memory")
        m.def_function[item]("item")
        m.def_function[realize]("realize")
        m.def_function[unary]("unary")
        m.def_function[binary]("binary")
        m.def_function[scalar]("scalar")
        m.def_function[scalar_pow]("scalar_pow")
        m.def_function[with_number]("with_number")
        m.def_function[where_]("where")
        m.def_function[cast]("cast")
        m.def_function[reduce]("reduce")
        m.def_function[along]("along")
        m.def_function[movement]("movement")
        m.def_function[join]("join")
        m.def_function[conv2d]("conv2d")
        m.def_function[pool]("pool")
        m.def_function[linear]("linear")
        m.def_function[batchnorm]("batchnorm")
        m.def_function[attention]("attention")
        m.def_function[train]("train")
        m.def_function[training]("training")
        m.def_function[requires_grad_]("requires_grad_")
        m.def_function[requires_grad]("requires_grad")
        m.def_function[backward]("backward")
        m.def_function[grad]("grad")
        m.def_function[zero_grad]("zero_grad")
        m.def_function[gradient]("gradient")
        m.def_function[assign]("assign")
        m.def_function[same]("same")
        m.def_function[take_rows]("take_rows")
        m.def_function[write_]("write_")
        m.def_function[put_rows_]("put_rows_")
        m.def_function[jit_new]("jit_new")
        m.def_function[jit_begin]("jit_begin")
        m.def_function[jit_abort]("jit_abort")
        m.def_function[jit_end]("jit_end")
        m.def_function[jit_replay]("jit_replay")
        return m.finalize()
    except e:
        abort(String("creating minigrad._ext: ", e))
