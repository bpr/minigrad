"""`Tensor` for Python: tinygrad's API over the Mojo engine.

A `Tensor` holds a handle to an engine tensor (`_ext.Handle`). Copies of
a Python reference are the same tensor, as in tinygrad; `assign` changes
the value every reference sees. Everything stays lazy until `realize`,
`numpy`, `tolist` or `item`.
"""

from __future__ import annotations

import array
import contextlib
import ctypes
import math
from typing import Iterable, Sequence

from .dtype import DType, dtypes, to_dtype

try:
    from . import _ext
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "minigrad's engine is not built; run `pixi run python-build`"
    ) from e

try:
    import numpy as _np
except ImportError:  # NumPy is optional
    _np = None


def _shape_arg(shape) -> tuple:
    """`f(2, 3)` and `f((2, 3))` both mean shape (2, 3)."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(shape)


def _flatten(data, shape: list, depth: int, out: list):
    if isinstance(data, (list, tuple)):
        if len(shape) == depth:
            shape.append(len(data))
        elif shape[depth] != len(data):
            raise ValueError("ragged nested sequence")
        for x in data:
            _flatten(x, shape, depth + 1, out)
    else:
        if len(shape) != depth:
            raise ValueError("ragged nested sequence")
        out.append(data.item() if isinstance(data, Tensor) else data)


def _infer_dtype(values: list) -> DType:
    if all(isinstance(v, bool) for v in values):
        return dtypes.bool
    if all(isinstance(v, int) for v in values):
        return dtypes.default_int
    return dtypes.default_float


class _TensorMeta(type):
    @property
    def training(cls) -> bool:
        return _ext.training()

    @training.setter
    def training(cls, mode: bool):
        _ext.train(bool(mode))


class Tensor(metaclass=_TensorMeta):
    """A lazy n-dimensional array with automatic differentiation."""

    __slots__ = ("_h", "_shape", "_dtype", "__weakref__")

    # ---- construction

    def __init__(self, data, dtype=None, requires_grad: bool | None = None):
        """A tensor from a number, a nested sequence, a NumPy array, or
        an object with the buffer protocol. The data is copied once."""
        dt = to_dtype(dtype) if dtype is not None else None
        if isinstance(data, _ext.Handle):
            h = data
        elif isinstance(data, Tensor):
            h = data._h if dt is None else _ext.cast(data._h, dt.code)
        elif isinstance(data, (bool, int, float)):
            if dt is None:
                dt = _infer_dtype([data])
            h = _ext.full((), float(data), dt.code)
        elif _np is not None and isinstance(data, (_np.ndarray, _np.generic)):
            h = _from_numpy(_np.asarray(data), dt)
        elif isinstance(data, (list, tuple)):
            shape: list = []
            values: list = []
            _flatten(data, shape, 0, values)
            if dt is None:
                dt = _infer_dtype(values)
            h = _from_values(values, tuple(shape), dt)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            a = array.array("B")
            a.frombytes(data)
            h = _ext.from_memory(a.buffer_info()[0], (len(a),), dtypes.uint8.code)
            if dt is not None and dt != dtypes.uint8:
                h = _ext.cast(h, dt.code)
        else:
            raise TypeError(f"cannot make a Tensor from {type(data).__name__}")
        self._h = h
        self._shape = None
        self._dtype = None
        if requires_grad is not None:
            _ext.requires_grad_(h, bool(requires_grad))

    @staticmethod
    def _new(h) -> Tensor:
        t = object.__new__(Tensor)
        t._h = h
        t._shape = None
        t._dtype = None
        return t

    @staticmethod
    def _make(h, dtype, requires_grad) -> Tensor:
        t = Tensor._new(h)
        if dtype is not None and to_dtype(dtype) != t.dtype:
            t = t.cast(dtype)
        if requires_grad is not None:
            t.requires_grad_(requires_grad)
        return t

    @staticmethod
    def full(shape, fill_value, dtype=None, requires_grad=None) -> Tensor:
        """A tensor of `shape` filled with `fill_value`."""
        dt = to_dtype(dtype) if dtype is not None else _infer_dtype([fill_value])
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return Tensor._make(
            _ext.full(shape, float(fill_value), dt.code), None, requires_grad
        )

    @staticmethod
    def zeros(*shape, dtype=None, requires_grad=None) -> Tensor:
        return Tensor.full(
            _shape_arg(shape), 0.0, dtype or dtypes.default_float, requires_grad
        )

    @staticmethod
    def ones(*shape, dtype=None, requires_grad=None) -> Tensor:
        return Tensor.full(
            _shape_arg(shape), 1.0, dtype or dtypes.default_float, requires_grad
        )

    @staticmethod
    def empty(*shape, dtype=None) -> Tensor:
        return Tensor.zeros(*shape, dtype=dtype).contiguous()

    def full_like(self, fill_value, dtype=None, requires_grad=None) -> Tensor:
        return Tensor.full(self.shape, fill_value, dtype or self.dtype, requires_grad)

    def zeros_like(self, dtype=None, requires_grad=None) -> Tensor:
        return self.full_like(0, dtype, requires_grad)

    def ones_like(self, dtype=None, requires_grad=None) -> Tensor:
        return self.full_like(1, dtype, requires_grad)

    @staticmethod
    def arange(start, stop=None, step=1, dtype=None, requires_grad=None) -> Tensor:
        """`start, start + step, ...` below `stop` (or `0 .. start`)."""
        if stop is None:
            start, stop = 0, start
        if all(isinstance(v, int) for v in (start, stop, step)):
            t = Tensor._new(_ext.arange(start, stop, step))
        else:
            n = max(0, math.ceil((stop - start) / step))
            t = Tensor._new(_ext.arange(0, n, 1)) * step + start
        return Tensor._make(t._h, dtype, requires_grad)

    @staticmethod
    def eye(n: int, m: int | None = None, dtype=None, requires_grad=None) -> Tensor:
        dt = to_dtype(dtype or dtypes.default_float)
        t = Tensor._new(_ext.eye(n, dt.code))
        if m is not None and m != n:
            t = t.pad(((0, 0), (0, m - n))) if m > n else t.shrink(((0, n), (0, m)))
        return Tensor._make(t._h, None, requires_grad)

    @staticmethod
    def manual_seed(seed: int = 0):
        _ext.manual_seed(seed)

    @staticmethod
    def _random(name, shape, a=0.0, b=0.0, dtype=None, requires_grad=None):
        return Tensor._make(
            _ext.random(name, _shape_arg(shape), a, b), dtype, requires_grad
        )

    @staticmethod
    def rand(*shape, dtype=None, requires_grad=None) -> Tensor:
        return Tensor._random("rand", shape, dtype=dtype, requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, dtype=None, requires_grad=None) -> Tensor:
        return Tensor._random("randn", shape, dtype=dtype, requires_grad=requires_grad)

    @staticmethod
    def normal(*shape, mean=0.0, std=1.0, dtype=None, requires_grad=None) -> Tensor:
        t = Tensor.randn(*shape) * std + mean
        return Tensor._make(t._h, dtype, requires_grad)

    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, dtype=None, requires_grad=None) -> Tensor:
        return Tensor._random("uniform", shape, low, high, dtype, requires_grad)

    @staticmethod
    def randint(*shape, low=0, high=10, dtype=None, requires_grad=None) -> Tensor:
        return Tensor._random("randint", shape, low, high, dtype, requires_grad)

    @staticmethod
    def kaiming_uniform(*shape, a: float = 0.01, dtype=None, requires_grad=None) -> Tensor:
        return Tensor._random("kaiming_uniform", shape, a, 0.0, dtype, requires_grad)

    @staticmethod
    def glorot_uniform(*shape, dtype=None, requires_grad=None) -> Tensor:
        return Tensor._random("glorot_uniform", shape, 0.0, 0.0, dtype, requires_grad)

    def rand_like(self, dtype=None, requires_grad=None) -> Tensor:
        return Tensor.rand(*self.shape, dtype=dtype or self.dtype, requires_grad=requires_grad)

    def randn_like(self, dtype=None, requires_grad=None) -> Tensor:
        return Tensor.randn(*self.shape, dtype=dtype or self.dtype, requires_grad=requires_grad)

    # ---- properties

    @property
    def shape(self) -> tuple:
        if self._shape is None:
            self._shape = tuple(_ext.shape(self._h))
        return self._shape

    @property
    def dtype(self) -> DType:
        if self._dtype is None:
            self._dtype = dtypes.from_code(_ext.dtype(self._h))
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def device(self) -> str:
        return "CPU"

    def numel(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return self.dtype.itemsize

    def nbytes(self) -> int:
        return self.numel() * self.dtype.itemsize

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __repr__(self) -> str:
        state = "realized" if _ext.is_realized(self._h) else "lazy"
        return f"<Tensor shape={self.shape} dtype={self.dtype} {state}>"

    def __hash__(self) -> int:
        return id(self)

    def __bool__(self) -> bool:
        return bool(self.item())

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    # ---- realization and data

    def realize(self, *others: Tensor) -> Tensor:
        """Computes this tensor's value now (with `others`, in one
        schedule, so shared work is done once)."""
        _ext.realize([self._h] + [t._h for t in others])
        return self

    def _memory(self):
        """A read-only memoryview of the elements in row-major order,
        which keeps their buffer alive."""
        if self.dtype == dtypes.bfloat16:
            raise TypeError("bfloat16 has no Python format; cast it first")
        h, address = _ext.memory(self._h)
        n = self.nbytes()
        buf = (ctypes.c_char * n).from_address(address) if n else bytearray()
        mv = memoryview(_Kept(buf, h)).cast("B")
        return mv.cast(self.dtype.fmt) if n else mv

    def numpy(self):
        """The value as a NumPy array: a read-only view of the tensor's
        memory, with no copy when the tensor is a realized buffer."""
        if _np is None:
            raise ImportError("numpy() needs NumPy")
        if self.dtype == dtypes.bfloat16:
            return self.float().numpy()
        if self.numel() == 0:
            return _np.zeros(self.shape, dtype=self.dtype.name)
        h, address = _ext.memory(self._h)
        buf = (ctypes.c_char * self.nbytes()).from_address(address)
        a = _np.frombuffer(_Kept(buf, h), dtype=self.dtype.name).reshape(self.shape)
        a.flags.writeable = False
        return a

    def tolist(self):
        """The value as nested lists (a number for a 0-d tensor)."""
        if self.dtype == dtypes.bfloat16:
            return self.float().tolist()
        flat = self._memory().tolist()
        if not self.shape:
            return flat[0]
        out = flat
        for d in reversed(self.shape[1:]):
            out = [out[i : i + d] for i in range(0, len(out), d)]
        return out

    def item(self):
        """The single element, as a Python number."""
        return _ext.item(self._h)

    def data(self) -> memoryview:
        """A read-only memoryview of the value, shaped like the tensor."""
        mv = self._memory()
        return mv.cast("B").cast(self.dtype.fmt, self.shape) if self.ndim > 1 else mv

    # ---- gradients and state

    @property
    def requires_grad(self) -> bool | None:
        return _ext.requires_grad(self._h)

    @requires_grad.setter
    def requires_grad(self, value: bool):
        _ext.requires_grad_(self._h, bool(value))

    def requires_grad_(self, requires_grad: bool = True) -> Tensor:
        _ext.requires_grad_(self._h, bool(requires_grad))
        return self

    @property
    def grad(self) -> Tensor | None:
        g = _ext.grad(self._h)
        return None if g is None else Tensor._new(g)

    @grad.setter
    def grad(self, value):
        if value is not None:
            raise ValueError("only `t.grad = None` is supported")
        _ext.zero_grad(self._h)

    def backward(self) -> Tensor:
        """Computes the gradient of this scalar with respect to every
        tensor that requires one."""
        _ext.backward(self._h)
        return self

    def gradient(self, *targets: Tensor) -> list[Tensor]:
        """The gradients of this scalar with respect to `targets`."""
        return [Tensor._new(h) for h in _ext.gradient(self._h, [t._h for t in targets])]

    def detach(self) -> Tensor:
        return Tensor._new(_ext.unary("detach", self._h))

    def assign(self, x) -> Tensor:
        """Gives this tensor (every reference to it) the value `x`."""
        if not isinstance(x, Tensor):
            x = Tensor(x, dtype=self.dtype)
        _ext.assign(self._h, x._h)
        return self

    def contiguous(self) -> Tensor:
        return Tensor._new(_ext.unary("contiguous", self._h))

    def clone(self) -> Tensor:
        return self.contiguous()

    # ---- elementwise

    def _operand(self, x):
        if isinstance(x, Tensor):
            return x._h
        if isinstance(x, (bool, int, float)):
            return _ext.scalar(self._h, float(x), isinstance(x, int))
        return Tensor(x)._h

    def _binary(self, op: str, x, reverse: bool = False) -> Tensor:
        o = self._operand(x)
        if reverse:
            return Tensor._new(_ext.binary(op, o, self._h))
        return Tensor._new(_ext.binary(op, self._h, o))

    def add(self, x, reverse=False):
        return self._binary("add", x, reverse)

    def sub(self, x, reverse=False):
        return self._binary("sub", x, reverse)

    def mul(self, x, reverse=False):
        return self._binary("mul", x, reverse)

    def div(self, x, reverse=False):
        return self._binary("div", x, reverse)

    def idiv(self, x, reverse=False):
        return self._binary("idiv", x, reverse)

    def mod(self, x, reverse=False):
        return self._binary("mod", x, reverse)

    def maximum(self, x):
        return self._binary("maximum", x)

    def minimum(self, x):
        return self._binary("minimum", x)

    def pow(self, x, reverse=False):
        if not reverse and isinstance(x, (int, float)) and not isinstance(x, bool):
            return Tensor._new(_ext.scalar_pow(self._h, float(x)))
        return self._binary("pow", x, reverse)

    def lt(self, x):
        return self._binary("lt", x)

    def gt(self, x):
        return self._binary("gt", x)

    def le(self, x):
        return self._binary("le", x)

    def ge(self, x):
        return self._binary("ge", x)

    def eq(self, x):
        return self._binary("eq", x)

    def ne(self, x):
        return self._binary("ne", x)

    def logical_and(self, x):
        return self._binary("logical_and", x)

    def logical_or(self, x):
        return self._binary("logical_or", x)

    def __add__(self, x):
        return self._binary("add", x)

    def __radd__(self, x):
        return self._binary("add", x, True)

    def __sub__(self, x):
        return self._binary("sub", x)

    def __rsub__(self, x):
        return self._binary("sub", x, True)

    def __mul__(self, x):
        return self._binary("mul", x)

    def __rmul__(self, x):
        return self._binary("mul", x, True)

    def __truediv__(self, x):
        return self._binary("div", x)

    def __rtruediv__(self, x):
        return self._binary("div", x, True)

    def __floordiv__(self, x):
        return self._binary("idiv", x)

    def __rfloordiv__(self, x):
        return self._binary("idiv", x, True)

    def __mod__(self, x):
        return self._binary("mod", x)

    def __rmod__(self, x):
        return self._binary("mod", x, True)

    def __pow__(self, x):
        return self.pow(x)

    def __rpow__(self, x):
        return self.pow(x, True)

    def __matmul__(self, x):
        return self.matmul(x)

    def __rmatmul__(self, x):
        return Tensor(x, dtype=self.dtype).matmul(self)

    def __neg__(self):
        return self.neg()

    def __abs__(self):
        return self.abs()

    def __lt__(self, x):
        return self.lt(x)

    def __gt__(self, x):
        return self.gt(x)

    def __le__(self, x):
        return self.le(x)

    def __ge__(self, x):
        return self.ge(x)

    def __eq__(self, x):  # type: ignore[override]
        return self.eq(x)

    def __ne__(self, x):  # type: ignore[override]
        return self.ne(x)

    def __and__(self, x):
        return self.logical_and(x)

    def __or__(self, x):
        return self.logical_or(x)

    def __invert__(self):
        return self.logical_not()

    def where(self, x, y) -> Tensor:
        """`x` where this tensor is true, else `y`."""
        if not isinstance(x, Tensor) and not isinstance(y, Tensor):
            dt = _infer_dtype([x, y])
            x, y = Tensor(x, dtype=dt), Tensor(y, dtype=dt)
        elif not isinstance(x, Tensor):
            x = Tensor._new(y._operand(x))
        elif not isinstance(y, Tensor):
            y = Tensor._new(x._operand(y))
        return Tensor._new(_ext.where(self._h, x._h, y._h))

    def cast(self, dtype) -> Tensor:
        dt = to_dtype(dtype)
        if dt == self.dtype:
            return self
        return Tensor._new(_ext.cast(self._h, dt.code))

    def float(self) -> Tensor:
        return self.cast(dtypes.float32)

    def half(self) -> Tensor:
        return self.cast(dtypes.float16)

    def int(self) -> Tensor:
        return self.cast(dtypes.int32)

    def bool(self) -> Tensor:
        return self.cast(dtypes.bool)

    def leaky_relu(self, neg_slope: float = 0.01) -> Tensor:
        return Tensor._new(_ext.with_number("leaky_relu", self._h, neg_slope, 0))

    def softplus(self, beta: float = 1.0) -> Tensor:
        return Tensor._new(_ext.with_number("softplus", self._h, beta, 0))

    def clip(self, min_=None, max_=None) -> Tensor:
        lo = -math.inf if min_ is None else min_
        hi = math.inf if max_ is None else max_
        return Tensor._new(_ext.with_number("clip", self._h, lo, hi))

    clamp = clip

    def dropout(self, p: float = 0.5) -> Tensor:
        return Tensor._new(_ext.with_number("dropout", self._h, p, 0))

    def one_hot(self, num_classes: int) -> Tensor:
        return Tensor._new(_ext.with_number("one_hot", self._h, num_classes, 0))

    def tril(self, diagonal: int = 0) -> Tensor:
        return Tensor._new(_ext.with_number("tril", self._h, diagonal, 0))

    def triu(self, diagonal: int = 0) -> Tensor:
        return Tensor._new(_ext.with_number("triu", self._h, diagonal, 0))

    # ---- reductions

    def _reduce(self, op, axis, keepdim, correction=1) -> Tensor:
        if axis is None:
            axes = ()
        elif isinstance(axis, int):
            axes = (axis,)
        else:
            axes = tuple(axis)
            if not axes:
                return self
        return Tensor._new(_ext.reduce(op, self._h, axes, keepdim, correction))

    def sum(self, axis=None, keepdim=False, dtype=None) -> Tensor:
        t = self if dtype is None else self.cast(dtype)
        return t._reduce("sum", axis, keepdim)

    def max(self, axis=None, keepdim=False) -> Tensor:
        return self._reduce("max", axis, keepdim)

    def min(self, axis=None, keepdim=False) -> Tensor:
        return self._reduce("min", axis, keepdim)

    def mean(self, axis=None, keepdim=False) -> Tensor:
        return self._reduce("mean", axis, keepdim)

    def var(self, axis=None, keepdim=False, correction=1) -> Tensor:
        return self._reduce("var", axis, keepdim, correction)

    def std(self, axis=None, keepdim=False, correction=1) -> Tensor:
        return self._reduce("std", axis, keepdim, correction)

    def _along(self, op, axis, keepdim=False) -> Tensor:
        return Tensor._new(_ext.along(op, self._h, axis, keepdim))

    def argmax(self, axis=None, keepdim=False) -> Tensor:
        if axis is None:
            return self.flatten()._along("argmax", 0)
        r = self._along("argmax", axis)
        return r.unsqueeze(axis) if keepdim else r

    def argmin(self, axis=None, keepdim=False) -> Tensor:
        if axis is None:
            return self.flatten()._along("argmin", 0)
        r = self._along("argmin", axis)
        return r.unsqueeze(axis) if keepdim else r

    def softmax(self, axis=-1) -> Tensor:
        return self._along("softmax", axis)

    def log_softmax(self, axis=-1) -> Tensor:
        return self._along("log_softmax", axis)

    def logsumexp(self, axis=None, keepdim=False) -> Tensor:
        if axis is None:
            r = self.flatten()._along("logsumexp", 0, False)
            return r.reshape((1,) * self.ndim) if keepdim else r
        return self._along("logsumexp", axis, keepdim)

    # ---- movement

    def _move(self, op, arg) -> Tensor:
        return Tensor._new(_ext.movement(op, self._h, tuple(arg)))

    def reshape(self, *shape) -> Tensor:
        shape = _shape_arg(shape)
        shape = tuple(self.shape[i] if s is None else s for i, s in enumerate(shape))
        if -1 in shape:
            known = math.prod(s for s in shape if s != -1)
            shape = tuple(self.numel() // known if s == -1 else s for s in shape)
        if shape == self.shape:
            return self
        return self._move("reshape", shape)

    view = reshape

    def permute(self, *order) -> Tensor:
        order = tuple(o % self.ndim for o in _shape_arg(order))
        return self._move("permute", order)

    def expand(self, *shape) -> Tensor:
        shape = _shape_arg(shape)
        lead = len(shape) - self.ndim
        shape = tuple(
            self.shape[i - lead] if s == -1 else s for i, s in enumerate(shape)
        )
        if shape == self.shape:
            return self
        x = self.reshape((1,) * lead + self.shape) if lead else self
        return x._move("expand", shape)

    def pad(self, padding, value: float = 0.0) -> Tensor:
        """Pads with `value`. `padding` is a `(before, after)` pair (or
        None) per dimension, or flat in PyTorch's order: the last
        dimension's pair first."""
        if padding and all(isinstance(p, int) for p in padding):
            pairs = [(0, 0)] * self.ndim
            for i in range(len(padding) // 2):
                pairs[self.ndim - 1 - i] = (padding[2 * i], padding[2 * i + 1])
        else:
            pairs = [(0, 0) if p is None else tuple(p) for p in padding]
        if any(b < 0 or a < 0 for b, a in pairs):
            bounds = [(max(0, -b), s - max(0, -a)) for (b, a), s in zip(pairs, self.shape)]
            return self.shrink(bounds).pad([(max(0, b), max(0, a)) for b, a in pairs], value)
        flat = [v for p in pairs for v in p]
        out = self._move("pad", flat)
        if value != 0.0:
            mask = Tensor.ones(*self.shape, dtype=dtypes.bool)._move("pad", flat)
            out = mask.where(out, value)
        return out

    def shrink(self, arg) -> Tensor:
        """Keeps `[lo, hi)` of each dimension; `arg` holds a pair (or
        None, for all) per dimension."""
        flat = []
        for p, s in zip(arg, self.shape):
            flat.extend((0, s) if p is None else p)
        return self._move("shrink", flat)

    def flip(self, axis) -> Tensor:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        return self._move("flip", [a % self.ndim for a in axes])

    def repeat(self, *repeats) -> Tensor:
        repeats = _shape_arg(repeats)
        x = self.reshape((1,) * (len(repeats) - self.ndim) + self.shape)
        return x._move("repeat", repeats)

    def transpose(self, dim0: int = 1, dim1: int = 0) -> Tensor:
        order = list(range(self.ndim))
        order[dim0], order[dim1] = order[dim1], order[dim0]
        return self.permute(order)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        if self.ndim == 0:
            return self.reshape(1)
        s, e = start_dim % self.ndim, end_dim % self.ndim
        return self.reshape(self.shape[:s] + (math.prod(self.shape[s : e + 1]),) + self.shape[e + 1 :])

    def unflatten(self, dim: int, sizes) -> Tensor:
        dim %= self.ndim
        return self.reshape(self.shape[:dim] + tuple(sizes) + self.shape[dim + 1 :])

    def squeeze(self, dim: int | None = None) -> Tensor:
        if dim is None:
            return self.reshape(tuple(s for s in self.shape if s != 1))
        dim %= max(1, self.ndim)
        if self.ndim == 0 or self.shape[dim] != 1:
            return self
        return self.reshape(self.shape[:dim] + self.shape[dim + 1 :])

    def unsqueeze(self, dim: int) -> Tensor:
        dim = dim % (self.ndim + 1)
        return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

    def cat(self, *others: Tensor, dim: int = 0) -> Tensor:
        return Tensor._new(_ext.join("cat", [self._h] + [t._h for t in others], dim))

    def stack(self, *others: Tensor, dim: int = 0) -> Tensor:
        return Tensor._new(_ext.join("stack", [self._h] + [t._h for t in others], dim))

    def chunk(self, chunks: int, dim: int = 0) -> list[Tensor]:
        size = math.ceil(self.shape[dim] / chunks)
        return self.split(size, dim)

    def split(self, sizes, dim: int = 0) -> list[Tensor]:
        dim %= self.ndim
        n = self.shape[dim]
        if isinstance(sizes, int):
            sizes = [min(sizes, n - i) for i in range(0, n, sizes)]
        out, start = [], 0
        for s in sizes:
            out.append(self.shrink([(start, start + s) if d == dim else None for d in range(self.ndim)]))
            start += s
        return out

    def __getitem__(self, index) -> Tensor:
        """Indexing by integers, slices (any step), None and `...`; or by
        one integer tensor, which gathers rows of dimension 0 (eagerly,
        and without a gradient to this tensor)."""
        if isinstance(index, Tensor):
            if not index.dtype.is_int:
                raise TypeError("a tensor index must have an integer type")
            rows = Tensor._new(_ext.take_rows(self._h, index.flatten()._h))
            return rows.reshape(index.shape + self.shape[1:])
        if not isinstance(index, tuple):
            index = (index,)
        if any(isinstance(i, Tensor) for i in index):
            raise NotImplementedError("a tensor index with other indices")
        n_real = sum(1 for i in index if i is not None and i is not Ellipsis)
        if n_real > self.ndim:
            raise IndexError(f"too many indices for a {self.ndim}-d tensor")
        if Ellipsis in index:
            k = index.index(Ellipsis)
            index = index[:k] + (slice(None),) * (self.ndim - n_real) + index[k + 1 :]
        else:
            index = index + (slice(None),) * (self.ndim - n_real)
        x, d, final = self, 0, []
        for i in index:
            if i is None:
                final.append(1)
                continue
            size = x.shape[d]
            if isinstance(i, int):
                j = i + size if i < 0 else i
                if not 0 <= j < size:
                    raise IndexError(f"index {i} is out of bounds for size {size}")
                x = x.shrink([(j, j + 1) if e == d else None for e in range(x.ndim)])
            elif isinstance(i, slice):
                start, stop, step = i.indices(size)
                x = x._slice(d, start, stop, step)
                final.append(x.shape[d])
            else:
                raise TypeError(f"cannot index with {type(i).__name__}")
            d += 1
        return x.reshape(tuple(final))

    def _slice(self, d: int, start: int, stop: int, step: int) -> Tensor:
        """`[start:stop:step]` of dimension `d`, by shrink, flip and
        reshape."""
        x = self
        if step < 0:
            size = x.shape[d]
            x = x.flip(d)
            start, stop, step = size - 1 - start, size - 1 - stop, -step
        n = max(0, math.ceil((stop - start) / step))
        x = x.shrink([(start, start + max(0, min(n * step, x.shape[d] - start))) if e == d else None for e in range(x.ndim)])
        if step == 1 or n == 0:
            return x
        # Pad to n * step, view as (n, step), keep column 0.
        x = x.pad([(0, n * step - x.shape[d]) if e == d else None for e in range(x.ndim)])
        x = x.reshape(x.shape[:d] + (n, step) + x.shape[d + 1 :])
        x = x.shrink([(0, 1) if e == d + 1 else None for e in range(x.ndim)])
        return x.reshape(x.shape[:d + 1] + x.shape[d + 2 :])

    def sequential(self, ll) -> Tensor:
        """Applies each function of `ll` in turn."""
        x = self
        for f in ll:
            x = f(x)
        return x

    def __setitem__(self, index, value):
        raise TypeError("tensors are immutable; use assign")

    # ---- linear algebra and layers

    def dot(self, w: Tensor) -> Tensor:
        return Tensor._new(_ext.binary("dot", self._h, w._h))

    def matmul(self, x, reverse=False) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x, dtype=self.dtype)
        return x.dot(self) if reverse else self.dot(x)

    def linear(self, weight: Tensor, bias: Tensor | None = None) -> Tensor:
        return Tensor._new(_ext.linear(self._h, weight._h, None if bias is None else bias._h))

    def conv2d(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0) -> Tensor:
        return Tensor._new(
            _ext.conv2d(
                self._h,
                weight._h,
                None if bias is None else bias._h,
                groups,
                _one(stride, "stride"),
                _one(dilation, "dilation"),
                _one(padding, "padding"),
            )
        )

    def max_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1, padding=0) -> Tensor:
        return self._pool("max_pool2d", kernel_size, stride, dilation, padding)

    def avg_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1, padding=0) -> Tensor:
        return self._pool("avg_pool2d", kernel_size, stride, dilation, padding)

    def _pool(self, op, kernel_size, stride, dilation, padding) -> Tensor:
        if _one(dilation, "dilation") != 1 or _one(padding, "padding") != 0:
            raise NotImplementedError("pooling supports neither dilation nor padding")
        k = _one(kernel_size, "kernel_size")
        s = -1 if stride is None else _one(stride, "stride")
        return Tensor._new(_ext.pool(op, self._h, k, s))

    def batchnorm(self, weight, bias, mean, invstd, axis=1) -> Tensor:
        if axis != 1:
            raise NotImplementedError("batchnorm along an axis other than 1")
        return Tensor._new(_ext.batchnorm(self._h, weight._h, bias._h, mean._h, invstd._h))

    def layernorm(self, axis: int = -1, eps: float = 1e-5) -> Tensor:
        return Tensor._new(_ext.with_number("layernorm", self._h, axis, eps))

    def scaled_dot_product_attention(
        self, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    ) -> Tensor:
        if dropout_p:
            raise NotImplementedError("attention dropout")
        mask = None if attn_mask is None else attn_mask._h
        return Tensor._new(_ext.attention(self._h, key._h, value._h, mask, is_causal))

    def sparse_categorical_crossentropy(self, Y: Tensor, ignore_index=-1, label_smoothing=0.0) -> Tensor:
        if label_smoothing:
            raise NotImplementedError("label smoothing")
        return Tensor._new(_ext.binary("sparse_categorical_crossentropy", self._h, Y._h))

    def binary_crossentropy_logits(self, Y: Tensor) -> Tensor:
        return (self.maximum(0) - Y * self + (1 + self.abs().neg().exp()).log()).mean()

    # ---- training mode

    class train(contextlib.ContextDecorator):
        """`with Tensor.train():` turns training mode on (or off, with
        `mode=False`) for the block, as in tinygrad."""

        def __init__(self, mode: bool = True):
            self.mode = mode

        def __enter__(self):
            self.prev = _ext.training()
            _ext.train(self.mode)

        def __exit__(self, *exc):
            _ext.train(self.prev)

    class test(train):
        def __init__(self, mode: bool = False):
            super().__init__(mode)


def _unary(name: str):
    def f(self: Tensor) -> Tensor:
        return Tensor._new(_ext.unary(name, self._h))

    f.__name__ = name
    return f


for _name in (
    "neg relu exp log sqrt rsqrt reciprocal sigmoid tanh gelu silu sin cos tan"
    " trunc floor ceil round square abs sign logical_not"
).split():
    setattr(Tensor, _name, _unary(_name))

Tensor.swish = Tensor.silu
Tensor.exp2 = lambda self: (self * math.log(2)).exp()
Tensor.log2 = lambda self: self.log() * (1 / math.log(2))
Tensor.relu6 = lambda self: self.relu() - (self - 6).relu()
Tensor.hardtanh = lambda self, min_val=-1, max_val=1: self.clip(min_val, max_val)
Tensor.quick_gelu = lambda self: self * (self * 1.702).sigmoid()
Tensor.elu = lambda self, alpha=1.0: self.relu() - alpha * (1 - self.exp()).relu()


def _one(v, what: str) -> int:
    """An integer argument that may be given as an equal pair."""
    if isinstance(v, int):
        return v
    v = tuple(v)
    if len(set(v)) != 1:
        raise NotImplementedError(f"different {what} per dimension")
    return v[0]


class _Kept:
    """A buffer-protocol view of `buf` that keeps the engine tensor `h`
    (and so its memory) alive."""

    __slots__ = ("buf", "h")

    def __init__(self, buf, h):
        self.buf = buf
        self.h = h

    def __buffer__(self, flags):
        return memoryview(self.buf)


def _from_values(values: list, shape: tuple, dt: DType):
    if dt.fmt is None:
        return _ext.cast(_from_values(values, shape, dtypes.float32), dt.code)
    if dt == dtypes.bool:
        a = array.array("B", (1 if v else 0 for v in values))
    elif dt.fmt == "e":
        return _ext.cast(_from_values(values, shape, dtypes.float32), dt.code)
    else:
        a = array.array(dt.fmt, values)
    address = a.buffer_info()[0]
    h = _ext.from_memory(address, shape, dt.code)
    del a
    return h


def _from_numpy(a, dt: DType | None):
    if a.dtype.name not in [d.name for d in dtypes.ALL]:
        raise TypeError(f"no minigrad type for NumPy's {a.dtype}")
    a = _np.ascontiguousarray(a)
    h = _ext.from_memory(a.ctypes.data, a.shape, dtypes.from_name(a.dtype.name).code)
    if dt is not None and dt.name != a.dtype.name:
        h = _ext.cast(h, dt.code)
    return h
