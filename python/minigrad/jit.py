"""`TinyJit`: capture a step function's kernels and replay them, through
the engine's `MiniJit` (`minigrad/jit.mojo`, whose docstring says what a
step may do)."""

from __future__ import annotations

import functools

from . import _ext
from .tensor import Tensor


def _flatten(ret, out: list):
    """`ret`'s tensors in order, and a template to rebuild `ret` from new
    ones."""
    if isinstance(ret, Tensor):
        out.append(ret)
        return ("T",)
    if isinstance(ret, (list, tuple)):
        return (type(ret), [_flatten(r, out) for r in ret])
    if ret is None or isinstance(ret, (int, float, bool, str)):
        return ("V", ret)
    raise TypeError(f"TinyJit: cannot return {type(ret).__name__}")


def _rebuild(template, tensors):
    if template[0] == "T":
        return next(tensors)
    if template[0] == "V":
        return template[1]
    kind, parts = template
    return kind(_rebuild(p, tensors) for p in parts)


class TinyJit:
    """Wraps a step function of tensors, as tinygrad's `TinyJit` does:
    the first call runs it, the second runs it while recording the
    kernels it causes, and later calls replay the recording on the new
    inputs without running Python at all.

    The inputs are the tensor arguments (positional, then keyword in
    sorted order); they must keep their shapes and types. Everything
    else the function reads, numbers included, is fixed at the capture.
    Returned tensors live in the capture's buffers: the next call
    overwrites them.
    """

    def __init__(self, fxn):
        self.fxn = fxn
        self.cnt = 0
        self._jit = _ext.jit_new()
        self._template = None
        self.__wrapped__ = fxn

    def _inputs(self, args, kwargs) -> list:
        ts = [a for a in args if isinstance(a, Tensor)]
        ts += [kwargs[k] for k in sorted(kwargs) if isinstance(kwargs[k], Tensor)]
        return [t._h for t in ts]

    def reset(self):
        self.cnt = 0
        self._jit = _ext.jit_new()
        self._template = None

    def __call__(self, *args, **kwargs):
        inputs = self._inputs(args, kwargs)
        self.cnt += 1
        if self.cnt == 1:
            ret = self.fxn(*args, **kwargs)
            outs: list = []
            _flatten(ret, outs)
            _ext.realize([t._h for t in outs])
            return ret
        if self.cnt == 2:
            _ext.jit_begin(self._jit, inputs)
            try:
                ret = self.fxn(*args, **kwargs)
                outs = []
                self._template = _flatten(ret, outs)
                new = _ext.jit_end(self._jit, [t._h for t in outs])
            except BaseException:
                _ext.jit_abort(self._jit)
                self.reset()
                raise
            return _rebuild(self._template, iter(Tensor._new(h) for h in new))
        new = _ext.jit_replay(self._jit, inputs)
        return _rebuild(self._template, iter(Tensor._new(h) for h in new))

    def __get__(self, obj, objtype=None):
        """As a method decorator, `model.step(x)` calls the function with
        `model` first. One capture serves every instance, as in tinygrad,
        so use it on one instance."""
        if obj is None:
            return self
        return functools.partial(self.__call__, obj)
