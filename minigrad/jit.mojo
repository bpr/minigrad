"""Capture and replay of a step function (`minigrad.jit`).

`MiniJit` runs a function of tensors the way tinygrad's `TinyJit` does:
the first call runs it normally, the second runs it while recording
every kernel and eager operation it causes, and every later call replays
the recording with the new inputs' buffers, without running the
function's code at all. No graph is built, scheduled, or planned.

```mojo
var jit = MiniJit()
def step(xs: List[Tensor]) raises -> List[Tensor]:
    opt.zero_grad()
    var loss = model(xs[0]).sparse_categorical_crossentropy(xs[1])
    loss.backward()
    opt.step([loss])
    return [loss]
for batch in batches:
    var loss = jit(step, [batch.x, batch.y])[0].item()
```

What a replay does, and so what the function may do:

- **Inputs** are the tensors passed in. Every call must pass tensors of
  the shapes and element types of the second call. Anything else the
  function reads (weights, a cache) is read from the buffers it had
  during the capture.
- **Host values are fixed at the capture.** Python-style numbers, loop
  counts, and branches are not re-evaluated: a learning rate or a shape
  stays what it was. Values that change belong in tensors: pass them as
  inputs. `take_rows` and `write_` take their indices as tensors for
  this; their host-integer forms raise during a capture.
- **State is updated with `assign`.** Each tensor assigned during the
  capture (parameters, optimizer moments) keeps the buffer it had
  before; a replay computes the new value and copies it back there.
  Other ways of changing state are not replayed.
- **Outputs** are tensors on the capture's output buffers: the next call
  overwrites them. Read or copy them before calling again.
- The function must not read values on the host (`item`, `tolist`).
"""

from std.memory import unsafe_memcpy
from std.time import perf_counter_ns

from minigrad.dtype import DT
from minigrad.graph import BytePtr, OP_BUFFER, graph
from minigrad.runtime.kernel import (
    JitStep,
    STEP_COPY,
    STEP_GATHER,
    STEP_KERNEL,
    STEP_SCATTER,
    run_copy,
    run_gather,
    run_scatter,
    run_plan,
    runtime,
)
from minigrad.tensor import Tensor, jit_state
from minigrad.uop import UOp


struct MiniJit(Movable):
    """A captured step function; see the module docstring."""

    var calls: Int
    var tape: List[JitStep]
    var inputs: List[Int]
    """The buffer of each input during the capture."""
    var in_shapes: List[List[Int]]
    var in_dtypes: List[DT]
    var outputs: List[Int]
    """The buffer of each output."""
    var out_shapes: List[List[Int]]
    var out_dtypes: List[DT]
    var state_before: List[Int]
    """For each tensor assigned in the step, the buffer the tape reads
    its value from..."""
    var state_after: List[Int]
    """...and the buffer the tape writes its new value to."""
    var held: List[Int]
    """Buffers the capture keeps alive (one entry per hold)."""

    def __init__(out self):
        self.calls = 0
        self.tape = List[JitStep]()
        self.inputs = List[Int]()
        self.in_shapes = List[List[Int]]()
        self.in_dtypes = List[DT]()
        self.outputs = List[Int]()
        self.out_shapes = List[List[Int]]()
        self.out_dtypes = List[DT]()
        self.state_before = List[Int]()
        self.state_after = List[Int]()
        self.held = List[Int]()

    def __deinit__(deinit self):
        var g = graph()
        for b in self.held:
            g[].unhold(b)

    def __call__[
        F: def(List[Tensor]) raises -> List[Tensor]
    ](mut self, f: F, inputs: List[Tensor]) raises -> List[Tensor]:
        """Runs, captures, or replays `f` on `inputs`.

        Parameters:
            F: The function's type.

        Args:
            f: The step function. Pass the same one on every call.
            inputs: Its tensor inputs.

        Returns:
            Its outputs, realized.

        Raises:
            If a replay's inputs differ in shape or type from the
            capture's, or `f` does something that cannot be captured.
        """
        self.calls += 1
        if self.calls == 1:
            var out = f(inputs)
            Tensor.realize_all(out)
            return out^
        if self.calls == 2:
            return self._capture(f, inputs)
        return self.replay(inputs)

    def _capture[
        F: def(List[Tensor]) raises -> List[Tensor]
    ](mut self, f: F, inputs: List[Tensor]) raises -> List[Tensor]:
        self.begin_capture(inputs)
        var out: List[Tensor]
        try:
            out = f(inputs)
        except e:
            self.abort_capture()
            raise e^
        return self.end_capture(out)

    def begin_capture(mut self, inputs: List[Tensor]) raises:
        """Starts recording a step on `inputs`: what `__call__` does on
        its second call, for callers that run the step themselves (the
        Python bindings). End with `end_capture`, or `abort_capture` if
        the step fails."""
        var rt = runtime()
        for x in inputs:
            self.inputs.append(_buffer_of(x))
            self.in_shapes.append(x.shape())
            self.in_dtypes.append(x.dtype())
        rt[].tape.clear()
        jit_state()[].tensors.clear()
        jit_state()[].before.clear()
        rt[].recording = True

    def abort_capture(mut self):
        """Stops a recording without keeping it."""
        runtime()[].recording = False
        runtime()[].tape.clear()
        jit_state()[].tensors.clear()
        jit_state()[].before.clear()
        self.inputs.clear()
        self.in_shapes.clear()
        self.in_dtypes.clear()
        self.outputs.clear()

    def end_capture(mut self, outputs: List[Tensor]) raises -> List[Tensor]:
        """Finishes the recording begun by `begin_capture`; `outputs` are
        the step's outputs.

        Returns:
            The outputs, realized.
        """
        var g = graph()
        var rt = runtime()
        try:
            Tensor.realize_all(outputs)
            Tensor.realize_all(jit_state()[].tensors)
            for t in outputs:
                self.outputs.append(_buffer_of(t))
        except e:
            self.abort_capture()
            raise e^
        rt[].recording = False
        self.tape = rt[].tape.copy()
        rt[].tape.clear()
        # Steps held each buffer they touch; keep one hold per buffer
        # that is not an input (inputs change every call). Take the new
        # holds before dropping the old, so no buffer is freed between.
        var keep = List[Int]()
        for s in self.tape:
            for b in s.ins:
                if b not in self.inputs and b not in keep:
                    keep.append(b)
            if s.out not in self.inputs and s.out not in keep:
                keep.append(s.out)
        for b in self.outputs:
            if b not in keep:
                keep.append(b)
        for b in keep:
            g[].hold(b)
        for s in self.tape:
            for b in s.ins:
                g[].unhold(b)
            g[].unhold(s.out)
        self.held = keep^
        for t in outputs:
            self.out_shapes.append(t.shape())
            self.out_dtypes.append(t.dtype())
        # Assigned tensors go back to the buffers the tape reads, holding
        # the new values.
        ref st = jit_state()[]
        for i in range(len(st.tensors)):
            var before = st.before[i]
            if before.op() != OP_BUFFER:
                raise Error(
                    "MiniJit: a tensor assigned in the step must be realized"
                    " before it"
                )
            var b0 = g[].nodes[before.id].arg[0]
            var b1 = _buffer_of(st.tensors[i])
            if b0 == b1:
                continue
            for b in [b0, b1]:
                g[].hold(b)
                self.held.append(b)
            self.state_before.append(b0)
            self.state_after.append(b1)
            _copy_buffer(b1, b0)
            _ = st.tensors[i].assign(Tensor(before))
        st.tensors.clear()
        st.before.clear()
        return outputs.copy()

    def replay(self, inputs: List[Tensor]) raises -> List[Tensor]:
        """Runs the recorded step on `inputs`: what `__call__` does from
        its third call on.

        Returns:
            The outputs, on the capture's output buffers.

        Raises:
            If an input's shape or type differs from the capture's.
        """
        var g = graph()
        var rt = runtime()
        if len(inputs) != len(self.inputs):
            raise Error(
                t"MiniJit: {len(inputs)} inputs, captured with"
                t" {len(self.inputs)}"
            )
        var now = Dict[Int, Int]()
        for i in range(len(inputs)):
            if (
                inputs[i].shape() != self.in_shapes[i]
                or inputs[i].dtype() != self.in_dtypes[i]
            ):
                raise Error(
                    t"MiniJit: input {i} is {inputs[i].dtype()}"
                    t" {inputs[i].shape()}, captured as {self.in_dtypes[i]}"
                    t" {self.in_shapes[i]}"
                )
            now[self.inputs[i]] = _buffer_of(inputs[i])
        var ins = List[BytePtr]()
        var trace = rt[].debug >= 2
        var start = perf_counter_ns()
        for s in self.tape:
            var t0 = perf_counter_ns() if trace else 0
            ins.clear()
            for b in s.ins:
                ins.append(g[].buffers[now.get(b, b)].ptr)
            var dst = g[].buffers[now.get(s.out, s.out)].ptr
            if s.kind == STEP_KERNEL:
                if s.plan >= 0:
                    run_plan(rt[].plans[s.plan], dst, ins, s.instrs)
                    g[].stats.kernels_run += 1
                    if rt[].plans[s.plan].gemm:
                        g[].stats.gemm_kernels += 1
            elif s.kind == STEP_GATHER:
                run_gather(ins[0], ins[1], dst, s)
            elif s.kind == STEP_SCATTER:
                run_scatter(ins[0], ins[1], dst, s)
            else:
                run_copy(ins[0], ins[1], dst, s)
            if trace:
                var us = Float64(perf_counter_ns() - t0) / 1e3
                if s.kind == STEP_KERNEL and s.plan >= 0:
                    ref plan = rt[].plans[s.plan]
                    var kind = String("gemm  ") if plan.gemm else String(
                        "blocks"
                    )
                    print(t"replay {kind} {us} us  loops {plan.sizes}")
                else:
                    print(t"replay step{s.kind} {us} us")
        for i in range(len(self.state_before)):
            _copy_buffer(self.state_after[i], self.state_before[i])
        g[].stats.ns_execute += Int(perf_counter_ns() - start)
        var out = List[Tensor]()
        for i in range(len(self.outputs)):
            out.append(
                Tensor(
                    UOp.on_buffer(
                        self.outputs[i], self.out_dtypes[i], self.out_shapes[i]
                    )
                )
            )
        return out^


def _buffer_of(t: Tensor) raises -> Int:
    """The buffer holding `t`, realizing it in its own memory if need be.
    A view is copied, and `t` is pointed at the copy, so the step and
    later replays read the buffer the capture recorded."""
    var u = t._contiguous_uop()
    if u.id != t.uop().id:
        t._replace(u)
    return graph()[].nodes[t.uop().id].arg[0]


def _copy_buffer(src: Int, dst: Int):
    var g = graph()
    unsafe_memcpy(
        dest=g[].buffers[dst].ptr,
        src=g[].buffers[src].ptr,
        count=min(g[].buffers[src].nbytes, g[].buffers[dst].nbytes),
    )
