# minigrad from Python

`python/minigrad` is a Python package with tinygrad's API (`Tensor`,
`TinyJit`, `nn`, `nn.optim`, `nn.state`, `dtypes`) running on minigrad's
engine. Python code drives the engine without any Mojo: tinygrad's
`examples/beautiful_mnist.py` runs with its imports changed and three
small edits (`examples/beautiful_mnist.py`, 98% test accuracy in 70
steps).

```sh
pixi run python-build        # compiles python/minigrad/_ext.so
pixi run python-test         # python/tests, against NumPy
PYTHONPATH=python pixi run python examples/beautiful_mnist.py
```

```python
from minigrad import Tensor, TinyJit, nn, dtypes

x = Tensor.rand(4, 3, requires_grad=True)
loss = (x @ Tensor.ones(3, 2)).relu().sum()
loss.backward()
print(x.grad.numpy())
```

## How it is built

Two layers:

- **`_ext.mojo`**, a Python extension module compiled from Mojo
  (`std.python.bindings`). Each tensor crosses into Python as a
  `Handle` holding one engine `Tensor`. Its functions take plain Python
  values and handles. Operations of one signature share an entry
  point and are chosen by name (`_ext.unary("exp", h)`), so a new
  operation is one line there.
- **The Python package** does the API work: argument forms
  (`reshape(2, 3)` and `reshape((2, 3))`, `-1`, pad pairs in either
  order), scalars, operators, `__getitem__`, the layers, the
  optimizers, and `TinyJit`'s bookkeeping. The engine does everything
  else: graph building, scheduling, kernels, gradients.

A `Tensor` is a Python object around a handle. Copies of a Python
reference are the same tensor, as in tinygrad, and `assign` changes
what every reference sees.

**Data in** is copied once: a nested list goes through
`array.array`, and a NumPy array or a `bytes`-like object is copied
from its memory into a new engine buffer. The engine cannot adopt
memory it does not own: its buffers are pooled and immutable.

**Data out** is not copied. `numpy()` returns a read-only array over
the tensor's buffer, and that array keeps the buffer alive. A view (a
transpose, say) is first copied into order. `tolist()` and `data()` read
the same memory through a `memoryview`, so they need no NumPy.

**`TinyJit`** drives the engine's `MiniJit` through
`begin_capture`/`end_capture`/`replay`. The first call runs the
function and the second records it. Later calls replay the recording
on the new tensor arguments without running any Python. The function
may return tensors in tuples, lists, and nesting; the replay rebuilds
that structure. `minigrad/jit.mojo` lists what a step may do.

## What is covered

- **`Tensor`**:
  - creation from numbers, nested lists, NumPy, and buffers;
  - `zeros`/`ones`/`full`/`empty` and their `_like` forms, `arange`,
    `eye`, `rand`/`randn`/`uniform`/`normal`/`randint`,
    `kaiming_uniform`, `glorot_uniform`, `manual_seed`;
  - arithmetic and comparison operators with tinygrad's promotion
    rules for scalars;
  - the unary math and activations, `where`, `cast`;
  - the reductions (`sum`, `max`, `min`, `mean`, `var`, `std`,
    `argmax`, `softmax`, `logsumexp`) over an int, a tuple, or all
    axes;
  - the movement ops, including `pad` with a value and in PyTorch's
    flat order;
  - indexing by integers, slices with any step, `None`, and `...`, and
    by one integer tensor;
  - `cat`/`stack`/`split`/`chunk`, `matmul`, `linear`, `conv2d`,
    pooling, `batchnorm`, `layernorm`, attention,
    `sparse_categorical_crossentropy`, `dropout`, `one_hot`;
  - `backward`, `.grad`, `gradient`, `assign`, `realize`, `numpy`,
    `tolist`, `item`;
  - `Tensor.train()` as a context manager or decorator, and
    `Tensor.training`.
- **`nn`**: `Linear`, `Conv2d`, `BatchNorm`, `LayerNorm`, `RMSNorm`,
  `Embedding`; `nn.optim.SGD`/`Adam`/`AdamW` with `step()` and
  `schedule_step()`; `nn.state.get_state_dict`, `get_parameters`,
  `load_state_dict`, `safe_load`, `safe_save`; `nn.datasets.mnist`.

## Where it differs from tinygrad in Python

These differ on top of the engine's own differences
(`docs/differences-from-tinygrad.md`); the ones met first in Python
are these.

- **Call `backward()` before reading the loss.** `loss.item()` realizes
  the loss and ends its history, so a later `backward()` raises. The
  usual tinygrad form, `loss.backward()` then
  `loss.realize(*opt.schedule_step())`, works as it is.
- **`Tensor.randint` and the other random functions draw on the host**
  as they are called. Inside a `TinyJit` step they are fixed at the
  capture, so draw them outside and pass them in. tinygrad's are graph
  operations that a replay draws again.
- **Indexing by a tensor** supports one integer tensor gathering along
  dimension 0. It runs eagerly and has no gradient to the indexed
  tensor (minigrad's `take_rows`). `nn.Embedding` uses a one-hot
  product instead, so its table is trained.
- **`numpy()` arrays are read-only views.** A `TinyJit` output's array
  sees the next replay's values, as the output tensor does.
- **No `__setitem__`**: tensors are immutable here; use `assign`.
- **One device**, `"CPU"`. `device=` arguments, `to()`, `shard`,
  `Variable` and symbolic shapes are absent.
- **Pooling and `conv2d`** take one size, stride, dilation and padding
  for every spatial dimension (a pair of equal values is accepted).
- **Missing**: tinygrad's `function` decorator, `Context` and
  `getenv` helpers, `GlobalCounters`, `torch_load`, `LSTMCell`,
  `GroupNorm`, `ConvTranspose2d`, `cumsum`, `multinomial`, gather and
  scatter beyond dimension 0.

## Cost

Each operation is one call into the engine: a small MLP step of about
100 operations takes 94 µs from Python against 76 µs in Mojo. A
step that `TinyJit` captures runs no Python at all after the capture.
`benchmarks/README.md` compares `benchmarks/mlp_step.py` with the Mojo
and tinygrad versions.
