# Differences from tinygrad

minigrad follows tinygrad's API and semantics where it can. This page
lists where it does not, and why. Add to it whenever a deviation is
introduced. The Python package's own differences (host-side random
numbers inside `TinyJit`, indexing by tensors, read-only `numpy()`) are
in `docs/python.md`.

## Semantics

- **`assign` never writes in place.** `t.assign(x)` points `t` (every
  copy of it) at `x`'s node; tinygrad writes into `t`'s buffer when `t`
  is realized. The values any tensor reads are the same. What differs is
  that a tensor computed lazily from the old value and realized later
  sees the old value here (in tinygrad it may see the new one), and that
  there is no way to update part of a buffer. Reason: nodes are
  hash-consed, so tensors with equal values share buffers.
- **`+=`, `-=`, `*=`** rebind the tensor the same way.
- **`MiniJit`**, minigrad's `TinyJit`, is called as `jit(f, inputs)`
  with a list of tensors,
  not as a decorator. A replay copies each tensor assigned in the step
  back into its old buffer, where tinygrad's in-place `assign` needs no
  copy. Host values are fixed at capture as in tinygrad, but `take_rows`
  and `write_` with host integers raise during a capture instead of
  replaying stale indices; tinygrad uses symbolic `Variable`s for
  changing positions, minigrad tensors of indices.
- **`write_(dim, start, value)` is the one in-place write**, eager and
  outside the graph, for key-value caches; tinygrad writes caches with
  `assign` into a slice (`cache[:, start:end].assign(x)`). The tensor
  must own its buffer.
- **`put_rows_(rows, value)`** writes rows in place at scattered
  indices, for a batch's caches; tinygrad assigns into a slice.
- **Batched GPT-2 generation** takes prompts of different lengths, each
  in its own cache slot with its own mask; tinygrad's example repeats one
  prompt `batch_size` times. `Batcher` adds continuous batching, which
  tinygrad's GPT-2 example does not have.
- **`take_rows(ids)` is an eager, host-side embedding lookup** with a
  list of ints; tinygrad's `Embedding` is lazy and differentiable.
- **GPT-2 decode attends to the cache rounded up to 64 rows**, masked;
  tinygrad's example masks the whole `MAX_CONTEXT` cache and uses
  symbolic shapes. Both keep shapes fixed across steps for replay.
- **`scaled_dot_product_attention`'s `is_causal`** aligns the mask at the
  end (query `i` sees keys up to `L - T + i`), so it is right for a
  decoder with a cache; tinygrad's aligns it at the start.
- **Reading `p.grad()` after `optimizer.step()`** recomputes the
  gradient from whatever of the graph is left, as in tinygrad; realize
  it before the step if it is needed afterwards.
- **`item()`, `tolist()`, and `realize()` realize the tensor itself**,
  so the value is computed once however often it is read, but the tensor
  then has no history: `backward()` on it raises. In tinygrad `item()`
  realizes a copy and `backward()` still works afterwards. Call
  `backward()` first, and pass the loss to the optimizer
  (`opt.step([loss])`, tinygrad's `loss.realize(*opt.schedule_step())`)
  so it is computed in the same schedule as the update.
- **`max_pool2d` and `avg_pool2d`** take one kernel size and stride for
  both dimensions and no padding. Trailing input that does not fill a
  window is dropped (tinygrad's default `ceil_mode=False`).
- **`conv2d`** takes one stride, dilation, and padding for all spatial
  dimensions.
- **Integer division and modulo** truncate toward zero, as in C and
  tinygrad; `x / 0` and `x % 0` are 0 for integers rather than
  undefined.
- **Random numbers** come from a splitmix64 generator on the host, not
  tinygrad's counter-based Threefry, so seeds do not reproduce
  tinygrad's values.
- **Operations that can fail raise**, so code using tensors is in
  `raises` functions.

## API shape (Mojo is not Python)

- **Shapes and data are accessed by calls**: `t.shape()`, `t.dtype()`,
  `t.grad()`, `t.T()`.
- **Comparisons are methods**: `a.lt(b)`, `a.eq(b)`, ... returning
  `bool` tensors.
- **Indexing** is `t[i]` and `t[i, j]` with integers, plus
  `t.slice(dim, start, end)` and `t.select(dim, i)`. There is no slice
  syntax, fancy indexing, or `__setitem__` yet.
- **Variadic integers** stand in for tuples: `t.reshape(2, 3)`,
  `t.sum(0, 1)`, `t.permute(1, 0)`. `pad` and `shrink` take one flat
  list of `(before, after)` or `(lo, hi)` per dimension.
- **Scalars in arithmetic** are `Float64` or `Int` literals; an integer
  literal keeps an integer tensor's type and a float literal promotes it
  to `float32`, as in tinygrad.
- **`Tensor.var`** is spelled `` t.`var`() `` because `var` is a keyword.
- **Layers** implement the `Module` trait's `parameters()`; there is no
  `nn.state.get_parameters` that walks arbitrary objects. `collect(a, b,
  ...)` concatenates the parameters of several modules. As in tinygrad,
  the list includes state that is not trained (`BatchNorm`'s running
  estimates, marked `requires_grad_(False)`), which the optimizer
  realizes every step.
- **Training mode** is `Tensor.train(True)`, a global flag, rather than
  a context manager.

## Primitive operations

- **`EXP` and `LOG` are natural**, and `SUB` and `DIV` are primitives
  (tinygrad has `EXP2`, `LOG2`, and builds subtraction and division from
  `NEG` and `RECIP`). An interpreter pays per instruction, so fewer
  instructions matter more than a smaller instruction set.
- **`WINDOW`** is a movement op; tinygrad builds pooling from reshape,
  expand, and shrink. See `docs/architecture.md`.
- **Convolution has its own gradient rule** (a `CONV` marker node);
  tinygrad differentiates the operations it is built from.
- **Element types**: `bool`, signed and unsigned integers of 8 to 64
  bits, `float16`, `bfloat16`, `float32`, `float64`. Arithmetic on
  16-bit floats is done in their own type, element by element, as in
  tinygrad; their sums accumulate in `float32` and are converted back,
  as in tinygrad. The fp8 types are absent.
- **`minigrad.linalg`** (`solve`, `det`, `inv`, `cholesky`, `eigh`) has no
  counterpart in tinygrad, whose `Tensor` has `qr` and `svd` built from
  tensor operations. These are eager host routines in float64, not
  lazy or differentiable.
- **Quantization** is `nn.quantize_int8` (symmetric, per output) and a
  multiply after the product; tinygrad's GGUF loader instead
  dequantizes block formats in its graph.

## Engine

- **No code generation.** Kernels run on a block interpreter and a tuned
  matrix-multiply kernel, both compiled ahead of time; tinygrad renders
  and compiles a kernel per fused expression.
- **One view per load**; tinygrad's original ShapeTracker stacked views.
  minigrad copies instead when views do not compose.
- **Reductions always get a buffer**; tinygrad can fuse elementwise work
  after a reduction into the same kernel.
- **CPU only**, for now.
- **The GPT-2 tokenizer is a hand-written matcher** for GPT-2's
  pre-tokenizing regex. It is exact for ASCII and Latin-1 and agrees
  with tiktoken on the test texts (including emoji and CJK), but it
  classifies code points into letters, numbers, and others by block,
  not from Unicode's tables, so rare scripts may split differently.
