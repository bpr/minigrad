# Plan

What is not done, roughly in the order it should be, and designs that
were tried or considered and rejected. The goals and the bar for success
are in `~/markdown/tinygrad-mojo.md`.

## Where things stand

Done: the tensor API (creation, arithmetic with broadcasting and type
promotion, comparison, reductions, movement, matmul, convolution,
pooling, normalization, cross-entropy), reverse-mode gradients of all of
it, `SGD`/`Adam`/`AdamW`, `Linear`/`Conv2d`/`BatchNorm`/`LayerNorm`, the
scheduler, the block executor with flattened dimensions, a threaded
`gemm`, and MNIST training end to end (97.9% test accuracy in 70 steps,
as tinygrad's `beautiful_mnist`). CPU only. Numbers are in
`benchmarks/README.md`.

## Next

1. **Transformer inference**, the brief's headline benchmark. Needs:
   - `gather`/embedding lookup as a primitive (an index load in the
     executor), rather than one-hot times table;
   - slicing with steps and `__getitem__` with ranges; `cumsum`;
     `tril`/`triu`; `scaled_dot_product_attention`;
   - a weight loader (safetensors) and a tokenizer (GPT-2 BPE);
   - a way to append to a KV cache without copying it. This is the one
     place immutable buffers hurt: see "eager in-place writes" below.
   - batched matmul with a shared batch dimension in `gemm` (today it
     takes the block executor): an outer loop over the shared
     dimensions around the contraction.
2. **Fuse elementwise work after a reduction** (`sum * c`, `matmul +
   bias`, `relu` of a convolution) into the reduction's kernel. For
   `gemm` kernels: apply the tail to each tile, or run it as a second
   pass over the output while it is in cache.
3. **A persistent thread pool**, and threads for the block executor.
   `gemm` creates threads per call (about 50 µs each), so it stays
   serial below 32M multiply-adds and scales to about 3.6x on 8 cores at
   1024 cubed. With a pool: parallel packing of B, shared between
   threads, and elementwise kernels over large tensors split by outer
   dimension.
4. **Cache whole schedules.** A training step rebuilds its graph,
   gradient, and kernels every time (about 70 µs for a small MLP); the
   result is always the same list of kernels over different buffers.
   Key the schedule on the graph's structure and replay it (tinygrad's
   `TinyJit`, without the capture step).
5. **GPU.** Mojo compiles kernels for NVIDIA and AMD ahead of time
   (`max.gpu`, and `mojo build --emit` lists `nvptx`), which fits this
   design: the block instruction set and `gemm` become GPU kernels, and
   the plan picks them. Needs a device field on buffers, transfers, and
   an allocator per device.
6. **Python bindings**: a package exposing `Tensor` to Python with
   zero-copy NumPy interchange, so the engine can be driven from
   notebooks and existing tinygrad model code.
7. **The benchmark matrix** of the brief: PyTorch (eager and compiled)
   and MAX as comparators; vision models; dynamic shapes; memory.

## Smaller items

- The gradient of overlapping pooling windows is still a sum of one
  padded term per window offset (`_window_grad`); fine for 3 by 3, slow
  for large windows. Non-overlapping pooling takes the reshape path and
  has no such cost.
- A mark on buffers that were realized from an expression, so
  `backward` can warn when the loss depends on one (today the gradient
  silently stops there, as in tinygrad).

- Reduce the executor's cost per run of a loop (about 100 ns for a
  17-instruction kernel): one dispatch per instruction instead of three
  nested ones, and gather through SIMD for table loads.
- `gemm`: check offset tables for unit runs per panel in more cases
  (convolution packs its windows element by element), and pack with
  SIMD.
- Windows over a padded view copy the padded input. A mask that knows
  about windows, or a `gemm` packer that pads as it packs, would avoid
  the copy.
- `float16`/`bfloat16`; `prod` reduction; `scatter`; `einsum`.
- `mojo format` the tree and wire a doc-string check into `pixi run`.

## Rejected, or tried and dropped

- **In-place assignment.** The first version had an `ASSIGN` node that
  wrote into its target's buffer, with scheduling rules to order readers
  before the write and a temporary when a kernel read its own target out
  of place. It is unsound with hash-consing (equal tensors share a
  buffer) and with lazy readers of the old value, and the safe subset
  (write in place only when nothing else refers to the buffer) is
  exactly the set of cases where rebinding is free. Dropped for
  immutable buffers.
- **Eager in-place writes** are the likely answer for KV caches: an
  explicit, immediately executed "write these elements into that
  buffer", outside the lazy graph, on a buffer the caller created with
  `Tensor.empty` (which is never shared). Not built yet.
- **Reference-counted node objects** (`ArcPointer[Node]` with children):
  Mojo cannot derive the destructor of a type that refers to itself
  through `ArcPointer`, and an arena of ids gives dense traversal state
  and hash-consing besides.
- **A stack of views per load** (tinygrad's ShapeTracker): every load
  being one strided view keeps the executor and `gemm` simple; the cost
  is a copy where views do not compose.
- **Generating C and compiling it at run time** (tinygrad's CPU
  backend): needs a compiler on the user's machine and pays compile
  latency per kernel shape. The block interpreter plus tuned kernels
  has no cold start. Worth revisiting only if a workload appears that
  neither path serves.
- **Building sliding windows from reshape and expand** (tinygrad's
  `_pool`): the result is not one strided view, so it would copy the
  input enlarged by the kernel size.
