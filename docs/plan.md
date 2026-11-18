# Plan

What is not done, roughly in the order it should be, and designs that
were tried or considered and rejected. The goals and the bar for success
are in `~/markdown/tinygrad-mojo.md`.

## Where things stand

Done: the tensor API (creation, arithmetic with broadcasting and type
promotion, comparison, reductions, movement, matmul, convolution,
pooling, normalization, attention, cross-entropy), reverse-mode
gradients of all of it, `SGD`/`Adam`/`AdamW`,
`Linear`/`Conv2d`/`BatchNorm`/`LayerNorm`, the scheduler, the block
executor with flattened dimensions, `gemm` (tiled, matrix-vector, and
batched) on a persistent thread pool, MNIST training end to end, and
GPT-2 inference (97.9% test accuracy in 70 steps,
as tinygrad's `beautiful_mnist`), and Python bindings with tinygrad's
API (`docs/python.md`). CPU only. Numbers are in
`benchmarks/README.md`.

## Next

1. **Finish transformer inference.** GPT-2 runs: tokenizer, safetensors
   weights, key-value cache, greedy and sampled generation, checked
   against tinygrad and tiktoken. Decode is within about 25% of the
   memory-bandwidth floor for float32 weights, and 16-bit or int8
   weights cut decode time by half or more. What remains:
   - **Batched throughput.** `generate_batch` decodes up to `batch`
     sequences of different lengths in lockstep and gives each the
     tokens it gets alone. A step of 8 costs 1.2 to 1.3 steps of 1 with
     8 threads on a quiet machine (`benchmarks/README.md`). Attention
     scales with the batch by nature (each sequence reads its own
     cache); its many tiny products each pay `gemm`'s setup, which
     could be trimmed.
   - **Continuous batching performance.** `Batcher` serves 24 requests
     2.5x (int8) to 4.2x (float32) faster than one at a time with 8
     slots. Stepping every slot (one `MiniJit` capture per 64-position
     block) and stepping only up to the highest occupied slot measured
     the same within the noise; the first was kept as the simpler.
     What remains: prompts are processed one at a time as they are
     admitted (they could share a step); and a slot layout
     that does not reserve `max_context` rows per sequence (pages of
     the cache, as vLLM does) if memory, not time, becomes the limit.
   - **Better quantization.** Weights can be `float16`, `bfloat16`, or
     per-output `int8` (4.3% higher perplexity on Shakespeare). Scales
     per group of 32 to 128 inputs would recover most of that and allow
     4-bit weights, but they vary along the summed dimension and so do
     not factor out of the product: `gemm` would have to apply them as
     it converts B (a scale table beside B, read per block of k).
     Activations stay `float32`; quantizing them (int8 dot products) is
     a different, larger change.
   - **Fused attention** for long prompts: today scores, softmax, and
     the weighted sum are separate kernels and the scores are
     materialized (`T x L` per head).
   - **A larger model** (Llama-style: RoPE, RMSNorm, SwiGLU, grouped
     keys and values) to reach the brief's 1-8B sizes.
   - `gather` as a lazy, differentiable primitive (an index load in the
     executor); `take_rows` is eager and host-side today.
   - Slicing with steps and `__getitem__` with ranges; `cumsum`;
     `multinomial`.
2. **Host work per step** is mostly gone where it matters: schedules
   are cached by structure, and `MiniJit` replays whole steps (a small
   MLP training step 78 to 21 µs; GPT-2 decode saves about 0.9 ms per
   token in the one clean measurement so far, to be confirmed in quiet
   time). Left: steps that cannot be captured (host values that change,
   shapes that change), whose graphs could still be built more cheaply
   (each node allocates its shape, argument, and hash-cons key as
   lists). Replays copy assigned state back into place every step; for a
   large model that copy would want in-place writes into state buffers
   during replay instead.
3. **Fuse elementwise work after a reduction** (`sum * c`, `matmul +
   bias`, `relu` of a convolution) into the same pass as the reduction:
   for `gemm`, apply the tail to each output tile as it is stored; for
   a reduction along the vector dimension, to each block of outputs as
   it completes. A separate in-place pass gained nothing (see Rejected).
4. **GPU.** Mojo compiles kernels for NVIDIA and AMD ahead of time
   (`max.gpu`, and `mojo build --emit` lists `nvptx`), which fits this
   design: the block instruction set and `gemm` become GPU kernels, and
   the plan picks them. Needs a device field on buffers, transfers, and
   an allocator per device.
5. **The benchmark matrix** of the brief: PyTorch (eager and compiled),
   llama.cpp, and MAX as comparators; vision models; dynamic shapes;
   memory.

## Smaller items

- **Python bindings**: building a graph from Python adds about 20 µs
  to an MLP step of about 100 operations (8/32: 76 µs in Mojo, 94 µs
  from Python, 20 µs under `TinyJit`). Batching several operations
  per call would cut
  it. Random functions draw on the host, so a `TinyJit` step must take
  its random indices as inputs; a counter-based generator in the graph
  (tinygrad's Threefry) would lift that. Indexing by tensors beyond
  dimension 0, `cumsum`, `GroupNorm`, `ConvTranspose2d` are missing.

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
- **Eager in-place writes** are how key-value caches work: `write_`
  copies into a buffer the tensor owns, immediately, outside the lazy
  graph. It refuses a buffer shared with another tensor.
- **Creating threads per `gemm` call** (the first version): about 50 µs
  per call, so `gemm` stayed serial below 32M multiply-adds, and a
  GPT-2 decode step makes 49 products. Replaced by the pool. Its
  workers spin for 2 ms after a job before they sleep: waking a
  sleeping thread can take most of a millisecond on this machine.
- **One chunk per thread**: a single thread preempted by another
  process held up every product; with four column chunks per thread,
  claimed as threads come free, it holds up a quarter as much. Kept
  where every chunk would pack all of A (a transposed A): there the
  repeated packing cost more than preemption did (512 / 512 MLP step
  1.0 to 0.9 ms).
- **Packing buffers kept per pool thread** (instead of `gemm`
  allocating them per call): not built, as measurement showed nothing to
  gain. glibc reuses the freed blocks, so steady-state steps fault no
  pages (MLP and convolutional training: none per step; GPT-2 decode:
  about 8 per step, batch 8: none), and a few allocations per product
  are, by estimate (49 products, about 50 ns each), under 1% of a
  decode step. It would need a thread index or thread-local storage in
  the pool.
- **Fusing a reduction's elementwise tail as an in-place second pass**
  (tried 2026-11-17, reverted): a reduction read only by one
  elementwise kernel of its shape and type ran into that kernel's
  buffer, which the kernel then finished in place, saving the
  reduction's buffer. Fusing only when nothing outside the schedule
  could need the reduction later meant the schedule cache's key had to
  record outside references per node. Measured on a quiet machine
  against the unfused build: MLP steps 75.5 / 116 / 719 µs unfused, 77.4
  / 116 / 723 fused (the small one slower from the longer key); the
  convolutional network 10.1 / 91-94 ms, 10.6 / 86-90 fused; GPT-2
  decode unchanged. Still two passes over the output, and the outputs
  are mostly in cache already: nothing to gain. About 200 lines.
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
