# Architecture

minigrad keeps tinygrad's model of computation and changes what it is
made of. This page follows one computation from the user's code to
machine loops and says, at each step, what the design is and why.

```
Tensor ops ──► graph of nodes ──► kernels ──► plans ──► block loops / gemm
   (tensor)     (graph, uop)     (schedule)   (kernel)   (blockops, gemm)
                    ▲
                    └── backward adds gradient nodes (gradient)
```

## The graph

Every operation appends a `Node` to one process-wide arena
(`minigrad/graph.mojo`). A node is an operation code, an element type, a
shape, up to three input ids, and a few integers of argument. Three
properties of the arena carry most of the design.

**Integer ids, dense arrays.** Traversals (`toposort`) mark visits in an
array indexed by node id and stamped with an epoch, and record each
node's position in the order in a second array. The scheduler's per-node
state is then plain lists indexed by position: no hashing in the inner
loops.

**Hash-consing.** `Graph.add` looks the node up by (operation, type,
inputs, argument) before creating it, so building an expression twice
yields one node. A gradient graph repeats large parts of the forward
graph (the ReLU mask, the softmax, every broadcast); they are shared as
they are built, with no separate common-subexpression pass.

**Reference counting.** A node's count is the number of `UOp` handles
and parent nodes that refer to it. When it drops to zero the slot goes
on a free list and its inputs are released, iteratively. A training loop
therefore runs in a bounded arena; `tests/test_nn.mojo` checks that the
node count, buffer count, and plan count are identical at steps 5
and 19.

Mojo structs cannot refer to themselves through `ArcPointer` (the
compiler cannot derive the destructor of a recursive type), which is one
reason the graph is an arena of ids rather than a web of
reference-counted objects. The other reasons are the three above.

### Realization rewrites nodes in place

When a node is computed, `become_buffer` turns that same node into a
`BUFFER` node. Everything that had it as an input now reads memory. There
is no substitution map and no second graph.

### Buffers never change

Nothing writes to a buffer after the kernel that fills it. `Tensor.assign`
points the tensor at a new node. This is a deliberate difference from
tinygrad, and the reason is hash-consing: two tensors built from equal
expressions are one node and so share one buffer, and an in-place write
through one would change the other. (The first version had in-place
assignment; it broke `BatchNorm`, whose `bias` and `running_mean` are both
"three zeros".) The cost is that an optimizer step writes parameters to
fresh memory from the pool rather than over the old values.

## Tensors are references; values are immutable

`Tensor` holds an `ArcPointer[TensorCell]`, and the cell holds the `UOp`.
Copying a `Tensor` copies the pointer, so a parameter stored in a layer
and in an optimizer is one tensor: when the optimizer assigns it a new
value, the layer sees it. A tensor that requires a gradient registers a
*gradient slot* in the graph; `backward` finds the slots whose node the
loss depends on and stores (or accumulates) a gradient node there.

## Gradients are graph transformations

`compute_gradients` (`minigrad/gradient.mojo`) sorts the graph below the
loss, and walking it from the loss down asks each node for the gradients
of its inputs given the gradient of its output. The answers are ordinary
nodes built with the same `UOp` operations as the forward pass, so they
are folded, hash-consed, scheduled, fused, and differentiable again.
Nothing is recorded while the forward pass is built.

### Realizing a node ends its history

Because realization rewrites the node, a realized tensor is a buffer
with no record of how it was computed. `backward` on a realized loss
raises, and a tensor realized in the middle of a computation is a
constant to any gradient taken through it afterwards (tinygrad behaves
the same way there). This is why nothing in a forward pass may realize:
`BatchNorm` leaves its running estimates lazy and lists them for the
optimizer's step to realize, as tinygrad does.

### Parameters have nodes of their own

Hash-consing would make two parameters that start equal (two biases of
zeros) one node, and each would then receive the sum of both gradients;
and a parameter that is a lazy constant would be folded out of whatever
is computed from it. `requires_grad_` therefore gives such a tensor a
`CONTIGUOUS` node, which is never shared.

### Convolution is differentiated as a convolution

The forward pass builds a convolution from windows, a multiply, and a
sum, and differentiating those operations one by one is correct but
wasteful: the gradient with respect to the windows is a tensor the size
of the input times the kernel (650 MB for the second layer of the MNIST
model at batch 512), which is then folded back. So `conv2d` wraps its
result in a `CONV` node that names the input and the weight. The
scheduler looks through it; `gradient` stops at it and builds the
transposed convolution for the input and the direct contraction for the
weight, both of which are again convolutions made of the same
operations. This is the only place a gradient is not derived from the
primitive operations.

## Movement operations are views

Reshape, permute, expand, pad, shrink, flip, and sliding windows never
move data. `View` (`minigrad/view.mojo`) describes how an index maps to a
buffer element: an offset, a stride per dimension, and an optional mask
(indices outside it read as zero, which is how padding costs nothing).
Each movement op maps a view to a view.

Two of them can fail to produce a single view: a reshape that regroups
dimensions a permutation has interleaved, and windows over a masked
view. tinygrad's original answer was a stack of views per load; minigrad
instead materializes the input of the offending op and continues from
the copy. Every load in every kernel is therefore one strided view,
which is what lets the executor walk it with an odometer and lets `gemm`
address it with offset tables.

Sliding windows are a movement op here (`OP_WINDOW`) rather than a
composition of reshapes and expands as in tinygrad, because as one
strided view they are trivially a single view: position `p` and offset
`q` read element `p * stride + q * dilation`. Convolution is then
"windows of the input, times the broadcast weight, summed", as in
tinygrad, and that sum is a contraction.

## Scheduling

`realize` (`minigrad/schedule.mojo`) decides which nodes get buffers and
fuses the rest.

Roots are: the requested nodes; every reduction; `CONTIGUOUS` nodes; an
elementwise node that more than one kernel reads (found by propagating
"which kernel reads me" from parents to inputs; movement nodes pass the
mark through); and both operands of a contraction, so that the kernel is
exactly two loads and a multiply.

A kernel is built by walking down from its root. Elementwise nodes
become instructions. Movement nodes are pushed on a chain; at a buffer
(or another root, which will be a buffer by then) the chain is applied,
innermost first, to the buffer's identity view. The result is an SSA
program whose leaves are `(buffer, view)` loads and constants.

### Schedules are cached by structure

A training loop or a decode loop builds a graph of the same structure
every step: only which buffers sit at the leaves, and so what they hold,
changes. Before scheduling, `realize` computes a signature of the graph:
each node's operation, element type, shape, argument, and inputs, by
position in the topological order, with buffers anonymous. A signature
seen before replays the recorded kernels (each kept as a plan, its
root's position, its inputs' positions, and its program) against the
current graph's nodes, skipping root selection, kernel building, and
plan lookups. The signature includes constants' values, so a step whose
constants change (Adam's bias correction, if it were a number) misses;
Adam keeps that correction in tensors for this reason, as tinygrad does.
Set `MINIGRAD_SCHEDULE_CACHE=0` to schedule every time.

What a hit still costs is building the graph: the user's code runs every
step, creating about 2,500 nodes for a GPT-2 decode step (about 0.65 ms).

### Batches of sequences

`GPT2(..., batch=B)` keeps `B` slots: each layer's caches are
`(B * max_context, n_embd)`, slot `s` at rows `s * max_context ...`, and
the mask is one flat `(B * max_context)` buffer (flat because only a
tensor that owns its buffer can be written in place). `generate_batch`
processes each prompt into its own slot with the single-sequence path,
then `decode_batch` advances every sequence by one token per step, each
at its own position: the new keys and values go to scattered rows by
`put_rows_` (the in-place inverse of `take_rows`, replayable by
`MiniJit`), and each sequence attends to the first `n` rows of its slot
(`n` from the longest sequence, in blocks of 64) with its own row of the
mask hiding the rest.

`Batcher` (`minigrad/llm/batching.mojo`) runs requests continuously on
top of this: each step admits waiting requests to the lowest free slots
(processing their prompts), then advances every slot by one token, so
the step's shape changes only every 64 positions and `MiniJit` captures
it once per block; free slots compute a dummy token at their own
position 1, which the next prompt in that slot overwrites. A request that reaches its token limit, `<|endoftext|>`, or
the end of the context frees its slot. There is no paging and no
copying of caches: a slot is `max_context` rows whether used or not.

The point of batching is that each step reads the weights once for all
sequences. With few rows `gemm` streams the weights in its
matrix-vector path; that path keeps each row's sums in memory and slows
as rows are added, so from 4 rows the tiled kernel takes over.

### Capture and replay

`MiniJit` (`minigrad/jit.mojo`) removes that too, as tinygrad's `TinyJit`
does: it
runs a step function once, runs it again while the runtime records a
tape of everything that touches memory, and from then on replays the
tape without running the function. The tape has three kinds of step: a
kernel (plan, program, input and output buffers), a gather
(`take_rows`), and a copy (`write_`). The function's input tensors are
substituted by buffer; everything else it read is read from the same
buffers again, which the capture holds (`Buffer.held`) so they are not
recycled.

Two things differ from tinygrad because buffers here are immutable.
State such as weights and optimizer moments is updated by `assign`,
which in minigrad points the tensor at a new buffer; a replay would go on
reading the old one. So `assign` records, during a capture, each
tensor's buffer before the step, and every replay copies the new value
back into it at the end (tinygrad's `assign` writes in place, so it
needs no copy). And values that change every call must be tensors: the
eager operations take their indices from tensors (`take_rows(rows:
Tensor)`, `write_(dim, start: Tensor, ...)`), and their host-integer
forms raise during a capture rather than baking a stale value into the
tape. GPT-2's `decode` keeps one `MiniJit` per block of 64 positions,
since attention's shapes change at each block.

## Plans

`make_plan` (`minigrad/runtime/kernel.mojo`) lowers a kernel to loops.
It drops size-1 dimensions, merges adjacent dimensions that every load
walks contiguously (a contiguous elementwise kernel becomes one loop),
and then picks an execution shape by estimated cost:

- **A contraction** goes to `gemm`. The test is structural: a sum of the
  product of two unmasked loads in which every output dimension has
  stride zero in exactly one of them. Dimensions are sorted into I (only
  A varies), J (only B varies), and K (summed); each set becomes an
  offset table, so I, J, and K can each be several dimensions with
  arbitrary strides. A matrix product, a batch of them sharing one
  matrix, the three gradients of a linear layer, and convolution with
  its weight gradient are all this shape.
- **One vector dimension**: the innermost output dimension or a reduced
  one, preferring long dimensions that loads walk with stride 0 or 1.
- **Flattened**: when the best dimension is short (the 2 by 2 window of
  a pooling layer), trailing dimensions are flattened into the vector
  dimension. Loads read through a table of offsets over the flattened
  index, and runs of reduced elements fold into consecutive outputs.

Plans are cached by the kernel's structure. Buffer identities and the
values of constants are not part of the key: the second step of a
training loop builds no plans, and neither does a step whose learning
rate changed.

## Execution

`run_plan` walks the outer dimensions with an odometer and, at each
position, runs the kernel's instructions over the vector dimension one
block (256 elements) at a time. An instruction is one call to a typed
SIMD loop in `blockops.mojo`. A load with stride 1 is not copied: its
register points into the buffer. The last instruction of an unreduced
kernel writes straight into the output.

This is an interpreter with a block as its unit of work, in the manner
of numexpr, rather than a code generator. It has no compile step (the
first step of a training loop costs what the hundredth does, plus plan
building measured in microseconds), needs no C compiler or JIT at run
time, and its loops are memory-bound for long vectors. Its weakness is
short vectors, where the cost per run of the loop dominates; flattening
is the answer to that, and `gemm` is the answer where it matters most.

`gemm` (`minigrad/runtime/gemm.mojo`) is a Goto/BLIS-style kernel: pack
blocks through the offset tables, accumulate 8 by 32 tiles of the output
in vector registers with fused multiply-adds, split the work over the
thread pool (by columns where it can, so B is packed once). On the development machine (Ryzen 7 9800X3D, AVX-512) it runs
at about 290 GFLOPS on one thread, close to the processor's peak.

## Inference

GPT-2 (`minigrad/llm/gpt2.mojo`) uses the same engine; three things
were added for it.

**Two eager operations.** `take_rows` looks up embeddings on the host:
token ids are host integers anyway, and a lookup as one-hot times the
table would be a 50,257-wide matrix product. `write_` copies new keys
and values into each layer's cache, a `(max_context, 768)` tensor the
layer owns. It is the only in-place write in the system, and it sits
outside the lazy graph on purpose: the immutable-buffer argument above
applies to values the graph shares, and a cache made by `Tensor.empty`
is never shared (`write_` checks). Writing realizes the new keys and
values, so a step is realized in a dozen pieces, one per layer.

**Matrix-vector products.** At batch size one, a linear layer's
contraction has no row dimension, so it was not recognized as a
contraction and ran in the block executor at 5 to 9 GB/s. Contractions
with only one side are now contractions, and `gemm` handles up to 16
rows by streaming B once without packing it: a dot product per column
when B's terms are contiguous (the output projection reads `wte`
transposed), a scaled row per term when its columns are (Hugging Face
stores GPT-2's linear layers as `(in, out)`). Columns are split over the
pool. Every weight is then read at about 60 GB/s, this machine's memory
bandwidth: 497 MB of float32 weights per token sets a floor of about 8
ms per token, and a step takes about 10.7.

**Batched products.** Attention multiplies per head: the head dimension
is shared by both operands and the output. Such dimensions are now
*batch* dimensions of a contraction, with their own offset tables, and
the executor runs one `gemm` per batch index, spread over the pool when
each is small. Before this, a 527-token prompt took 3.6 s (attention in
the block executor at 1.4 GFLOPS); after, 0.19 s.

**The thread pool** (`minigrad/runtime/pool.mojo`) replaced threads
created per call, which cost about 50 µs each and made small products
serial. Workers claim tasks with a compare-and-swap on a counter tagged
with the job's generation; between jobs they spin for 2 ms, then sleep
on a futex. Products split into several column chunks per thread, so a
thread the system preempts delays only its chunk.

**Lower-precision weights.** Decode reads every weight once per token,
so storing them in fewer bytes is the largest lever left. There is no
quantized-matmul kernel and no weight format: a weight stored as
`float16`, `bfloat16`, or `int8` is an ordinary tensor, and `x.dot(w)`
promotes `w` to `float32` with a `CAST` node. The scheduler recognizes a
contraction operand that is a cast of a buffer and does not materialize
the cast; the planner accepts `load, cast, mul` where it accepted
`load, mul`, puts the converted operand in B, and records its stored
type; `gemm`, given a second type parameter for B, converts as it packs
B (tiled path) or streams it (matrix-vector path). The arithmetic is
float32 throughout. For `int8`, `quantize_int8` keeps one scale per
output, which factors out of the sum: `x.dot(q) * scale`, a contraction
followed by an elementwise multiply. Scales that vary along the summed
dimension (group-wise quantization, as llama.cpp uses for 4-bit
weights) do not factor out and would need more than this.

A decode step attends to the cache's first rows rounded up to a
multiple of 64, with a mask buffer (0 for rows written, -inf after) that
`write_` keeps current. Within each block of 64 positions every step
builds a graph of the same shapes, so the schedule cache replays it, and
plans exist only per block. tinygrad masks the whole cache instead; here
that read 72 MB of cache per token and cost more than scheduling saved.

## What to read first

`minigrad/view.mojo`, then `minigrad/graph.mojo` and `uop.mojo`, then
`schedule.mojo`, then `runtime/kernel.mojo`. Set `MINIGRAD_DEBUG=2` to
watch kernels run.
