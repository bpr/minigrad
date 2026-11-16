# AGENTS.md

Project context for AI coding assistants working on minigrad.
Tool-specific workflow instructions live in each tool's own file, e.g.
CLAUDE.md.

## What this is

minigrad is a tensor library with automatic differentiation, written in
Mojo and modeled on [tinygrad](https://github.com/tinygrad/tinygrad). It
is a new implementation with compatibility goals, not a line-by-line
port: it keeps tinygrad's mental model (a small tensor API, lazy
evaluation, an inspectable graph, gradients as graph transformations,
aggressive fusion) and replaces the implementation underneath.

- tinygrad source (the reference for API names and semantics):
  `~/src/python/tinygrad`
- The brief this project follows, including what would count as
  success: `~/markdown/tinygrad-mojo.md`
- `docs/architecture.md`: how the pieces fit, and why.
- `docs/plan.md`: what is not done yet, in rough order, and designs that
  were tried and rejected. Check it before proposing a design change.
- `docs/differences-from-tinygrad.md`: every deliberate deviation.
- `benchmarks/README.md`: the numbers, and how to reproduce them.

## Commands

All tooling runs through pixi (linux-64 only):

```sh
pixi install                 # Mojo 1.3 nightly
pixi run test                # every tests/test_*.mojo
pixi run test tests/test_view.mojo   # one file
pixi run format              # mojo format
pixi run package             # build/minigrad.mojoc
```

Run a program against the source tree with `-I .`:

```sh
pixi run mojo run -I . examples/mnist.mojo
pixi run mojo build -I . -o build/mlp benchmarks/mlp_step.mojo
MINIGRAD_DEBUG=2 build/mlp   # print every kernel run, with its time
MINIGRAD_THREADS=1 build/mlp # cap the threads of the matmul kernel
```

Tests import `tests/helpers.mojo`, so they need `-I tests` as well
(`tests/run_all.sh` passes it).

## Tech stack

- Mojo 1.3 (nightly), managed with pixi. No other dependencies: no MAX,
  no Python, no C compiler at run time.
- Stdlib sources: `~/src/mojo/repos/modular/Mojo/stdlib/std`
- Language docs and changelog: `~/src/mojo/repos/modular/Mojo/docs`
  (`docs/site/reference/cheat-sheets/` is the fastest way in; check
  these when unsure of an API, the language changes monthly).

## Project layout

| Path | Contents |
|---|---|
| `minigrad/dtype.mojo` | `DT`, the element types, and their promotion lattice |
| `minigrad/view.mojo` | `View`: strided, masked views; every movement op as a view-to-view map |
| `minigrad/graph.mojo` | The node arena: hash-consing, reference counts, buffers and their pool |
| `minigrad/uop.mojo` | `UOp`, a counted handle to a node, and the node-building operations (shape checks, constant folding) |
| `minigrad/gradient.mojo` | Reverse-mode differentiation as a graph transformation |
| `minigrad/schedule.mojo` | Graph to kernels: which nodes get buffers, fusion, execution order |
| `minigrad/runtime/kernel.mojo` | `Kernel`, `Plan`, `make_plan` (lowering), `run_plan` (the executor) |
| `minigrad/runtime/blockops.mojo` | Typed SIMD loops over one block: the executor's instruction set |
| `minigrad/runtime/gemm.mojo` | The tuned matrix-multiply kernel (also runs convolutions), and its matrix-vector path |
| `minigrad/runtime/pool.mojo` | The persistent thread pool: `run_parallel[task](ctx, n)` |
| `minigrad/tensor.mojo` | `Tensor`: the user API |
| `minigrad/nn/` | Layers (`Linear`, `Conv2d`, `BatchNorm`, `LayerNorm`) and optimizers (`SGD`, `Adam`, `AdamW`) |
| `minigrad/datasets.mojo` | MNIST loader |
| `minigrad/nn/__init__.mojo` (end) | `quantize_int8`: symmetric int8 with a scale per output |
| `minigrad/jit.mojo` | `TinyJit`: capture a step function's kernels and eager operations, then replay them |
| `minigrad/linalg.mojo` | Eager dense linear algebra: `solve`, `det`, `inv`, `cholesky`, `eigh` (copied from MojoDEC) |
| `minigrad/json.mojo`, `minigrad/safetensors.mojo` | A minimal JSON reader; reading safetensors weights |
| `minigrad/llm/` | `bpe.mojo` (GPT-2's tokenizer), `gpt2.mojo` (the model, with a key-value cache) |
| `tests/` | One `test_*.mojo` per area; `helpers.mojo` has `assert_close` |
| `examples/` | `mnist.mojo`, `gpt2.mojo` (need `examples/get_mnist.sh`, `examples/get_gpt2.sh` first) |
| `benchmarks/` | Mojo programs and their tinygrad counterparts in `benchmarks/tinygrad/` |

## Architecture in one page

See `docs/architecture.md` for the reasoning. The path of a computation:

1. **Tensor ops build nodes.** `Tensor` wraps an `ArcPointer[TensorCell]`
   (copies of a tensor are the same tensor, like Python references); the
   cell holds a `UOp`, a counted handle to a node id in the global
   `Graph` arena. Nodes are hash-consed, so equal expressions are one
   node. Builders fold constants and identities (`x * 1`, no-op
   movements) as they go.
2. **`backward` builds more nodes.** `compute_gradients` walks the graph
   from the loss down, emitting gradient nodes per op. Gradients are
   stored per tensor in the graph's `grad_slots` (a tensor that requires
   a gradient registers a slot).
3. **`realize` schedules.** `schedule.mojo` sorts the unrealized graph,
   picks kernel roots (requested nodes, reductions, `CONTIGUOUS`, nodes
   read by more than one kernel, operands of contractions), and builds a
   `Kernel` per root: an SSA program of elementwise instructions over
   *loads*. Movement ops never become instructions: they are collected
   on the way down and applied to the identity `View` of the buffer at
   the bottom, so a load is one strided view. If a reshape (or a window
   over a padded view) cannot be expressed as one view, the node below
   it becomes a root and scheduling restarts.
4. **Kernels are lowered to plans, once.** `make_plan` turns a kernel
   into loops; plans are cached by the kernel's structure (not its
   buffers, not its constants' values), so a training loop builds each
   plan once. Three execution shapes, chosen per kernel:
   - *contraction* (`sum_k A[b,i,k] B[b,k,j]`, which covers matmul, its
     gradients, convolution, and attention): offset tables for
     `gemm.mojo`, one product per batch index `b`. Up to 16 rows, `gemm`
     streams B once (matrix-vector); otherwise it tiles;
   - *one vector dimension*: an odometer over the other dimensions, and
     blocks of 256 along the vector one;
   - *flattened*: when the best single dimension is short, trailing
     dimensions are flattened into the vector one through offset tables,
     with segmented reduction.
5. **The executor runs blocks.** Each instruction is one typed SIMD loop
   over a block (`blockops.mojo`); temporaries stay in L1.
6. **Roots become buffers in place.** `Graph.become_buffer` rewrites the
   root node into a `BUFFER` node; nodes that read it need no patching.

Design points that span several files:

- **Buffers are immutable.** There is no in-place assignment in the
  engine; `Tensor.assign` points the tensor at a new node. This is what
  makes hash-consing sound (two tensors with equal values may share a
  buffer). It was not always so: see `docs/plan.md`, rejected designs.
- **Realizing a node ends its history.** A realized node is a buffer;
  gradients do not flow through it. Nothing in a forward pass may
  realize (`BatchNorm` once did, and trained to 89% instead of 98%).
  State a layer updates is left lazy and listed in `parameters()` with
  `requires_grad_(False)`; the optimizer's step realizes it.
- **Parameters have nodes of their own.** `requires_grad_` wraps a
  constant or already-claimed node in `CONTIGUOUS` (never hash-consed),
  so equal-valued parameters get separate gradients and constants are
  not folded away.
- **`conv2d` marks its result with a `CONV` node** so `gradient` can
  build the transposed convolution instead of differentiating through
  the windows. The scheduler looks through the marker.
- **Reductions are always kernel roots**, so elementwise work after a
  reduction is a second kernel. Fusing it is in the plan.
- **Padding before a convolution is a copy.** `View.window` refuses a
  masked view, so the padded input is materialized first.
- **Constants live in the graph as `CONST` reshaped and expanded**; a
  padded constant is read from a one-element buffer through a masked
  view.
- **Reference counts**: a node's `rc` counts `UOp` handles and parent
  nodes. Gradient slots hold a raw id with a manual `incref`/`decref`.
- **Two eager operations sit outside the graph**: `take_rows` (an
  embedding lookup, done on the host) and `write_` (the one in-place
  write, for key-value caches; the target must own its buffer). Views of
  a written tensor see the writes.
- **Weights in lower precision are a cast, not a format.** A
  contraction operand may be a cast of a buffer (`float16`, `bfloat16`,
  `int8` into a `float32` kernel): the scheduler leaves it unmaterialized,
  the planner makes it `gemm`'s B, and `gemm[dt, bdt]` converts B as it
  packs or streams it. `x.dot(w)` with a 16-bit `w` gets this from type
  promotion; int8 is `x.dot(q) * scale`. GPT-2's `Weight` struct wraps
  it. The test for exactness: stored natively must equal float32
  weights rounded the same way (`test_gpt2_perplexity`).
- **Schedules are cached by the graph's structure** (`realize`,
  `_signature`): a hit replays recorded kernels against the current
  nodes. Keep step graphs structurally stable: values that change every
  step belong in tensors (Adam's `b1_t`), not constants, and shapes in
  decode change only every 64 positions (`DECODE_ROWS`).
  `MINIGRAD_SCHEDULE_CACHE=0` turns the cache off for comparison.
- **`minigrad/linalg.mojo` is a copy of MojoDEC's
  `mojodec/routines/dense.mojo`** (pivoted LU, Cholesky, `tred2`/`tql2`
  `eigh`), on flat lists instead of NuMojo arrays. Fix bugs in both. The
  user chose copying over a shared library for now and will revisit if
  duplication spreads across projects.
- **`TinyJit` replays a tape** recorded in `runtime().tape` while
  `recording` is set: kernels (from `schedule._run`), gathers, and copies
  (`_record` in `tensor.mojo`), by buffer id. Captured buffers are held
  (`Graph.hold`; `write_`'s ownership check subtracts holds). `assign`
  records state during a capture (`jit_state()`), copied back after each
  replay. Anything new that touches memory outside a kernel must record
  itself, or replays silently skip it.
- **Threads come from one pool** (`runtime/pool.mojo`), used by `gemm`
  and batched products. Jobs are split into several chunks per thread
  where that costs nothing, so one preempted thread holds up only a
  small share.

## Mojo traps this project has hit

Each of these cost a debugging session.

- **Handles die at their last use.** Mojo destroys a value right after
  its last use, not at the end of the scope. `f(x.id)` can therefore
  free node `x` before `f` runs. Pass `UOp` handles, never bare ids, to
  anything that creates nodes (`UOp._make1(..., a)` exists for this).
  The same rule makes "at the end of the test, nothing leaked" checks
  see zero live nodes: sample inside the loop.
- **`x.id` is not a use of `x` either.** In `f(g.buffer_ptr(a.id),
  g.buffer_ptr(b.id))` the handle `a` can be destroyed (freeing its node)
  right after `a.id` is read; `write_` crashed this way. End such code
  with `_ = a^` to keep the handle to that point.
- **An address passed as an `Int` does not keep its value alive.** In
  `run_parallel[t](Int(jobs.unsafe_ptr()), n)` the conversion is the
  last use of `jobs`, so the list may be freed (or, for a stack struct,
  its stores optimized away) before the tasks read it. Write
  `_ = jobs^` after the call. It cost a crash in release builds only
  and silently wrong results (zeros) in another case.
- **`mojo run` unmaps the program's code at exit.** Threads still
  running it then crash the process after `main` returns. The pool is
  registered with `_get_global`'s destroy hook, which stops and joins
  its threads first; any new long-lived thread needs the same.
- `fn` is still a keyword: it cannot name a parameter or a field.
- **References into the arena dangle.** `graph()[].nodes[i].shape` is a
  reference into a `List` that reallocates when a node is added. Copy
  (`x.shape()`) before calling anything that builds nodes. Passing such a
  reference to `View` methods is fine: they do not touch the graph.
- **Offset tables can be negative** (a flipped view walks backward), so
  -1 is not a usable "masked" marker: use `MASKED`.
- `out` is a keyword (an argument convention): it cannot name an
  argument.
- `for x in list` needs `Copyable` elements; iterate by index over
  `List[Kernel]` and friends, and bind with `ref`.
- t-strings take no format specifiers (`{x:>10}` is an error).
- A generic function that calls `exp`, `log`, ... needs evidence that
  its dtype is a float: `where dt.is_floating_point()`, or a
  `comptime if dt.is_floating_point():` branch.
- Runtime dispatch to a function generic over `DType` is a
  `comptime for c in range(DT.COUNT): if dt.code == c: f[mojo_dtype(c)](...)`.
- Converting pointer origins is painful; when a pointer must cross a
  thread or a struct boundary, pass its address as an `Int` and rebuild
  it with `Pointer[T, MutUntrackedOrigin](unsafe_from_address=...)`.
- `Pointer` arithmetic is `p.unsafe_offset(i)` and `p[unsafe_offset=i]`;
  the `+` and positional forms are deprecated.
- Mojo has no global variables. The graph and the runtime are
  `std.ffi._Global` singletons (`graph()`, `runtime()`), about 5 ns per
  access: fetch the pointer once in a hot function.
- `TestSuite`'s per-test timings are not reliable (it reported 325 s for
  a test file that ran in 0.4 s); use `time` on a built binary or
  `perf_counter_ns` when measuring.
- **Benchmarks on this machine are noisy.** The CPU governor is
  `powersave` (idle cores sit at 600 MHz and a lone thread can land on
  one), and other sessions compile in bursts. Take numbers only when
  `/proc/loadavg` is below 1, report medians over many steps, and
  distrust a single-threaded figure with a large gap to its minimum.
- Integer division by zero in a SIMD loop traps the process; the integer
  division loops are scalar and exact for that reason.

## Coding standards

- Use American spelling everywhere.
- Return a small struct with named fields rather than a tuple.
- Public symbols get Google-style doc strings (`Args:`, `Returns:`,
  `Raises:`). Comments say what the code does or why, not how it came
  to be.
- Follow tinygrad's names and semantics in the `Tensor` API unless
  `docs/differences-from-tinygrad.md` records a reason not to; record
  each new deviation there.
- Every execution path in the executor needs a test that reaches it with
  sizes past its boundaries (`tests/test_engine.mojo`); small tensors
  only exercise the trivial path.

## Testing

- Each test file ends with
  `TestSuite.discover_tests[__functions_in_module()]().run()`.
- `tests/test_view.mojo` checks the view algebra against a dense table
  of source indices over random chains of movement ops. Extend the
  reference when adding a movement op.
- `tests/test_gradient.mojo` and `tests/test_conv.mojo` check gradients
  against central finite differences in float64 (`check_grad`). A new
  differentiable op gets a case there.
- `tests/test_engine.mojo` compares the executor with plain loops on
  sizes that cross block, tile, and thread boundaries.
- `tests/test_nn.mojo` includes the loop invariants: a training loop
  must not grow the graph, the buffer table, or the plan cache.
- `tests/test_llm.mojo` covers JSON, safetensors, `write_`/`take_rows`,
  attention against a direct computation, and the pool; with
  `data/gpt2` present, the tokenizer against tiktoken's ids and GPT-2's
  greedy output against tinygrad's.
- Larger checks against references live in the benchmark harness:
  `benchmarks/gpt2_bench.mojo` and `benchmarks/tinygrad/gpt2_bench.py`
  have matching `logits` and `greedy` actions (see
  `benchmarks/README.md`).

## Working alongside other sessions

Other sessions run Mojo builds and tests in the sibling projects
(`../MojoDEC`, `../proptest`) at the same time. Never kill processes by
name or pattern (`pkill -f mojo`, `kill $(pgrep ...)`): kill only a PID
this session started, and run anything that might hang under `timeout`.
