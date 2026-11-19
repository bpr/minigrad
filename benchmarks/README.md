# Benchmarks

Every number here was measured on one machine and is a statement about
that machine: an Intel Core i7-12700H laptop (6 performance and 8
efficiency cores, 20 threads, AVX2, no AVX-512), Linux, Mojo
1.3.0.dev2026111316. tinygrad is at commit `b0e391d72`, on its CPU
backend with the LLVM renderer (`DEV=CPU:LLVM`: the laptop has no
`clang` binary), under Python 3.12. minigrad's Python bindings run on
the pixi environment's Python 3.15. Times are medians over the steps of
one run after the first; where two runs were taken, both are given.

The numbers until commit `027551d` came from an AMD Ryzen 7 9800X3D
(8 cores, AVX-512), which is now too busy to measure on; see that
commit's version of this file. That machine ran minigrad's kernels
about twice as fast, with ratios over tinygrad of the same order.

**What these do and do not show.** They compare minigrad on CPU with
tinygrad on CPU: GPT-2 (124M) inference and training two small models.
tinygrad's CPU backend is not where its effort goes, and there is no
GPU backend here yet, so read the ratios as "against tinygrad as it
runs on a CPU", not as a verdict on the rewrite. No PyTorch, llama.cpp,
or MAX baselines yet.

**Noise.** The CPU governor is `powersave`. Each group of runs started
at a load average below 1; minigrad's own threads then raise it to 2 to
9, so the "load below 1" rule can only be checked at the start.
Multi-thread medians repeat to within about 10% except where a range is
given; a step with few threads' worth of work can land on efficiency
cores, which shows as a wide gap between median and minimum.

**Power limit.** Under load all cores the package boosts to about
3.5 GHz for roughly ten seconds, then holds 2 to 2.8 GHz (at 74 to 77
°C, so a power limit rather than heat). A long run is 25-30% slower than
a short one, and of two runs back to back the second is penalized. To
compare two builds, use short runs (a few seconds), pause about 30 s
between them, and alternate which goes first. Runs of 20 s or more,
such as MNIST start to finish, mostly run at the sustained clock.

## Reproducing

```sh
pixi run mojo build -I . -o build/mlp_step benchmarks/mlp_step.mojo
pixi run mojo build -I . -o build/convnet_step benchmarks/convnet_step.mojo
build/mlp_step 64 128 3000          # batch, hidden units, steps [jit]
build/convnet_step 512 20           # batch, steps

cd ~/src/python/tinygrad            # a tinygrad checkout
PYTHONPATH=. DEV=CPU:LLVM python3 <minigrad>/benchmarks/tinygrad/mlp_step.py 64 128 300 jit
PYTHONPATH=. DEV=CPU:LLVM python3 <minigrad>/benchmarks/tinygrad/convnet_step.py 512 8 jit
```

Drop the trailing `jit` to run tinygrad without `TinyJit`; add it to
minigrad's `mlp_step` to run under `MiniJit`. `MINIGRAD_THREADS=1`
limits minigrad to one thread; `MINIGRAD_DEBUG=2` prints every kernel
with its time.

## GPT-2 (124M) inference

`benchmarks/gpt2_bench.mojo` against `benchmarks/tinygrad/gpt2_bench.py`,
which runs tinygrad's own `examples/gpt2.py` model on the same
safetensors weights (decode under `TinyJit`, as tinygrad's example
does). float32 weights in both, batch of one, greedy decoding, 100
decode steps after the prompt. The 527-token prompt is the first 527
tokens of Tiny Shakespeare.

```sh
bash examples/get_gpt2.sh
pixi run mojo build -I . -o build/gpt2_bench benchmarks/gpt2_bench.mojo
IDS=$(build/gpt2_bench encode "What is the answer to life, the universe, and everything?")
LONG=$(build/gpt2_bench encode "$(head -c 3000 data/gpt2/tinyshakespeare.txt)" | tr ' ' '\n' | head -527 | tr '\n' ' ')
build/gpt2_bench bench 100 $IDS
cd ~/src/python/tinygrad
PYTHONPATH=. DEV=CPU:LLVM python3 <minigrad>/benchmarks/tinygrad/gpt2_bench.py \
    <minigrad>/data/gpt2/model.safetensors bench 100 $IDS
```

| | minigrad | tinygrad | ratio |
|---|---:|---:|---:|
| Prompt of 13 tokens, first run | 23.5 ms | 2,318 ms | 99x |
| Prompt of 13 tokens, again | 20.5 ms | 1,138 ms | 56x |
| Decode, context 14 to 113: per token | 15.3 ms | 178.6 ms | 12x |
| Prompt of 527 tokens, first run | 252 to 270 ms | 35,428 ms | 131x to 141x |
| Prompt of 527 tokens, again | 250 to 261 ms | 33,924 ms | 130x to 136x |
| Decode, context 528 to 627: per token | 15.8 ms | 174.6 ms | 11x |
| A request: 13-token prompt, 100 new tokens | about 1.6 s | about 20.8 s | 13x |

The last row adds the components (prompt, first decode step, 99 more
steps); it is not separately timed. On one thread, minigrad decodes at
22.2 ms per token.

**Where a decode step goes.** Decode runs through `MiniJit`: after the
first steps of each block of 64 positions, a step replays the recorded
kernels without building or scheduling a graph. It reads all 497 MB of
weights once per token. Of the 15.3 ms:

| | ms per token |
|---|---:|
| Running 309 kernels | 15.2 |
| Host work (graph building on capture steps, sampling) | 0.17 |
| Choosing kernels and building them | 0.01 |

The kernels read the weights at about 33 GB/s. An elementwise pass over
256 MB on this laptop moves data at 30 to 31 GB/s (read plus write), so
float32 decode is at about the speed memory allows here; narrower
weights are the lever (below).

The ratio over tinygrad is large because tinygrad's CPU backend runs
these matrix-vector products far below memory bandwidth, not because
minigrad does anything a CPU cannot. For the 527-token prompt, tinygrad
does not use `TinyJit` (its example only JITs single-token steps) and
spends most of the 35 s in attention; read that row as "tinygrad's CPU
path for long prompts is slow", not as a 130x better design. The
brief's warning applies: a ratio over 100x means the baseline is on a
slow path.

### Schedule cache

Graphs of a structure scheduled before replay the cached schedule.
Since decode moved to `MiniJit` it matters only on capture steps:
`MINIGRAD_SCHEDULE_CACHE=0` raises kernel building from 1 to 27-29 µs
per decode step on average, and decode from 15.29 to 15.37 ms (float32)
and 6.74 to 6.87 ms (int8). Training steps that are not captured still
rely on it: the MLP step below builds no kernels after the first.

### MiniJit

A replay runs the recorded kernels without building, scheduling, or
planning a graph. MLP training steps, two runs each:

| batch / hidden | plain | `MiniJit` | tinygrad, `TinyJit` |
|---|---:|---:|---:|
| 8 / 32 | 149 to 150 µs | 69 to 70 µs | 1,104 µs |
| 64 / 128 | 331 to 450 µs | 381 to 388 µs | 1,716 µs |
| 512 / 512 | 2,555 to 2,584 µs | 2,633 to 2,709 µs | 28,814 µs |

The final losses with and without the JIT are identical. The first row
is host work, which the JIT removes (its minimum is 28 µs: the median
is a replay landing on efficiency cores). The other two are kernel time,
which the JIT does not change.

### Batched generation

`build/gpt2_bench batch B N IDS...` decodes `B` sequences of different
lengths at once; each sequence gets the greedy tokens it gets alone.
One decode step for every sequence of a batch, median of 60 steps, one
process per batch size, prompts from the first 32 tokens of Tiny
Shakespeare:

| weights | batch 1 | batch 8 | batch 16 |
|---|---:|---:|---:|
| int8, step | 6.95 ms | 12.07 ms | 22.98 ms |
| int8, tokens/s | 144 | 663 | 696 |
| float32, step | 15.85 ms | 22.25 ms | 30.00 ms |
| float32, tokens/s | 63 | 359 | 533 |

A step of 8 costs 1.4 (float32) to 1.7 (int8) steps of 1: with one
sequence a product is limited by reading the weights, and the other
seven sequences' arithmetic is spread over the cores. With int8 weights
there is less reading to hide the arithmetic behind. Attention grows
with the batch by nature, since each sequence reads its own cache.

### Continuous batching

`build/gpt2_bench serve S R IDS...` serves `R` requests (prompts of 3
or more tokens, 16 to 63 new tokens each, all arriving at once) with a
`Batcher` of `S` slots, then one at a time, and prints throughput and
mean completion time for both. 8 slots, 24 requests (948 tokens):

| weights | `Batcher`, 8 slots | one at a time | gain |
|---|---:|---:|---:|
| int8 | 377 tokens/s | 123 tokens/s | 3.1x |
| float32 | 235 tokens/s | 62 tokens/s | 3.8x |

The gain is below the batch's 8x because each admission processes its
prompt alone, and the batch drains as the longest requests finish.

### Lower-precision weights

The token embedding and every layer's matrices stored narrower (biases,
normalizations, and the position embedding stay float32; arithmetic is
float32 throughout). Perplexity is over the first 8 windows of 512
tokens of Tiny Shakespeare; decode is the median of 100 steps after the
13-token prompt.

```sh
build/gpt2_bench --dtype bfloat16 ppl data/gpt2/tinyshakespeare.txt 8
build/gpt2_bench --dtype bfloat16 bench 100 $IDS
```

| weights | size | perplexity | greedy, 30 tokens | decode | of which kernels |
|---|---:|---:|---|---:|---:|
| float32 | 497 MB | 94.79 | reference | 15.3 ms | 15.2 ms |
| float16 | 249 MB | 94.88 | identical | 8.2 ms | 8.1 ms |
| bfloat16 | 249 MB | 92.41 | identical | 9.1 ms | 8.9 ms |
| int8, a scale per output | 124 MB | 98.87 | differs from token 7 | 6.7 ms | 6.6 ms |

bfloat16 lowering perplexity is not an improvement in the model: a
float32 model with the same rounded weights gives the same number to
every printed digit, so it is what rounding GPT-2 does to its fit on
this text, which is unlike what it was trained on. int8 decode reads a
quarter of float32's bytes but takes 44% of its time; where the rest
goes has not been measured on this machine.

**Correctness.** For a 28-token prompt, minigrad's logits agree with
tinygrad's to 2.2e-4 (tinygrad's values printed to 4 decimals; logits
range from -93 to -59), with the same top five tokens. 30 greedy tokens
agree exactly (checked again on this machine, for float32, float16, and
bfloat16 weights). minigrad's tokenizer gives tiktoken's ids on 18 test
texts covering contractions, whitespace runs, numbers, code, URLs,
accented Latin, typographic punctuation, emoji, CJK, and the special
token. To repeat the checks: the `logits` and `greedy` actions of both
scripts, and `tests/test_llm.mojo`.

## A two-layer MLP: one training step

784 inputs, one hidden layer with ReLU, 10 outputs; cross-entropy; SGD
with momentum; random data. A step is forward, backward, the update, and
reading the loss. Two runs of minigrad:

| batch / hidden | minigrad | tinygrad, `TinyJit` | tinygrad, no JIT | vs JIT | vs no JIT |
|---|---:|---:|---:|---:|---:|
| 8 / 32 | 149 to 150 µs | 1,104 µs | 15,320 µs | 7.4x | 103x |
| 64 / 128 | 331 to 450 µs | 1,716 µs | 15,754 µs | 3.8x to 5.2x | 35x to 48x |
| 512 / 512 | 2,555 to 2,584 µs | 28,814 µs | 42,521 µs | 11x | 17x |

With one thread minigrad's larger two steps take 548 µs and 8,466 µs.

The first row is almost all host work, and is the comparison the brief
cares about for small and dynamic workloads. minigrad rebuilds the
graph, differentiates it, and schedules it on every step; `TinyJit`
replays a recorded schedule and does none of that. Where minigrad's
step went:

| | µs per step |
|---|---:|
| Building the graph (forward, gradient, update) | 62 |
| Choosing kernel roots (a schedule cache hit) | 8 |
| Building kernels | 0 |
| Running 29 kernels | 82 |

Plans are built once: 33 over the whole run, none after the first step.

### From Python

`benchmarks/mlp_step.py` is tinygrad's `mlp_step.py` with minigrad's
imports, on the Python bindings (`docs/python.md`). Medians over the
second half of 3,000 steps (300 for 512/512), two runs each:

| batch / hidden | Mojo | Python | Python, `TinyJit` | tinygrad, `TinyJit` |
|---|---:|---:|---:|---:|
| 8 / 32 | 149 to 150 µs | 175 to 180 µs | 77 µs | 1,104 µs |
| 64 / 128 | 331 to 450 µs | 535 to 560 µs | 285 to 396 µs | 1,716 µs |
| 512 / 512 | 2,555 to 2,584 µs | 2,653 to 2,772 µs | 2,583 to 2,708 µs | 28,814 µs |

Without the JIT, Python adds about 28 µs to a step of about 100
operations: each is one call into the engine. Under `TinyJit` a replay
runs no Python and matches Mojo's `MiniJit`.

```sh
pixi run python-build
PYTHONPATH=python pixi run python benchmarks/mlp_step.py 8 32 3000 jit
STEPS=9 PYTHONPATH=python pixi run python examples/beautiful_mnist.py
```

## The MNIST convolutional network: one training step

tinygrad's `beautiful_mnist` model (four convolutions, two batch
normalizations, two max pools, a linear layer), Adam, random data. Two
runs of minigrad:

| batch | minigrad | tinygrad, `TinyJit` | tinygrad, no JIT | vs JIT |
|---|---:|---:|---:|---:|
| 64 | 21.8 to 23.1 ms | 300.0 ms | 359.1 ms | 13x to 14x |
| 512 | 188 to 190 ms | 2,426 ms | (not run) | 13x |

At batch 512 with one thread minigrad takes 699 ms (3.7x). Reductions
over the batch and spatial dimensions (batch normalization's statistics,
convolution bias gradients) are split over threads by channel; before
that they ran on one thread and the step took 199 to 204 ms at batch
512.

First step, which for tinygrad includes generating and compiling its
kernels and for minigrad includes building 110 plans:

| batch | minigrad | tinygrad, `TinyJit` |
|---|---:|---:|
| 64 | 47 to 61 ms | 2,425 ms |
| 512 | 231 to 237 ms | 4,964 ms |

## MNIST, start to finish

70 steps at batch 512, test accuracy on all 10,000 test images every 10
steps, including startup and loading the data: `examples/mnist.mojo`,
`examples/beautiful_mnist.py` on minigrad's Python bindings, and
tinygrad's own `examples/beautiful_mnist.py`.

| | wall time | final test accuracy |
|---|---:|---:|
| minigrad, Mojo | 20.2 to 21.6 s | 97.89% |
| minigrad, Python | 19.3 to 19.4 s | 97.90% |
| tinygrad (`DEV=CPU:LLVM`, `STEPS=70`) | 221 s | 98.09% |

A training step takes about 200 ms from Mojo and from Python alike; a
step that evaluates the test set takes about 1 s more. Python's peak
memory is 3.6 GB, mostly from evaluating the test set in one batch, as
tinygrad's file does. tinygrad's time includes downloading the test set
(about a second). The runs draw different random batches and initial
weights, so the accuracies agree only as well as different seeds would.

## The matrix-multiply kernel

`minigrad/runtime/gemm.mojo` alone, float32, best of 21 runs
(`benchmarks/gemm_bench.mojo`):

| A x B | threads | time | GFLOPS |
|---|---:|---:|---:|
| 64 x 784 times 784 x 128 | 1 | 289 µs | 45 |
| 512 x 784 times 784 x 512 | 1 | 3.31 ms | 124 |
| 512 x 784 times 784 x 512 | 14 | 0.58 ms | 706 |
| 512 x 784 times (512 x 784) transposed | 14 | 0.62 ms | 662 |
| 1024 x 1024 times 1024 x 1024 | 1 | 17.5 ms | 123 |
| 1024 x 1024 times 1024 x 1024 | 14 | 2.66 ms | 807 |

One performance core peaks at about 144 GFLOPS (two AVX2 fused
multiply-adds per cycle, 8 lanes, about 4.5 GHz); `gemm` reaches 85% of
that on one thread. On AVX2 its tiles are 4 rows high, not 8 as with
AVX-512: 8 rows need more than AVX2's 16 vector registers, and ran at
60% of the speed. The pool uses one thread per physical core (14); the
efficiency cores are slower, so 14 threads reach 5.7x to 6.6x one. The
smallest product is mostly packing and thread overhead.

## Against the brief's bar

`~/markdown/tinygrad-mojo.md` sets seven criteria. Where this stands:

| Criterion | Status |
|---|---|
| Runs a meaningful subset correctly, with backward and optimizers | Yes for MLPs, convnets, and GPT-2 inference: 130 Mojo tests and 24 Python tests, gradients checked against finite differences, MNIST to 97.9%, GPT-2 matching tinygrad's output |
| Usable from Python without writing Mojo | Yes: tinygrad's `Tensor` API, `TinyJit`, layers and optimizers (`docs/python.md`); tinygrad's `beautiful_mnist.py` runs with changed imports, as fast as the Mojo example |
| At least 5x on transformer inference | 12x on GPT-2 decode, 13x for a whole short request, against tinygrad on CPU; float32 decode is at about this laptop's measured memory bandwidth. Batches of 8 sequences decode at 1.4-1.7x the cost of one. No GPU or larger models yet |
| At least 2-3x on a training workload | 3.8x to 14x over tinygrad's JIT on CPU, on these two models; 10x to 11x for MNIST start to finish |
| At least 10x less graph, scheduling, and dispatch overhead | 103x against tinygrad without JIT at the smallest size; 7.4x against `TinyJit`, which skips that work, and 16x with `MiniJit`. GPT-2 decode spends 0.2 ms per token on host work |
| Competitive kernels | `gemm` reaches 85% of peak on one thread; decode reads weights at the memory bandwidth an elementwise pass reaches; everything else is unmeasured against other libraries |
| No hidden trade | Same models, float32, same batch sizes, on CPU. No PyTorch or MAX baselines yet |
