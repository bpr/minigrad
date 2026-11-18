# Benchmarks

Every number here was measured on one machine and is a statement about
that machine: an AMD Ryzen 7 9800X3D (8 cores, AVX-512), Linux, Mojo
1.3.0.dev2026111316, tinygrad at commit `b0e391d72` running on its CPU
backend (`DEV=CPU`) under Python 3.12. Times are medians over the steps
of one run after the first.

**What these do and do not show.** They compare minigrad on CPU with
tinygrad on CPU: GPT-2 (124M) inference and training two small models.
tinygrad's CPU backend is not where its effort goes, and there is no
GPU backend here yet, so read the ratios as "against tinygrad as it
runs on a CPU", not as a verdict on the rewrite. No PyTorch, llama.cpp,
or MAX baselines yet.

**Noise.** The CPU governor is `powersave`, and other sessions on this
machine compile in bursts. The tables come from runs started when the
load average was below 1 (it rose to about 2 during some). Multi-thread
medians repeat to within about 10%; single-thread figures do not (a
lone thread may run on a core still clocked down) and are given only
as rough guides.

## Reproducing

```sh
pixi run mojo build -I . -o build/mlp benchmarks/mlp_step.mojo
pixi run mojo build -I . -o build/convnet benchmarks/convnet_step.mojo
build/mlp 64 128 300          # batch, hidden units, steps
build/convnet 512 20          # batch, steps

cd ~/src/python/tinygrad      # a tinygrad checkout
PYTHONPATH=. DEV=CPU python3 <minigrad>/benchmarks/tinygrad/mlp_step.py 64 128 300 jit
PYTHONPATH=. DEV=CPU python3 <minigrad>/benchmarks/tinygrad/convnet_step.py 512 8 jit
```

Drop the trailing `jit` to run tinygrad without `TinyJit`.
`MINIGRAD_THREADS=1` limits minigrad to one thread; `MINIGRAD_DEBUG=2`
prints every kernel with its time.

## GPT-2 (124M) inference

`benchmarks/gpt2_bench.mojo` against `benchmarks/tinygrad/gpt2_bench.py`,
which runs tinygrad's own `examples/gpt2.py` model on the same
safetensors weights (decode under `TinyJit`, as tinygrad's example
does). float32 weights in both, batch of one, greedy decoding, 100
decode steps after the prompt.

```sh
bash examples/get_gpt2.sh
pixi run mojo build -I . -o build/gpt2_bench benchmarks/gpt2_bench.mojo
IDS=$(build/gpt2_bench encode "What is the answer to life, the universe, and everything?")
build/gpt2_bench bench 100 $IDS
cd ~/src/python/tinygrad
PYTHONPATH=. DEV=CPU python3 <minigrad>/benchmarks/tinygrad/gpt2_bench.py \
    <minigrad>/data/gpt2/model.safetensors bench 100 $IDS
```

| | minigrad | tinygrad | ratio |
|---|---:|---:|---:|
| Prompt of 13 tokens, first run | 23.5 ms | 1,570 ms | 67x |
| Prompt of 13 tokens, again | 19.7 ms | 942 ms | 48x |
| Decode, context 14 to 113: per token | 10.0 ms | 122.4 ms | 12x |
| Prompt of 527 tokens, first run | 188 ms | 31,006 ms | 165x |
| Prompt of 527 tokens, again | 153 ms | 30,257 ms | 198x |
| Decode, context 528 to 627: per token | 10.5 ms | 128.9 ms | 12x |
| A request: 13-token prompt, 100 new tokens | about 1.0 s | about 14.1 s | 14x |

The last row adds the components (prompt, first decode step, 99 more
steps); it is not separately timed. On one thread, minigrad decodes at
about 20.6 ms per token.

**Where a decode step goes.** Decode reads all 497 MB of weights once
per token. At this machine's memory bandwidth (about 60 GB/s, which the
matrix-vector kernels reach) that takes about 8 ms, so float32 decode
cannot get much below that. Of the 10.0 ms:

| | ms per token |
|---|---:|
| Running 307 kernels | 8.9 |
| Building the graph | 0.8 |
| Choosing kernels and building them | 0.4 |

The ratio over tinygrad is large because tinygrad's CPU backend runs
these matrix-vector products far below memory bandwidth, not because
minigrad does anything a CPU cannot. For the 527-token prompt, tinygrad
does not use `TinyJit` (its example only JITs single-token steps) and
spends most of the 30 s in attention; read that row as "tinygrad's CPU
path for long prompts is slow", not as a 165x better design. The
brief's warning applies: a ratio over 100x means the baseline is on a
slow path.

### Schedule cache

Decode steps of the same shapes replay a cached schedule (63 of every 64
steps). In one run with int8 weights it saved about 0.3 ms per token
(4.62 to 4.33 ms), and for float32 about the same (9.75 to 9.39 ms):
kernel building, about 0.3 ms per step, disappears, and computing the
graph's signature adds about 0.06 ms. A small MLP training step went
from 103 to 78 µs. Compare with `MINIGRAD_SCHEDULE_CACHE=0`.

### MiniJit

A replay runs the recorded kernels without building, scheduling, or
planning a graph. On a busy machine (load average 2 to 3), so treat the
decode figure as provisional:

| | plain | `MiniJit` |
|---|---:|---:|
| MLP training step, batch 8, hidden 32 | 78 µs | 21 µs |
| MLP training step, batch 64, hidden 128 | 169 µs | 120 µs |
| GPT-2 decode, int8 weights (cleanest round of three) | 4.28 ms | 3.40 ms |

The final losses with and without the JIT are identical. tinygrad's
`TinyJit` takes 828 µs for the first MLP row.

### Batched generation

`build/gpt2_bench batch B N IDS...` decodes `B` sequences of different
lengths at once; each sequence gets the greedy tokens it gets alone.
One decode step for every sequence of a batch, median of 60 steps
alternating the batch sizes in one process, 8 threads, load average
below 1:

| weights | batch 1 | batch 8 | batch 16 |
|---|---:|---:|---:|
| int8, step | 3.42 ms | 4.44 ms | 7.04 ms |
| int8, tokens/s | 292 | 1,803 | 2,274 |
| float32, step | 8.64 ms | 10.14 ms | 12.79 ms |
| float32, tokens/s | 116 | 789 | 1,251 |

A step of 8 costs 1.2 to 1.3 steps of 1: with one sequence a product is
limited by reading the weights, and the other seven sequences'
arithmetic is spread over the cores. Single-threaded the ratio is 3.0
(int8) and 2.2 (float32). Attention grows with the batch by nature,
since each sequence reads its own cache.

### Continuous batching

`build/gpt2_bench serve S R IDS...` serves `R` requests (prompts of 3
or more tokens, 16 to 63 new tokens each, all arriving at once) with a
`Batcher` of `S` slots, then one at a time, and prints throughput and
mean completion time for both. 8 slots, 24 requests (948 tokens), load
average about 2:

| weights | `Batcher`, 8 slots | one at a time | gain |
|---|---:|---:|---:|
| int8 | 1,056 tokens/s | 421 tokens/s | 2.5x |
| float32 | 480 tokens/s | 113 tokens/s | 4.2x |

The gain is below the batch's 8x because each admission processes its
prompt alone, and the batch drains as the longest requests finish.

### Lower-precision weights

The token embedding and every layer's matrices stored narrower (biases,
normalizations, and the position embedding stay float32; arithmetic is
float32 throughout). Perplexity is over the first 8 windows of 512
tokens of Tiny Shakespeare; decode is the median of 100 steps after the
13-token prompt. One run on a busy machine (load average about 1 to 2):
the decode times are good to perhaps 15%.

```sh
build/gpt2_bench --dtype bfloat16 ppl data/gpt2/tinyshakespeare.txt 8
build/gpt2_bench --dtype bfloat16 bench 100 $IDS
```

| weights | size | perplexity | greedy, 30 tokens | decode | of which kernels |
|---|---:|---:|---|---:|---:|
| float32 | 497 MB | 94.79 | reference | 9.9 ms | 8.9 ms |
| float16 | 249 MB | 94.88 | identical | 5.8 ms | 4.8 ms |
| bfloat16 | 249 MB | 92.41 | identical | 5.9 ms | 5.0 ms |
| int8, a scale per output | 124 MB | 98.87 | differs from token 7 | 4.3 ms | 3.3 ms |

bfloat16 lowering perplexity is not an improvement in the model: a
float32 model with the same rounded weights gives the same number to
every printed digit, so it is what rounding GPT-2 does to its fit on
this text, which is unlike what it was trained on. With int8 the host
work (about 1 ms per token of graph building and scheduling) is a
quarter of the step; caching schedules is the next lever.

**Correctness.** For a 28-token prompt, minigrad's logits agree with
tinygrad's to 2.2e-4 (tinygrad's values printed to 4 decimals; logits
range from -93 to -59), with the same top five tokens. 30 greedy tokens
agree exactly. minigrad's tokenizer gives tiktoken's ids on 18 test
texts covering contractions, whitespace runs, numbers, code, URLs,
accented Latin, typographic punctuation, emoji, CJK, and the special
token. To repeat the checks: the `logits` and `greedy` actions of both
scripts, and `tests/test_llm.mojo`.

## A two-layer MLP: one training step

784 inputs, one hidden layer with ReLU, 10 outputs; cross-entropy; SGD
with momentum; random data. A step is forward, backward, the update, and
reading the loss.

| batch / hidden | minigrad | tinygrad, `TinyJit` | tinygrad, no JIT | vs JIT | vs no JIT |
|---|---:|---:|---:|---:|---:|
| 8 / 32 | 76 µs | 828 µs | 10,510 µs | 11x | 138x |
| 64 / 128 | 116 µs | 1,324 µs | 11,242 µs | 11x | 97x |
| 512 / 512 | 717 µs | 27,702 µs | 37,454 µs | 39x | 52x |

minigrad's column is from a quiet machine (load average below 1), with
block kernels of 32K loop elements or more split over threads (before
that: 141 and 900 µs for the larger two sizes). Two earlier fixes
matter here: every product run through the executor had got one thread
(an argument passed through a stack struct by address read stale), and
a transposed A (a weight gradient's) is packed term by term, one chunk
per thread. With one thread the larger two steps take 228 µs and 3,316
µs. tinygrad's columns are from an earlier, noisier session.

The first row is almost all host work, and is the comparison the brief
cares about for small and dynamic workloads. minigrad rebuilds the
graph, differentiates it, and schedules it on every step; `TinyJit`
replays a recorded schedule and does none of that. Where minigrad's
step went, measured when it took 102 µs:

| | µs per step |
|---|---:|
| Building the graph (forward, gradient, update) | 46 |
| Choosing kernel roots | 3 |
| Building kernels and looking up plans | 20 |
| Running 29 kernels | 36 |

Plans are built once: 33 over the whole run, none after the first step.

### From Python

`benchmarks/mlp_step.py` is tinygrad's `mlp_step.py` with minigrad's
imports, on the Python bindings (`docs/python.md`). Medians over the
second half of 3,000 steps (300 for 512/512), on a quiet machine, two
runs each:

| batch / hidden | Mojo | Python | Python, `TinyJit` | tinygrad, `TinyJit` |
|---|---:|---:|---:|---:|
| 8 / 32 | 76 µs | 94 µs | 20 µs | 828 µs |
| 64 / 128 | 116 µs | 148 to 152 µs | 81 to 109 µs | 1,324 µs |
| 512 / 512 | 720 µs | 805 µs | 730 to 1,000 µs | 27,702 µs |

Without the JIT, Python adds about 20 µs to a step of about 100
operations: each is one call into the engine. Under `TinyJit` a replay
runs no Python and matches Mojo's `MiniJit` (21 µs for the first row).
The largest row is kernel time and the noisiest (the second run's
median rose with the load).

`examples/beautiful_mnist.py` (batches of 512, `TinyJit`) takes about
116 ms per training step against 107 ms for `examples/mnist.mojo`, run
back to back at a load average of 4 (not a quiet machine). The Python
step gathers uint8 images and converts them in the graph; the Mojo
example builds float batches on the host.

```sh
pixi run python-build
PYTHONPATH=python pixi run python benchmarks/mlp_step.py 8 32 3000 jit
STEPS=9 PYTHONPATH=python pixi run python examples/beautiful_mnist.py
```

## The MNIST convolutional network: one training step

tinygrad's `beautiful_mnist` model (four convolutions, two batch
normalizations, two max pools, a linear layer), Adam, random data.

| batch | minigrad | tinygrad, `TinyJit` | tinygrad, no JIT | vs JIT |
|---|---:|---:|---:|---:|
| 64 | 9.9 ms | 219.6 ms | 279.7 ms | 22x |
| 512 | 89.2 ms | 1,735 ms | (not run) | 19x |

Block kernels split over threads since: before, 17.9 and 153 ms (about
43% of the batch-512 step was block kernels on one thread). At batch 512
with one thread minigrad takes about 444 ms (5.0x).

First step, which for tinygrad includes generating and compiling its
kernels and for minigrad includes building 107 plans:

| batch | minigrad | tinygrad, `TinyJit` |
|---|---:|---:|
| 64 | 22 ms | 1,776 ms |
| 512 | 123 ms | 4,831 ms |

(Earlier figures of 59 and 472 ms did not reproduce: builds from before
the thread pool, from the commit that first reported them, and from
before block kernels used threads all took 28 to 36 ms and 200 to 255
ms on a quiet machine.)

## MNIST, start to finish

`examples/mnist.mojo` against tinygrad's `examples/beautiful_mnist.py`:
70 steps at batch 512, test accuracy on all 10,000 test images every 10
steps, including startup and loading the data.

| | wall time | final test accuracy |
|---|---:|---:|
| minigrad | 9.1 s | 98.04% |
| tinygrad (`DEV=CPU`, `STEPS=70`) | 183 s | 97.63% |

The two runs draw different random batches and initial weights, so the
accuracies agree only as well as two seeds would.

## The matrix-multiply kernel

`minigrad/runtime/gemm.mojo` alone, float32, best of several runs:

| A x B | threads | time | GFLOPS |
|---|---:|---:|---:|
| 64 x 784 times 784 x 128 | 1 | 45 µs | 288 |
| 512 x 784 times 784 x 512 | 1 | 1.34 ms | 307 |
| 512 x 784 times 784 x 512 | 8 | 0.27 ms | 1,466 |
| 512 x 784 times (512 x 784) transposed | 8 | 0.27 ms | 1,543 |
| 1024 x 1024 times 1024 x 1024 | 1 | 7.0 ms | 306 |
| 1024 x 1024 times 1024 x 1024 | 8 | 2.1 ms | 1,023 |

One thread is near this processor's peak for AVX-512 fused multiply-adds
(two per cycle, 16 lanes: about 330 GFLOPS at 5.2 GHz). Eight threads
reach 3.3x to 5x one: at these sizes packing and memory traffic take a
larger share as threads are added.

The matrix-vector path (up to 16 rows, B streamed once) reaches 54 to 62
GB/s on GPT-2's layers, the machine's memory bandwidth.

## Against the brief's bar

`~/markdown/tinygrad-mojo.md` sets seven criteria. Where this stands:

| Criterion | Status |
|---|---|
| Runs a meaningful subset correctly, with backward and optimizers | Yes for MLPs, convnets, and GPT-2 inference: 99 tests, gradients checked against finite differences, MNIST to 97.7-97.9%, GPT-2 matching tinygrad's output |
| Usable from Python without writing Mojo | Not started |
| At least 5x on transformer inference | 12x on GPT-2 decode, about 14x for a whole short request, against tinygrad on CPU; within 25% of the float32 memory-bandwidth floor. Batches of 8 sequences decode at 1.2-1.3x the cost of one. No GPU or larger models yet |
| At least 2-3x on a training workload | 9x to 17x over tinygrad's JIT on CPU, on these two models |
| At least 10x less graph, scheduling, and dispatch overhead | 138x against tinygrad without JIT at the smallest size; 11x against `TinyJit`, which skips that work. GPT-2 decode spends 1.2 ms per token on host work |
| Competitive kernels | `gemm` is near peak on one thread; matrix-vector products reach memory bandwidth; everything else is unmeasured against other libraries |
| No hidden trade | Same models, float32, same batch sizes, on CPU. No PyTorch or MAX baselines yet |
