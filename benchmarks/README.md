# Benchmarks

Every number here was measured on one machine and is a statement about
that machine: an AMD Ryzen 7 9800X3D (8 cores, AVX-512), Linux, Mojo
1.3.0.dev2026111316, tinygrad at commit `b0e391d72` running on its CPU
backend (`DEV=CPU`) under Python 3.12. Times are medians over the steps
of one run after the first.

**What these do and do not show.** They compare minigrad on CPU with
tinygrad on CPU, for training two small models. tinygrad's CPU backend
is not where its effort goes, there is no GPU backend here yet, and none
of the brief's transformer-inference benchmarks exist yet (see
`docs/plan.md`). Read the tables as "the design works and the host-side
costs are small", not as a verdict on the rewrite.

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

## A two-layer MLP: one training step

784 inputs, one hidden layer with ReLU, 10 outputs; cross-entropy; SGD
with momentum; random data. A step is forward, backward, the update, and
reading the loss.

| batch / hidden | minigrad | tinygrad, `TinyJit` | tinygrad, no JIT | vs JIT | vs no JIT |
|---|---:|---:|---:|---:|---:|
| 8 / 32 | 105 µs | 828 µs | 10,510 µs | 7.9x | 100x |
| 64 / 128 | 259 µs | 1,324 µs | 11,242 µs | 5.1x | 43x |
| 512 / 512 | 1,262 µs | 27,702 µs | 37,454 µs | 22x | 30x |

At 512 / 512 with one thread minigrad takes 3,651 µs (7.6x tinygrad's
JIT).

The first row is almost all host work, and is the comparison the brief
cares about for small and dynamic workloads. minigrad rebuilds the
graph, differentiates it, and schedules it on every step; `TinyJit`
replays a recorded schedule and does none of that. Where minigrad's 105
µs go:

| | µs per step |
|---|---:|
| Building the graph (forward, gradient, update) | 47 |
| Choosing kernel roots | 3 |
| Building kernels and looking up plans | 20 |
| Running 29 kernels | 40 |

Plans are built once: 33 over the whole run, none after the first step.

## The MNIST convolutional network: one training step

tinygrad's `beautiful_mnist` model (four convolutions, two batch
normalizations, two max pools, a linear layer), Adam, random data.

| batch | minigrad | tinygrad, `TinyJit` | tinygrad, no JIT | vs JIT |
|---|---:|---:|---:|---:|
| 64 | 19.9 ms | 219.6 ms | 279.7 ms | 11x |
| 512 | 145.5 ms | 1,735 ms | (not run) | 12x |

At batch 512 with one thread minigrad takes 404.6 ms (4.3x).

First step, which for tinygrad includes generating and compiling its
kernels and for minigrad includes building 107 plans:

| batch | minigrad | tinygrad, `TinyJit` |
|---|---:|---:|
| 64 | 26.5 ms | 1,776 ms |
| 512 | 201.8 ms | 4,831 ms |

## MNIST, start to finish

`examples/mnist.mojo` against tinygrad's `examples/beautiful_mnist.py`:
70 steps at batch 512, test accuracy on all 10,000 test images every 10
steps, including startup and loading the data.

| | wall time | final test accuracy |
|---|---:|---:|
| minigrad | 14.3 s | 97.86% |
| tinygrad (`DEV=CPU`, `STEPS=70`) | 183 s | 97.63% |

The two runs draw different random batches and initial weights, so the
accuracies agree only as well as two seeds would.

## The matrix-multiply kernel

`minigrad/runtime/gemm.mojo` alone, float32, best of several runs:

| A x B | threads | time | GFLOPS |
|---|---:|---:|---:|
| 64 x 784 times 784 x 128 | 1 | 44 µs | 290 |
| 512 x 784 times 784 x 512 | 1 | 1.35 ms | 304 |
| 512 x 784 times 784 x 512 | 8 | 0.38 ms | 1,081 |
| 1024 x 1024 times 1024 x 1024 | 1 | 7.4 ms | 291 |
| 1024 x 1024 times 1024 x 1024 | 8 | 2.0 ms | 1,063 |

One thread is near this processor's peak for AVX-512 fused multiply-adds
(two per cycle, 16 lanes: about 330 GFLOPS at 5.2 GHz). Eight threads
reach 3.6x one, not 8x: threads are created per call and each packs its
own copy of B. A thread pool is in `docs/plan.md`.

## Against the brief's bar

`~/markdown/tinygrad-mojo.md` sets seven criteria. Where this stands:

| Criterion | Status |
|---|---|
| Runs a meaningful subset correctly, with backward and optimizers | Yes for MLPs and convnets: 88 tests, gradients checked against finite differences, MNIST to 97.9% |
| Usable from Python without writing Mojo | Not started |
| At least 5x on transformer inference | Not measurable yet: no transformer |
| At least 2-3x on a training workload | 5x to 22x over tinygrad's JIT on CPU, on these two models |
| At least 10x less graph, scheduling, and dispatch overhead | 100x against tinygrad without JIT at the smallest size; 8x against `TinyJit`, which skips that work |
| Competitive kernels | `gemm` is near peak on one thread; everything else is unmeasured against other libraries |
| No hidden trade | Same models, float32, same batch sizes, on CPU. No PyTorch or MAX baselines yet |
