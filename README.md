# minigrad

A tensor library with automatic differentiation in
[Mojo](https://docs.modular.com/mojo/), modeled on
[tinygrad](https://github.com/tinygrad/tinygrad).

minigrad keeps tinygrad's way of thinking (a small tensor API, lazy
evaluation, a graph you can inspect, gradients as graph
transformations, fusion into few kernels) and rebuilds what is
underneath in one compiled language: the graph is an arena of
hash-consed nodes, movement operations are strided views, and kernels
run on a block interpreter and a tuned matrix-multiply kernel instead of
generated code. It is a new implementation with compatibility goals, not
a translation.

```mojo
from minigrad import Tensor
from minigrad.nn import Adam, Linear, collect


def main() raises:
    # Lazy: nothing is computed until a value is read.
    var x = Tensor.rand(4, 3).requires_grad_()
    var loss = (x @ Tensor.ones(3, 2)).relu().sum()
    loss.backward()
    print(x.grad().to_string())

    # A two-layer network learning XOR.
    var inputs: List[List[Float64]] = [[0, 0], [0, 1], [1, 0], [1, 1]]
    var targets: List[Float64] = [0, 1, 1, 0]
    var xs = Tensor(inputs)
    var ys = Tensor(targets).reshape(4, 1)
    var l1 = Linear(2, 8)
    var l2 = Linear(8, 1)
    var opt = Adam(collect(l1, l2), lr=0.05)
    for step in range(300):
        opt.zero_grad()
        var err = ((l2(l1(xs).tanh()).sigmoid() - ys) ** 2).mean()
        err.backward()
        opt.step([err])
    print(l2(l1(xs).tanh()).sigmoid().to_string())
```

## Status

Early. What works, on CPU:

- **Tensors**: creation, arithmetic with broadcasting and type promotion,
  comparisons, `where`, reductions, reshape/permute/expand/pad/shrink/
  flip, `cat`/`stack`, matmul, convolution (stride, dilation, groups,
  padding), pooling, softmax, cross-entropy, layer and batch
  normalization, attention; thirteen element types, `float16` and
  `bfloat16` among them.
- **Gradients** of all of the above, including second derivatives.
- **`minigrad.nn`**: `Linear`, `Conv2d`, `BatchNorm`, `LayerNorm`; `SGD`,
  `Adam`, `AdamW`.
- **`minigrad.linalg`**: `solve`, `det`, `inv`, `cholesky`, `eigh` on small
  dense matrices (eager, in float64, not differentiable), shared with
  MojoDEC.
- **MNIST**: `examples/mnist.mojo` trains tinygrad's `beautiful_mnist`
  model to 97.9% test accuracy in 70 steps.
- **Batched generation**: `GPT2(..., batch=B)` and `generate_batch`
  continue up to `B` prompts of different lengths at once, each getting
  the tokens it would get alone; each step reads the weights once for
  all of them.
- **GPT-2 inference**: `examples/gpt2.mojo` loads Hugging Face's
  safetensors weights, tokenizes with GPT-2's BPE, and generates with a
  key-value cache. Its greedy output matches tinygrad's token for token,
  its logits agree to 2e-4, and its tokenizer agrees with tiktoken.

- **Schedule caching and `MiniJit`**: a graph of the same structure as
  one scheduled before replays its recorded kernels; `MiniJit` (like
  tinygrad's `TinyJit`) goes
  further and replays a whole step without running its code (a small
  training step: 78 to 21 µs). GPT-2's `decode` uses it.
- **Lower-precision weights**: GPT-2's matrices can be stored as
  `float16`, `bfloat16`, or `int8` (one scale per output) and are
  converted as the matrix-multiply kernel reads them. 16-bit weights
  halve decode time with no measurable loss; int8 cuts it by about 55%
  for about 4% higher perplexity.

- **Continuous batching**: `Batcher` serves requests that arrive at any
  time, up to `batch` at once, a finished one's slot going to the next.

What does not exist yet: a GPU backend, Python bindings, an HTTP server
in front of the batcher, group-wise or 4-bit quantization, indexing with slices and
index tensors, and models beyond GPT-2. `docs/plan.md` has the order of
work.

Measured against tinygrad's CPU backend on one machine, GPT-2 decodes
at 10 ms per token with float32 weights against tinygrad's 122 ms (close
to the floor that reading 497 MB of weights per token sets), 5.8 ms with
16-bit weights, and 4.3 ms with int8; a batch of 8 sequences decodes at
1.2 to 1.3 times the cost of one. A training step is 9 to 17 times
faster than under `TinyJit`. Most numbers are single runs on a busy
machine.
`benchmarks/README.md` has the tables, how to reproduce them, and what
they do not show.

## Getting started

minigrad needs [pixi](https://pixi.sh) and Linux on x86-64.

```sh
pixi install
pixi run test                                  # 122 tests, about a minute

bash examples/get_mnist.sh                     # 55 MB into data/mnist/
pixi run mojo run -I . examples/mnist.mojo     # 70 steps, batch 512

bash examples/get_gpt2.sh                      # 550 MB into data/gpt2/
pixi run mojo run -I . examples/gpt2.mojo --prompt "Hello." --count 40
```

`examples/gpt2.mojo` takes `--temperature` (0 for greedy; default 0.8),
`--top_k`, `--seed`, `--timing`, and `--dtype` (`float32`, `float16`,
`bfloat16`, `int8`).

Programs import the package from the source tree with `-I .`. Set
`MINIGRAD_DEBUG=2` to print every kernel as it runs:

```
gemm   15366.671 us  float32 [512, 1, 32, 24, 24, 32, 5, 5] reduce [5, 6, 7]  loads 2 instrs 3 loops []
blocks 8719.815 us  float32 [512, 32, 20, 20]  loads 8 instrs 29 loops [512, 32, 400]
```

## How it differs from tinygrad to use

The API follows tinygrad's names. The differences that come up first:

- Shapes, types, and gradients are calls: `t.shape()`, `t.dtype()`,
  `t.grad()`.
- Comparisons are methods (`a.lt(b)`), and tuples become variadic
  integers (`t.reshape(2, 3)`, `t.sum(0, 1)`).
- Tensor code lives in `raises` functions.
- `assign` and `+=` point the tensor at a new value; no buffer is ever
  overwritten.
- Call `backward()` before reading the loss, and pass the loss to the
  optimizer so it is computed with the update: `opt.step([loss])`.

`docs/differences-from-tinygrad.md` is the full list.

## Documentation

| | |
|---|---|
| `docs/architecture.md` | How a computation becomes loops, and why it is built this way |
| `docs/differences-from-tinygrad.md` | Every deliberate deviation |
| `docs/plan.md` | What is next; what was tried and dropped |
| `benchmarks/README.md` | Numbers and how to reproduce them |
| `AGENTS.md` | Layout, conventions, and Mojo traps, for people and coding assistants |

## License

The MIT License (MIT); see [LICENSE](LICENSE).

tinygrad, whose API and several algorithms minigrad follows (the view
algebra and the gradient rules in particular), is under its own MIT
license; see [NOTICE](NOTICE).
