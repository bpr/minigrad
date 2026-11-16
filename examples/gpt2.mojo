"""Generate text with GPT-2.

    bash examples/get_gpt2.sh
    mojo run -I . examples/gpt2.mojo [--prompt TEXT] [--count N]
        [--temperature T] [--top_k K] [--seed S] [--timing]
        [--dtype float32|float16|bfloat16|int8]

Defaults follow tinygrad's `examples/gpt2.py`, with the 124M model.
"""

from std.sys import argv
from std.time import perf_counter_ns

from minigrad.dtype import DT, parse_dtype
from minigrad.llm.bpe import BPETokenizer
from minigrad.llm.gpt2 import GPT2, sample
from minigrad.tensor import Tensor


def main() raises:
    var prompt = String(
        "What is the answer to life, the universe, and everything?"
    )
    var count = 100
    var temperature = 0.8
    var top_k = 0
    var timing = False
    var dtype = DT.float32
    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--prompt":
            prompt = String(args[i + 1])
            i += 1
        elif a == "--count":
            count = atol(args[i + 1])
            i += 1
        elif a == "--temperature":
            temperature = atof(args[i + 1])
            i += 1
        elif a == "--top_k":
            top_k = atol(args[i + 1])
            i += 1
        elif a == "--seed":
            Tensor.manual_seed(atol(args[i + 1]))
            i += 1
        elif a == "--dtype":
            dtype = parse_dtype(String(args[i + 1]))
            i += 1
        elif a == "--timing":
            timing = True
        else:
            raise Error(t"unknown argument {a}")
        i += 1

    var t0 = perf_counter_ns()
    var tok = BPETokenizer("data/gpt2/vocab.json", "data/gpt2/merges.txt")
    var model = GPT2("data/gpt2/model.safetensors", dtype=dtype)
    var t1 = perf_counter_ns()
    if timing:
        print("loaded in", Float64(t1 - t0) / 1e6, "ms")
    var ids = tok.encode(prompt)
    var out = ids.copy()
    var logits = model.forward(ids, 0)
    var t2 = perf_counter_ns()
    if timing:
        print(len(ids), "prompt tokens in", Float64(t2 - t1) / 1e6, "ms")
    var pos = len(ids)
    var steps = List[Int]()
    for k in range(count):
        var next = sample(logits.tolist(), temperature, top_k)
        out.append(next)
        if k + 1 < count:
            var s = perf_counter_ns()
            logits = model.decode(next, pos)
            steps.append(Int(perf_counter_ns() - s))
            pos += 1
    print(tok.decode(out))
    if timing and len(steps) > 0:
        sort(steps)
        print(
            "per token: median",
            Float64(steps[len(steps) // 2]) / 1e6,
            "ms, min",
            Float64(steps[0]) / 1e6,
            "ms",
        )
