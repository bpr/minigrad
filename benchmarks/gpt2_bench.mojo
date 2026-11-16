"""GPT-2 (124M) inference: checks against tinygrad, and timings.

    mojo build -I . -o build/gpt2_bench benchmarks/gpt2_bench.mojo
    build/gpt2_bench encode TEXT          # token ids of TEXT
    build/gpt2_bench logits IDS...        # last position's logits
    build/gpt2_bench greedy N IDS...      # ids of N greedy tokens
    build/gpt2_bench bench N IDS...       # time the prompt and N steps
    build/gpt2_bench ppl FILE N           # perplexity on N windows of FILE

Before the action, `--dtype NAME` stores the weights as `float16`,
`bfloat16`, or `int8` instead of `float32`.

The same actions as benchmarks/tinygrad/gpt2_bench.py, which runs
tinygrad's GPT-2 on the same weights. Weights and tokenizer come from
data/gpt2 (examples/get_gpt2.sh).
"""

from std.math import exp, log
from std.sys import argv
from std.time import perf_counter_ns

from minigrad.dtype import DT, parse_dtype
from minigrad.graph import graph
from minigrad.llm.bpe import BPETokenizer
from minigrad.llm.gpt2 import GPT2, sample


def main() raises:
    var all_args = argv()
    var dtype = DT.float32
    var args = List[String]()
    var i0 = 1
    if len(all_args) > 2 and String(all_args[1]) == "--dtype":
        dtype = parse_dtype(String(all_args[2]))
        i0 = 3
    args.append(String(all_args[0]))
    for i in range(i0, len(all_args)):
        args.append(String(all_args[i]))
    var action = String(args[1])
    if action == "encode":
        var tok = BPETokenizer("data/gpt2/vocab.json", "data/gpt2/merges.txt")
        var ids = tok.encode(String(args[2]))
        var s = String("")
        for i in range(len(ids)):
            s += (" " if i > 0 else "") + String(ids[i])
        print(s)
        return
    if action == "ppl":
        # Windows of 512 tokens, one after another from the start.
        var tok = BPETokenizer("data/gpt2/vocab.json", "data/gpt2/merges.txt")
        var text: String
        with open(args[2], "r") as f:
            text = f.read()
        var ids = tok.encode(text)
        var model = GPT2("data/gpt2/model.safetensors", 512, dtype)
        var windows = atol(args[3])
        var total = 0.0
        for w in range(windows):
            var chunk = List[Int]()
            for i in range(w * 512, (w + 1) * 512):
                chunk.append(ids[i])
            total += log(model.perplexity(chunk))
        print(
            t"{dtype}: perplexity {exp(total / Float64(windows))} over"
            t" {windows} windows of 512 tokens"
        )
        return
    var nums = List[Int]()
    for i in range(2, len(args)):
        nums.append(atol(args[i]))
    var model = GPT2("data/gpt2/model.safetensors", dtype=dtype)
    if action == "logits":
        var l = model.forward(nums, 0).tolist()
        var s = String("")
        for i in range(len(l)):
            s += (" " if i > 0 else "") + String(l[i])
        print(s)
    elif action == "greedy":
        var ids = List[Int]()
        for i in range(1, len(nums)):
            ids.append(nums[i])
        var out = model.generate(ids, nums[0])
        var s = String("")
        for i in range(len(out)):
            s += (" " if i > 0 else "") + String(out[i])
        print(s)
    elif action == "bench":
        var n = nums[0]
        var ids = List[Int]()
        for i in range(1, len(nums)):
            ids.append(nums[i])
        var t0 = perf_counter_ns()
        var logits = model.forward(ids, 0)
        var tok = sample(logits.tolist(), 0.0, 0)
        var t1 = perf_counter_ns()
        # Again: plans exist now (the cache is simply overwritten).
        logits = model.forward(ids, 0)
        tok = sample(logits.tolist(), 0.0, 0)
        var t2 = perf_counter_ns()
        var pos = len(ids)
        var times = List[Int]()
        var compiled = graph()[].stats.kernels_compiled
        var run0 = graph()[].stats.kernels_run
        var total = 0
        for i in range(n):
            if i == 1:
                graph()[].stats.ns_group = 0
                graph()[].stats.ns_build = 0
                graph()[].stats.ns_execute = 0
                total = 0
            var s = perf_counter_ns()
            logits = model.decode(tok, pos)
            tok = sample(logits.tolist(), 0.0, 0)
            var dt = Int(perf_counter_ns() - s)
            times.append(dt)
            total += dt
            pos += 1
        var first = times[0]
        sort(times)
        var st = graph()[].stats.copy()
        print(
            t"prompt of {len(ids)} tokens: {Float64(t1 - t0) / 1e6} ms cold,"
            t" {Float64(t2 - t1) / 1e6} ms warm; first decode step"
            t" {Float64(first) / 1e6} ms; decode median"
            t" {Float64(times[len(times) // 2]) / 1e6} ms, min"
            t" {Float64(times[0]) / 1e6} ms over {n} steps"
        )
        var per = Float64(max(1, n - 1)) * 1e3
        print(
            t"decode step (us): building the graph"
            t" {Float64(total - st.ns_group - st.ns_build - st.ns_execute) / per},"
            t" choosing kernels {Float64(st.ns_group) / per}, building kernels"
            t" {Float64(st.ns_build) / per}, running kernels"
            t" {Float64(st.ns_execute) / per}"
        )
        print(
            t"kernels per decode step: {(st.kernels_run - run0) // max(1, n)};"
            t" plans built during decode: {st.kernels_compiled - compiled}"
        )
    else:
        raise Error(t"unknown action {action}")
