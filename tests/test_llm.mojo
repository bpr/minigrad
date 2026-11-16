"""Inference pieces: JSON, safetensors, eager writes and gathers,
attention, and (when data/gpt2 exists) the tokenizer and GPT-2."""

from std.math import exp, sqrt
from std.os.path import exists
from std.testing import TestSuite, assert_equal, assert_true, assert_false

from minigrad.dtype import DT
from minigrad.json import JsonCursor
from minigrad.runtime.pool import run_parallel
from minigrad.safetensors import safe_load
from minigrad.tensor import Tensor
from helpers import assert_close, assert_shape


def _bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for b in s.as_bytes():
        out.append(b)
    return out^


def test_json() raises:
    var c = JsonCursor(
        _bytes(
            '{"a\\u00e9\\n": [1, -20, 3], "skip": {"x": [true, null, 1.5e3,'
            ' "}"]}, "s": "\\ud83d\\udd25\\"q\\""}'
        )
    )
    c.expect("{")
    assert_equal(c.string(), "aé\n")
    c.expect(":")
    var l = c.int_list()
    assert_equal(l, [1, -20, 3])
    c.expect(",")
    assert_equal(c.string(), "skip")
    c.expect(":")
    c.skip_value()
    c.expect(",")
    assert_equal(c.string(), "s")
    c.expect(":")
    assert_equal(c.string(), '🔥"q"')
    c.expect("}")


def test_safetensors_roundtrip() raises:
    var path = String("build/test.safetensors")
    var header = String(
        '{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2,3],'
        '"data_offsets":[0,24]},"i":{"dtype":"I64","shape":[2],'
        '"data_offsets":[24,40]}}'
    )
    var data = List[UInt8]()
    var n = header.byte_length()
    for i in range(8):
        data.append(UInt8((n >> (8 * i)) & 0xFF))
    for b in header.as_bytes():
        data.append(b)
    var w: List[Float64] = [1.5, -2, 3, 4, 5, 6.25]
    var wt = Tensor.from_list(w, [2, 3])
    var p = wt.data_ptr()
    for i in range(24):
        data.append(p[unsafe_offset=i])
    var iv: List[Float64] = [7, -9]
    var it = Tensor.from_list(iv, [2], DT.int64)
    var q = it.data_ptr()
    for i in range(16):
        data.append(q[unsafe_offset=i])
    with open(path, "w") as f:
        f.write_bytes(Span(data))
    var t = safe_load(path)
    assert_shape(t["w"], [2, 3])
    assert_close(t["w"], w)
    assert_equal(t["i"].dtype(), DT.int64)
    assert_close(t["i"], iv)
    var skipped = safe_load(path, skip_suffix="i")
    assert_false("i" in skipped)


def test_take_rows() raises:
    var t = Tensor.arange(12).reshape(4, 3).float()
    assert_close(t.take_rows([2, 0, 2]), [6, 7, 8, 0, 1, 2, 6, 7, 8])
    var raised = False
    try:
        _ = t.take_rows([4])
    except:
        raised = True
    assert_true(raised)


def test_write() raises:
    var cache = Tensor.empty([5, 3])
    cache.write_(0, 0, Tensor.zeros(5, 3))
    # A view of the cache sees later writes; a copy does not.
    var view = cache.slice(0, 0, 2)
    var copy = cache.slice(0, 0, 2).contiguous().realize()
    cache.write_(0, 1, Tensor.arange(6).reshape(2, 3).float() + 1.0)
    assert_close(cache, [0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0])
    assert_close(copy, [0, 0, 0, 0, 0, 0])
    assert_close(view, [0, 0, 0, 1, 2, 3])
    # Along an inner dimension.
    var m = Tensor.zeros(2, 4).contiguous().realize()
    m.write_(1, 2, Tensor.ones(2, 2) * 7.0)
    assert_close(m, [0, 0, 7, 7, 0, 0, 7, 7])
    var raised = False
    try:
        m.write_(1, 3, Tensor.ones(2, 2))
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        (m * 2.0).write_(0, 0, Tensor.ones(1, 4))
    except:
        raised = True
    assert_true(raised)


def test_tril_triu() raises:
    var x = Tensor.ones(3, 3)
    assert_close(x.tril(), [1, 0, 0, 1, 1, 0, 1, 1, 1])
    assert_close(x.triu(1), [0, 1, 1, 0, 0, 1, 0, 0, 0])


def _attention_ref(
    q: List[Float64],
    k: List[Float64],
    v: List[Float64],
    t: Int,
    l: Int,
    d: Int,
    causal: Bool,
) -> List[Float64]:
    var out = List[Float64]()
    for i in range(t):
        var s = List[Float64]()
        var m = -1e300
        for j in range(l):
            var dot = 0.0
            for e in range(d):
                dot += q[i * d + e] * k[j * d + e]
            dot /= sqrt(Float64(d))
            if causal and j > l - t + i:
                dot = -1e300
            s.append(dot)
            m = max(m, dot)
        var total = 0.0
        for j in range(l):
            s[j] = exp(s[j] - m) if s[j] > -1e299 else 0.0
            total += s[j]
        for e in range(d):
            var acc = 0.0
            for j in range(l):
                acc += s[j] / total * v[j * d + e]
            out.append(acc)
    return out^


def test_attention() raises:
    Tensor.manual_seed(3)
    for causal in [False, True]:
        for tl in [(1, 7), (5, 5), (3, 8)]:
            var t = tl[0]
            var l = tl[1]
            var q = Tensor.randn(t, 4)
            var k = Tensor.randn(l, 4)
            var v = Tensor.randn(l, 4)
            var got = q.scaled_dot_product_attention(k, v, None, causal)
            var want = _attention_ref(
                q.tolist(), k.tolist(), v.tolist(), t, l, 4, causal
            )
            assert_close(got, want, 1e-5)
    # Heads as a batch dimension, and a boolean mask.
    var q = Tensor.randn(3, 2, 4)
    var k = Tensor.randn(3, 6, 4)
    var v = Tensor.randn(3, 6, 4)
    var keep = Tensor.arange(6).lt(4.0)
    var masked = q.scaled_dot_product_attention(k, v, keep)
    var cut = q.scaled_dot_product_attention(k.slice(1, 0, 4), v.slice(1, 0, 4))
    assert_close(masked, cut.tolist(), 1e-5)


def _square(ctx: Int, i: Int):
    var p = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=ctx)
    p[unsafe_offset=i] = i * i


def test_pool() raises:
    var out = List[Int](length=1000, fill=-1)
    for n in [1, 2, 7, 64, 1000]:
        run_parallel[_square](Int(out.unsafe_ptr()), n)
        for i in range(n):
            assert_equal(out[i], i * i)


def test_tokenizer() raises:
    if not exists("data/gpt2/vocab.json"):
        print("skipped: run examples/get_gpt2.sh for data/gpt2")
        return
    from minigrad.llm.bpe import BPETokenizer

    var tok = BPETokenizer("data/gpt2/vocab.json", "data/gpt2/merges.txt")
    # Ids from tiktoken's "gpt2" encoding.
    var ids = tok.encode("Hello, world! It's   a test.\n\n<|endoftext|>x")
    assert_equal(
        ids,
        [15496, 11, 995, 0, 632, 338, 220, 220, 257, 1332, 13, 628, 50256, 87],
    )
    var text = String("café — naïve \U0001f525 日本")
    assert_equal(tok.decode(tok.encode(text)), text)


def test_gpt2_greedy() raises:
    if not exists("data/gpt2/model.safetensors"):
        print("skipped: run examples/get_gpt2.sh for data/gpt2")
        return
    from minigrad.llm.gpt2 import GPT2

    # "What is the answer to life, the universe, and everything?" and the
    # greedy continuation tinygrad's examples/gpt2.py gives; 16-bit
    # weights give the same tokens.
    var prompt: List[Int] = [
        2061,
        318,
        262,
        3280,
        284,
        1204,
        11,
        262,
        6881,
        11,
        290,
        2279,
        30,
    ]
    var expect: List[Int] = [
        198,
        198,
        464,
        3280,
        284,
        1204,
        11,
        262,
        6881,
        11,
        290,
        2279,
        318,
        326,
        356,
        389,
        477,
        5884,
        13,
    ]
    for dt in [DT.float32, DT.float16, DT.bfloat16]:
        var model = GPT2("data/gpt2/model.safetensors", 64, dt)
        assert_equal(model.generate(prompt, len(expect)), expect)


def test_gpt2_decode_matches_prompt() raises:
    """Decoding token by token (masked attention over 64-row blocks of
    the cache, schedules replayed, and through MiniJit) gives the logits
    that processing the whole sequence at once does, across a block
    boundary."""
    if not exists("data/gpt2/model.safetensors"):
        print("skipped: run examples/get_gpt2.sh for data/gpt2")
        return
    from minigrad.graph import graph
    from minigrad.llm.gpt2 import GPT2

    var model = GPT2("data/gpt2/model.safetensors", 128)
    var ids = List[Int]()
    for i in range(70):
        ids.append((i * 7919) % 50000)
    var head = List[Int]()
    for i in range(58):
        head.append(ids[i])
    var hits = graph()[].stats.schedule_hits
    var logits = model.forward(head, 0)
    for p in range(58, 70):
        logits = model.forward([ids[p]], p)
    var whole = model.forward(ids, 0)
    assert_close(logits, whole.tolist(), 2e-3)
    assert_true(graph()[].stats.schedule_hits > hits + 8)
    # The same through MiniJit: steps 3 onward in each block replay.
    _ = model.forward(head, 0)
    for p in range(58, 70):
        logits = model.decode(ids[p], p)
    assert_close(logits, whole.tolist(), 2e-3)


def test_gpt2_batch_matches_single() raises:
    """Generating several prompts of different lengths together gives
    each the greedy tokens it gets alone, across a block boundary."""
    if not exists("data/gpt2/model.safetensors"):
        print("skipped: run examples/get_gpt2.sh for data/gpt2")
        return
    from minigrad.llm.gpt2 import GPT2

    var prompts = List[List[Int]]()
    prompts.append(
        [2061, 318, 262, 3280, 284, 1204, 11, 262, 6881, 11, 290, 2279, 30]
    )
    prompts.append([15496, 13])
    var long = List[Int]()
    for i in range(55):
        long.append((i * 7919) % 50000)
    prompts.append(long^)
    var together = GPT2("data/gpt2/model.safetensors", 128, batch=3)
    var got = together.generate_batch(prompts, 16)
    var alone = GPT2("data/gpt2/model.safetensors", 128)
    for i in range(3):
        assert_equal(got[i], alone.generate(prompts[i], 16))


def test_continuous_batching() raises:
    """Requests arriving over time, more than there are slots, each get
    the greedy tokens they get alone; finished requests free their slots
    (including slots below occupied ones) for waiting ones."""
    if not exists("data/gpt2/model.safetensors"):
        print("skipped: run examples/get_gpt2.sh for data/gpt2")
        return
    from minigrad.llm.batching import Batcher
    from minigrad.llm.gpt2 import GPT2

    var prompts = List[List[Int]]()
    prompts.append(
        [2061, 318, 262, 3280, 284, 1204, 11, 262, 6881, 11, 290, 2279, 30]
    )
    prompts.append([15496, 13])
    var long = List[Int]()
    for i in range(55):
        long.append((i * 7919) % 50000)
    prompts.append(long^)
    prompts.append([464, 2068, 7586, 21831, 18045, 625, 262])
    prompts.append([40, 1101])
    var limits: List[Int] = [12, 4, 15, 8, 6]
    var b = Batcher(
        GPT2("data/gpt2/model.safetensors", 128, batch=3), stop_at_eot=False
    )
    var ids = List[Int]()
    for i in range(3):
        ids.append(b.submit(prompts[i], limits[i]))
    var steps = 0
    var peak = 0
    while b.step():
        steps += 1
        peak = max(peak, b.active())
        if steps == 2:
            ids.append(b.submit(prompts[3], limits[3]))  # waits for a slot
        if steps == 5:
            ids.append(b.submit(prompts[4], limits[4]))
    assert_equal(peak, 3)
    var alone = GPT2("data/gpt2/model.safetensors", 128)
    for i in range(5):
        assert_true(b.finished(ids[i]))
        assert_equal(b.output(ids[i]), alone.generate(prompts[i], limits[i]))


def test_gpt2_perplexity() raises:
    """Lower-precision weights are exact as a mechanism and cheap in
    quality. Stored as 16-bit floats, the model computes what a float32
    model with the same rounded weights does; perplexity on 256 tokens of
    Shakespeare moves by under 5% (bfloat16's rounding happens to lower
    it by 3%) and by under 10% for int8."""
    if not exists("data/gpt2/tinyshakespeare.txt"):
        print("skipped: run examples/get_gpt2.sh for data/gpt2")
        return
    from minigrad.llm.bpe import BPETokenizer
    from minigrad.llm.gpt2 import GPT2

    var tok = BPETokenizer("data/gpt2/vocab.json", "data/gpt2/merges.txt")
    var text: String
    with open("data/gpt2/tinyshakespeare.txt", "r") as f:
        text = f.read(4000)
    var ids = tok.encode(text)
    var window = List[Int]()
    for i in range(256):
        window.append(ids[i])
    var base = GPT2("data/gpt2/model.safetensors", 256).perplexity(window)
    assert_true(base > 20.0 and base < 300.0)
    for dt in [DT.float16, DT.bfloat16, DT.int8]:
        var p = GPT2("data/gpt2/model.safetensors", 256, dt).perplexity(window)
        var limit = 0.10 if dt == DT.int8 else 0.05
        if abs(p / base - 1.0) > limit:
            raise Error(t"{dt}: perplexity {p} against {base} in float32")
        if dt == DT.int8:
            continue
        var m = GPT2("data/gpt2/model.safetensors", 256)
        m.wte.w = m.wte.w.cast(dt).float().realize()
        for i in range(len(m.blocks)):
            m.blocks[i].attn_w.w = (
                m.blocks[i].attn_w.w.cast(dt).float().realize()
            )
            m.blocks[i].proj_w.w = (
                m.blocks[i].proj_w.w.cast(dt).float().realize()
            )
            m.blocks[i].fc_w.w = m.blocks[i].fc_w.w.cast(dt).float().realize()
            m.blocks[i].fc_proj_w.w = (
                m.blocks[i].fc_proj_w.w.cast(dt).float().realize()
            )
        var rounded = m.perplexity(window)
        if abs(p / rounded - 1.0) > 1e-6:
            raise Error(
                t"{dt}: {p} stored natively, {rounded} rounded in float32"
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
