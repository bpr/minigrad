"""GPT-2 inference (`minigrad.llm.gpt2`).

The model of tinygrad's `examples/gpt2.py`, loading Hugging Face's
safetensors weights. Each layer keeps a key-value cache: one
`(max_context, n_embd)` tensor each for keys and values, which `write_`
fills row by row, so a step processes only the new tokens. A prompt of
`T > 1` tokens at position `p` attends to the cache's first `p + T`
rows. A single token attends to the cache's first `p + 1` rows rounded up
to a multiple of `DECODE_ROWS`, through a mask that hides the rows not
yet written (tinygrad masks the whole cache). Decode steps then build
graphs of the same shapes for `DECODE_ROWS` steps at a time, which the
schedule cache replays without scheduling them again, while reading
little more of the cache than they need.

The large matrices can be stored in `float16`, `bfloat16`, or `int8`
(with a scale per output); everything is computed in `float32`. `gemm`
converts the weights as it reads them, so lower precision means less
memory traffic and nothing else.
"""

from std.math import exp
from std.utils.numerics import inf

from minigrad.dtype import DT
from minigrad.jit import MiniJit
from minigrad.nn import quantize_int8
from minigrad.safetensors import safe_load
from minigrad.tensor import Tensor, _next_uniform


@fieldwise_init
struct GPT2Config(Copyable, Writable):
    """The sizes of a GPT-2 model."""

    var n_layer: Int
    var n_head: Int
    var n_embd: Int
    var vocab_size: Int
    var n_ctx: Int
    """The longest sequence the position embedding covers."""


comptime DECODE_ROWS = 64
"""Decode attends to a multiple of this many cache rows."""


struct Weight(Copyable):
    """A weight matrix, perhaps stored in lower precision."""

    var w: Tensor
    """`float32`, a 16-bit float, or `int8`."""
    var scale: Optional[Tensor]
    """For `int8`: the scale of each output, shaped to broadcast."""

    def __init__(out self, w: Tensor, dtype: DT, out_axis: Int) raises:
        """Converts `w` to `dtype`.

        Args:
            w: The weight, in `float32`.
            dtype: How to store it.
            out_axis: The dimension indexing outputs, which `int8` gives
                a scale each.
        """
        if dtype == DT.int8:
            var qs = quantize_int8(w, out_axis)
            self.w = qs.q
            self.scale = qs.scale
        else:
            self.w = w.cast(dtype).realize()
            self.scale = None

    def __call__(self, x: Tensor) raises -> Tensor:
        """`x @ w`, for weights stored `(in, out)`."""
        var y = x.dot(self.w)
        if self.scale:
            return y * self.scale.value()
        return y


struct Block(Copyable):
    """One transformer layer, with its key-value cache."""

    var ln1_w: Tensor
    var ln1_b: Tensor
    var attn_w: Weight
    """`(n_embd, 3 n_embd)`: queries, keys, values side by side."""
    var attn_b: Tensor
    var proj_w: Weight
    var proj_b: Tensor
    var ln2_w: Tensor
    var ln2_b: Tensor
    var fc_w: Weight
    var fc_b: Tensor
    var fc_proj_w: Weight
    var fc_proj_b: Tensor
    var k_cache: Tensor
    """`(slots * max_context, n_embd)`: the keys of every position so
    far, `max_context` rows per sequence."""
    var v_cache: Tensor
    var max_context: Int

    def __init__(
        out self,
        mut w: Dict[String, Tensor],
        prefix: String,
        max_context: Int,
        slots: Int,
        n_embd: Int,
        dtype: DT,
    ) raises:
        self.ln1_w = w.pop(prefix + "ln_1.weight")
        self.ln1_b = w.pop(prefix + "ln_1.bias")
        self.attn_w = Weight(w.pop(prefix + "attn.c_attn.weight"), dtype, 1)
        self.attn_b = w.pop(prefix + "attn.c_attn.bias")
        self.proj_w = Weight(w.pop(prefix + "attn.c_proj.weight"), dtype, 1)
        self.proj_b = w.pop(prefix + "attn.c_proj.bias")
        self.ln2_w = w.pop(prefix + "ln_2.weight")
        self.ln2_b = w.pop(prefix + "ln_2.bias")
        self.fc_w = Weight(w.pop(prefix + "mlp.c_fc.weight"), dtype, 1)
        self.fc_b = w.pop(prefix + "mlp.c_fc.bias")
        self.fc_proj_w = Weight(w.pop(prefix + "mlp.c_proj.weight"), dtype, 1)
        self.fc_proj_b = w.pop(prefix + "mlp.c_proj.bias")
        # Zeros: rows not yet written are masked, and must not be NaN.
        self.k_cache = (
            Tensor.zeros(slots * max_context, n_embd).contiguous().realize()
        )
        self.v_cache = (
            Tensor.zeros(slots * max_context, n_embd).contiguous().realize()
        )
        self.max_context = max_context

    def prefill(
        self,
        x: Tensor,
        row: Int,
        start: Tensor,
        n_head: Int,
    ) raises -> Tensor:
        """The layer applied to `x`, `(T, n_embd)`: one sequence's
        tokens, whose keys and values go to cache rows `start ...`
        (`start`, a tensor, holds `row`; the sequence's slot begins at a
        multiple of `max_context`). They attend to the slot's rows up to
        their own, causally."""
        var t = x.shape()[0]
        var c = x.shape()[1]
        var hd = c // n_head
        var qkv = x.layernorm(-1, 1e-5) * self.ln1_w + self.ln1_b
        qkv = self.attn_w(qkv) + self.attn_b
        self.k_cache.write_(0, start, qkv.slice(1, c, 2 * c))
        self.v_cache.write_(0, start, qkv.slice(1, 2 * c, 3 * c))
        var q = qkv.slice(1, 0, c).reshape(t, n_head, hd).permute(1, 0, 2)
        var first = row // self.max_context * self.max_context
        var k = self.k_cache.slice(0, first, row + t).reshape(-1, n_head, hd)
        var v = self.v_cache.slice(0, first, row + t).reshape(-1, n_head, hd)
        var a = q.scaled_dot_product_attention(
            k.permute(1, 0, 2), v.permute(1, 0, 2), None, is_causal=True
        )
        return self._finish(x, a.permute(1, 0, 2).reshape(t, c))

    def decode(
        self,
        x: Tensor,
        rows: Tensor,
        n: Int,
        n_head: Int,
        mask: Tensor,
    ) raises -> Tensor:
        """The layer applied to `x`, `(B, n_embd)`: one new token for
        each of the sequences in slots `0 .. B-1`, whose keys and values
        go to cache rows `rows`. Each attends to the first `n` rows of
        its slot, of which `mask`, `(slots, max_context)`, hides those
        not yet written."""
        var b = x.shape()[0]
        var c = x.shape()[1]
        var hd = c // n_head
        var qkv = x.layernorm(-1, 1e-5) * self.ln1_w + self.ln1_b
        qkv = self.attn_w(qkv) + self.attn_b
        self.k_cache.put_rows_(rows, qkv.slice(1, c, 2 * c))
        self.v_cache.put_rows_(rows, qkv.slice(1, 2 * c, 3 * c))
        var q = qkv.slice(1, 0, c).reshape(b, 1, n_head, hd).permute(0, 2, 1, 3)
        var slots = self.k_cache.shape()[0] // self.max_context
        var k = self.k_cache.reshape(slots, self.max_context, n_head, hd)
        var v = self.v_cache.reshape(slots, self.max_context, n_head, hd)
        k = k.slice(0, 0, b).slice(1, 0, n).permute(0, 2, 1, 3)
        v = v.slice(0, 0, b).slice(1, 0, n).permute(0, 2, 1, 3)
        var m = mask.slice(0, 0, b).slice(1, 0, n).reshape(b, 1, 1, n)
        var a = q.scaled_dot_product_attention(k, v, m)
        return self._finish(x, a.permute(0, 2, 1, 3).reshape(b, c))

    def _finish(self, x: Tensor, a: Tensor) raises -> Tensor:
        """The rest of the layer after attention: projection, residual,
        and the MLP."""
        var h = x + (self.proj_w(a) + self.proj_b)
        var m = h.layernorm(-1, 1e-5) * self.ln2_w + self.ln2_b
        m = (self.fc_w(m) + self.fc_b).gelu()
        return h + (self.fc_proj_w(m) + self.fc_proj_b)


struct GPT2(Movable):
    """A GPT-2 model with its key-value caches."""

    var config: GPT2Config
    var wte: Weight
    """The token embedding, `(vocab_size, n_embd)`, also the output
    projection (so its `int8` scales are per token)."""
    var wpe: Tensor
    var blocks: List[Block]
    var lnf_w: Tensor
    var lnf_b: Tensor
    var max_context: Int
    var batch: Int
    """The most sequences generated at once: the caches' slots."""
    var mask: Tensor
    """`(batch * max_context,)`: 0 at the positions a slot's cache
    holds, -inf after. Flat, because only a tensor that owns its buffer
    can be written in place."""
    var jits: Dict[Int, MiniJit]
    """Captured decode steps, one per block of `DECODE_ROWS` positions."""

    def __init__(
        out self,
        path: String,
        max_context: Int = 1024,
        dtype: DT = DT.float32,
        batch: Int = 1,
    ) raises:
        """Loads a model from a Hugging Face `model.safetensors`.

        The sizes are read from the weights, so any of GPT-2's four
        sizes loads.

        Args:
            path: The weights file.
            max_context: The longest sequence to generate; the caches
                take `2 * n_layer * batch * max_context * n_embd` floats.
            dtype: How to store the token embedding and the layers'
                matrices: `float32`, `float16`, `bfloat16`, or `int8`.
                Biases, normalizations, and the position embedding stay
                `float32`.
            batch: The most sequences to generate at once.
        """
        if not (dtype == DT.float32 or dtype.is_half() or dtype == DT.int8):
            raise Error(t"GPT2: cannot store weights as {dtype}")
        var w = safe_load(path, skip_suffix=".attn.bias")
        # Some exports prefix every name with "transformer.".
        if "transformer.wte.weight" in w:
            var renamed = Dict[String, Tensor]()
            for e in w.items():
                if e.key.startswith("transformer."):
                    var short = String(e.key[byte=12:])
                    renamed[short] = e.value
                else:
                    renamed[e.key] = e.value
            w = renamed^
        var n_layer = 0
        while ("h." + String(n_layer) + ".ln_1.weight") in w:
            n_layer += 1
        var wte = w.pop("wte.weight")
        self.wpe = w.pop("wpe.weight")
        var n_embd = wte.shape()[1]
        self.config = GPT2Config(
            n_layer,
            n_embd // 64,
            n_embd,
            wte.shape()[0],
            self.wpe.shape()[0],
        )
        self.wte = Weight(wte, dtype, 0)
        self.max_context = min(max_context, self.config.n_ctx)
        self.blocks = List[Block]()
        for i in range(n_layer):
            self.blocks.append(
                Block(
                    w,
                    "h." + String(i) + ".",
                    self.max_context,
                    batch,
                    n_embd,
                    dtype,
                )
            )
        self.lnf_w = w.pop("ln_f.weight")
        self.lnf_b = w.pop("ln_f.bias")
        self.batch = batch
        self.mask = Tensor.empty([batch * self.max_context])
        self.jits = Dict[Int, MiniJit]()

    def forward(
        self,
        tokens: List[Int],
        start_pos: Int,
        all_positions: Bool = False,
        slot: Int = 0,
    ) raises -> Tensor:
        """The logits of the token after `tokens`, which sit at
        positions `start_pos ...` of the sequence in `slot`; the cache
        must hold every earlier position of it.

        Args:
            tokens: The new tokens.
            start_pos: The position of the first.
            all_positions: Return the logits after every token, not
                just the last (for measuring perplexity).
            slot: Which of the `batch` sequences these tokens belong to.

        Returns:
            A realized `(vocab_size,)` tensor, or `(T, vocab_size)` with
            `all_positions`.
        """
        var t = len(tokens)
        if start_pos + t > self.max_context:
            raise Error(
                t"GPT2: position {start_pos + t} is past the context of"
                t" {self.max_context}"
            )
        if slot < 0 or slot >= self.batch:
            raise Error(t"GPT2: slot {slot} of {self.batch}")
        var positions = List[Int]()
        for i in range(start_pos, start_pos + t):
            positions.append(i)
        var row = slot * self.max_context + start_pos
        if start_pos == 0:
            self.mask.write_(
                0,
                row,
                Tensor.full([self.max_context], -inf[DType.float64]()),
            )
        self.mask.write_(0, row, Tensor.zeros(t))
        var x = self._embed(Tensor(tokens), Tensor(positions))
        var start = Tensor([row])
        for i in range(len(self.blocks)):
            x = self.blocks[i].prefill(x, row, start, self.config.n_head)
        if not all_positions:
            x = x.slice(0, t - 1, t)
        var logits = self._logits(x)
        if not all_positions:
            logits = logits.reshape(self.config.vocab_size)
        return logits.realize()

    def decode(mut self, token: Int, pos: Int) raises -> Tensor:
        """`forward([token], pos)` for the sequence in slot 0, through a
        `MiniJit`; see `decode_batch`.

        Returns:
            The `(vocab_size,)` logits, in a buffer the next decode
            overwrites: read them first.
        """
        return self.decode_batch([token], [pos]).reshape(self.config.vocab_size)

    def decode_batch(
        mut self, tokens: List[Int], positions: List[Int]
    ) raises -> Tensor:
        """One step for the sequences in slots `0 .. B-1`: the logits of
        the token after `tokens[b]`, which sits at `positions[b]` of
        sequence `b`. The sequences may be of different lengths.

        Runs through a `MiniJit` per batch size and block of
        `DECODE_ROWS` positions (of the longest sequence): after two
        steps, a step replays the recorded kernels without building or
        scheduling a graph. The weights are read once for the whole
        batch, so a step of `B` sequences costs little more than one.

        Returns:
            `(B, vocab_size)` logits, in a buffer the next decode
            overwrites: read them first.
        """
        var b = len(tokens)
        if b < 1 or b > self.batch or len(positions) != b:
            raise Error(
                t"GPT2.decode_batch: {b} tokens and {len(positions)}"
                t" positions for {self.batch} slots"
            )
        var longest = 0
        var rows = List[Int]()
        for i in range(b):
            if positions[i] < 1 or positions[i] >= self.max_context:
                raise Error(
                    t"GPT2.decode_batch: position {positions[i]} out of range"
                )
            longest = max(longest, positions[i])
            rows.append(i * self.max_context + positions[i])
        var n = min(
            (longest // DECODE_ROWS + 1) * DECODE_ROWS, self.max_context
        )
        var key = b * 65536 + n
        var jit = self.jits.pop(key, MiniJit())

        def step(xs: List[Tensor]) raises {ref self, var n} -> List[Tensor]:
            self.mask.put_rows_(xs[2], Tensor.zeros(xs[2].shape()[0]))
            var mask = self.mask.reshape(self.batch, self.max_context)
            var x = self._embed(xs[0], xs[1])
            for i in range(len(self.blocks)):
                x = self.blocks[i].decode(x, xs[2], n, self.config.n_head, mask)
            return [self._logits(x)]

        var out = jit(step, [Tensor(tokens), Tensor(positions), Tensor(rows)])
        self.jits[key] = jit^
        return out[0]

    def _embed(self, tokens: Tensor, positions: Tensor) raises -> Tensor:
        """The token plus position embeddings; both are `int32` tensors."""
        var x = self.wte.w.take_rows(tokens).float()
        if self.wte.scale:
            x = x * self.wte.scale.value().take_rows(tokens)
        return x + self.wpe.take_rows(positions)

    def _logits(self, x: Tensor) raises -> Tensor:
        """The final normalization and the output projection."""
        var h = x.layernorm(-1, 1e-5) * self.lnf_w + self.lnf_b
        var logits = h.dot(self.wte.w.T())
        if self.wte.scale:
            logits = logits * self.wte.scale.value().reshape(
                1, self.config.vocab_size
            )
        return logits

    def perplexity(self, tokens: List[Int]) raises -> Float64:
        """The model's perplexity on `tokens`: `exp` of the mean
        negative log-likelihood of each token after the first, given
        those before it. Uses the cache from position 0."""
        var logits = self.forward(tokens, 0, all_positions=True)
        var t = len(tokens)
        var targets = List[Int]()
        for i in range(1, t):
            targets.append(tokens[i])
        var nll = logits.slice(0, 0, t - 1).sparse_categorical_crossentropy(
            Tensor(targets)
        )
        return exp(nll.item())

    def generate(
        mut self,
        prompt: List[Int],
        count: Int,
        temperature: Float64 = 0.0,
        top_k: Int = 0,
    ) raises -> List[Int]:
        """Continues `prompt` by `count` tokens.

        Args:
            prompt: The tokens to start from (at least one).
            count: How many tokens to add.
            temperature: 0 picks the most likely token every time;
                otherwise tokens are sampled from the softmax of
                `logits / temperature`, with the generator
                `Tensor.manual_seed` seeds.
            top_k: When sampling, only the `top_k` most likely tokens
                are considered (0: all).

        Returns:
            The new tokens.
        """
        var out = List[Int]()
        var logits = self.forward(prompt, 0)
        var pos = len(prompt)
        for i in range(count):
            var next = sample(logits.tolist(), temperature, top_k)
            out.append(next)
            if i + 1 < count:
                logits = self.decode(next, pos)
                pos += 1
        return out^

    def generate_batch(
        mut self,
        prompts: List[List[Int]],
        count: Int,
        temperature: Float64 = 0.0,
        top_k: Int = 0,
    ) raises -> List[List[Int]]:
        """Continues each prompt by `count` tokens, all at once.

        Each prompt is processed in its own slot; then every step adds
        one token to every sequence. A sequence that produces
        `<|endoftext|>` keeps going (trim it afterwards).

        Args:
            prompts: Up to `batch` prompts, each at least one token,
                each with room for `count` more within `max_context`.
            count: How many tokens to add to each.
            temperature: As for `generate`.
            top_k: As for `generate`.

        Returns:
            The new tokens of each prompt.
        """
        var b = len(prompts)
        if b < 1 or b > self.batch:
            raise Error(t"GPT2: {b} prompts for {self.batch} slots")
        var out = List[List[Int]]()
        var next = List[Int]()
        var positions = List[Int]()
        for i in range(b):
            var logits = self.forward(prompts[i], 0, slot=i)
            next.append(sample(logits.tolist(), temperature, top_k))
            positions.append(len(prompts[i]))
            out.append([next[i]])
        var v = self.config.vocab_size
        for _ in range(count - 1):
            var all = self.decode_batch(next, positions).tolist()
            for i in range(b):
                var row = List[Float64](capacity=v)
                for j in range(v):
                    row.append(all[i * v + j])
                next[i] = sample(row, temperature, top_k)
                out[i].append(next[i])
                positions[i] += 1
        return out^


def sample(logits: List[Float64], temperature: Float64, top_k: Int) -> Int:
    """A token chosen from `logits`: the most likely at temperature 0,
    else a sample from `softmax(logits / temperature)` over the `top_k`
    most likely (all if 0)."""
    var best = 0
    for i in range(1, len(logits)):
        if logits[i] > logits[best]:
            best = i
    if temperature <= 0.0:
        return best
    var cutoff = -1e300
    if top_k > 0 and top_k < len(logits):
        var sorted = logits.copy()
        sort(sorted)
        cutoff = sorted[len(sorted) - top_k]
    var total = 0.0
    var w = List[Float64](capacity=len(logits))
    for v in logits:
        var p = exp((v - logits[best]) / temperature) if v >= cutoff else 0.0
        w.append(p)
        total += p
    var r = _next_uniform() * total
    for i in range(len(w)):
        r -= w[i]
        if r <= 0.0 and w[i] > 0.0:
            return i
    return best
