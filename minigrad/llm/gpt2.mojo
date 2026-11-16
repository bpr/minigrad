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
    """`(max_context, n_embd)`: the keys of every position so far."""
    var v_cache: Tensor

    def __init__(
        out self,
        mut w: Dict[String, Tensor],
        prefix: String,
        max_context: Int,
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
        self.k_cache = Tensor.zeros(max_context, n_embd).contiguous().realize()
        self.v_cache = Tensor.zeros(max_context, n_embd).contiguous().realize()

    def __call__(
        self, x: Tensor, start_pos: Int, n_head: Int, mask: Tensor
    ) raises -> Tensor:
        """The layer applied to `x`, `(T, n_embd)`: the tokens at
        positions `start_pos ...`. Writes their keys and values into the
        cache. `mask` is 0 for the cache's written rows and -inf for the
        rest."""
        var t = x.shape()[0]
        var c = x.shape()[1]
        var hd = c // n_head
        var qkv = x.layernorm(-1, 1e-5) * self.ln1_w + self.ln1_b
        qkv = self.attn_w(qkv) + self.attn_b
        self.k_cache.write_(0, start_pos, qkv.slice(1, c, 2 * c))
        self.v_cache.write_(0, start_pos, qkv.slice(1, 2 * c, 3 * c))
        var q = qkv.slice(1, 0, c).reshape(t, n_head, hd).permute(1, 0, 2)
        var a: Tensor
        if t == 1:
            var n = min(
                (start_pos // DECODE_ROWS + 1) * DECODE_ROWS,
                self.k_cache.shape()[0],
            )
            var k = self.k_cache.slice(0, 0, n).reshape(n, n_head, hd)
            var v = self.v_cache.slice(0, 0, n).reshape(n, n_head, hd)
            a = q.scaled_dot_product_attention(
                k.permute(1, 0, 2), v.permute(1, 0, 2), mask.slice(0, 0, n)
            )
        else:
            var l = start_pos + t
            var k = self.k_cache.slice(0, 0, l).reshape(l, n_head, hd)
            var v = self.v_cache.slice(0, 0, l).reshape(l, n_head, hd)
            a = q.scaled_dot_product_attention(
                k.permute(1, 0, 2), v.permute(1, 0, 2), None, is_causal=True
            )
        a = a.permute(1, 0, 2).reshape(t, c)
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
    var mask: Tensor
    """`(max_context,)`: 0 at the positions the caches hold, -inf after."""

    def __init__(
        out self,
        path: String,
        max_context: Int = 1024,
        dtype: DT = DT.float32,
    ) raises:
        """Loads a model from a Hugging Face `model.safetensors`.

        The sizes are read from the weights, so any of GPT-2's four
        sizes loads.

        Args:
            path: The weights file.
            max_context: The longest sequence to generate; the caches
                take `2 * n_layer * max_context * n_embd` floats.
            dtype: How to store the token embedding and the layers'
                matrices: `float32`, `float16`, `bfloat16`, or `int8`.
                Biases, normalizations, and the position embedding stay
                `float32`.
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
                    w, "h." + String(i) + ".", self.max_context, n_embd, dtype
                )
            )
        self.lnf_w = w.pop("ln_f.weight")
        self.lnf_b = w.pop("ln_f.bias")
        self.mask = Tensor.empty([self.max_context])

    def forward(
        self, tokens: List[Int], start_pos: Int, all_positions: Bool = False
    ) raises -> Tensor:
        """The logits of the token after `tokens`, which sit at
        positions `start_pos ...`; the cache must hold every earlier
        position.

        Args:
            tokens: The new tokens.
            start_pos: The position of the first.
            all_positions: Return the logits after every token, not
                just the last (for measuring perplexity).

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
        var x = self.wte.w.take_rows(tokens).float()
        if self.wte.scale:
            x = x * self.wte.scale.value().take_rows(tokens)
        var positions = List[Int]()
        for i in range(start_pos, start_pos + t):
            positions.append(i)
        x = x + self.wpe.take_rows(positions)
        if start_pos == 0:
            self.mask.write_(
                0, 0, Tensor.full([self.max_context], -inf[DType.float64]())
            )
        self.mask.write_(0, start_pos, Tensor.zeros(t))
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, start_pos, self.config.n_head, self.mask)
        if not all_positions:
            x = x.slice(0, t - 1, t)
        x = x.layernorm(-1, 1e-5) * self.lnf_w + self.lnf_b
        var logits = x.dot(self.wte.w.T())
        if self.wte.scale:
            logits = logits * self.wte.scale.value().reshape(
                1, self.config.vocab_size
            )
        if not all_positions:
            logits = logits.reshape(self.config.vocab_size)
        return logits.realize()

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
        self,
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
                logits = self.forward([next], pos)
                pos += 1
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
