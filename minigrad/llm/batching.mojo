"""Continuous batching for GPT-2 (`minigrad.llm.batching`).

A `Batcher` serves a stream of generation requests with one model. It
keeps up to `model.batch` requests running, one per cache slot, and
advances all of them by a token per step; a request that finishes frees
its slot, and a waiting one takes it at the next step. Requests can be
submitted at any time.

```mojo
var b = Batcher(GPT2("data/gpt2/model.safetensors", batch=8))
var a = b.submit(prompt_a, 40)
var c = b.submit(prompt_c, 10)
while b.step():
    pass                   # submit more here as they arrive
print(b.output(a))
```

A step advances every slot through `GPT2.decode_batch`, so its shape
changes only with the longest position and `MiniJit` needs one capture
per block of 64 positions. Free slots compute a dummy token at their own
position 1, which the next prompt admitted to that slot overwrites.
"""

from std.collections import Deque

from minigrad.llm.bpe import BPETokenizer
from minigrad.llm.gpt2 import GPT2, sample


struct Batcher(Movable):
    """Runs generation requests together; see the module docstring."""

    var model: GPT2
    var temperature: Float64
    var top_k: Int
    var stop_at_eot: Bool
    """Whether a request ends when it produces `<|endoftext|>`."""
    var prompts: List[List[Int]]
    """Every request's prompt, by request id."""
    var limits: List[Int]
    """Every request's most new tokens."""
    var outputs: List[List[Int]]
    """Every request's new tokens so far."""
    var done: List[Bool]
    var waiting: Deque[Int]
    var slot_request: List[Int]
    """The request in each slot, or -1."""
    var slot_pos: List[Int]
    """The position of each slot's next token."""

    def __init__(
        out self,
        var model: GPT2,
        temperature: Float64 = 0.0,
        top_k: Int = 0,
        stop_at_eot: Bool = True,
    ):
        """A batcher with `model.batch` slots.

        Args:
            model: The model; its `batch` sets how many requests run at
                once.
            temperature: As for `GPT2.generate`, for every request.
            top_k: As for `GPT2.generate`.
            stop_at_eot: End a request at `<|endoftext|>` (kept in its
                output).
        """
        var slots = model.batch
        self.model = model^
        self.temperature = temperature
        self.top_k = top_k
        self.stop_at_eot = stop_at_eot
        self.prompts = List[List[Int]]()
        self.limits = List[Int]()
        self.outputs = List[List[Int]]()
        self.done = List[Bool]()
        self.waiting = Deque[Int]()
        self.slot_request = List[Int](length=slots, fill=-1)
        self.slot_pos = List[Int](length=slots, fill=0)

    def submit(mut self, prompt: List[Int], max_new_tokens: Int) raises -> Int:
        """Queues a request.

        Args:
            prompt: At least one token, with room after it within the
                model's context.
            max_new_tokens: The most tokens to generate (at least 1).

        Returns:
            The request's id, for `output` and `finished`.
        """
        if len(prompt) < 1 or len(prompt) >= self.model.max_context:
            raise Error(
                t"Batcher: a prompt of {len(prompt)} tokens for a context of"
                t" {self.model.max_context}"
            )
        if max_new_tokens < 1:
            raise Error("Batcher: max_new_tokens must be at least 1")
        self.prompts.append(prompt.copy())
        self.limits.append(max_new_tokens)
        self.outputs.append(List[Int]())
        self.done.append(False)
        var id = len(self.prompts) - 1
        self.waiting.append(id)
        return id

    def output(self, id: Int) -> List[Int]:
        """The tokens request `id` has generated so far."""
        return self.outputs[id].copy()

    def finished(self, id: Int) -> Bool:
        """Whether request `id` is complete."""
        return self.done[id]

    def active(self) -> Int:
        """The requests running now."""
        var n = 0
        for r in self.slot_request:
            if r >= 0:
                n += 1
        return n

    def step(mut self) raises -> Bool:
        """Admits waiting requests to free slots, then advances every
        running request by one token.

        Returns:
            Whether any request is still running or waiting.
        """
        # Admit: process each new prompt into the lowest free slot; its
        # logits give its first token.
        for s in range(len(self.slot_request)):
            if self.slot_request[s] >= 0 or len(self.waiting) == 0:
                continue
            var id = self.waiting.popleft()
            var logits = self.model.forward(self.prompts[id], 0, slot=s)
            self.slot_request[s] = id
            self.slot_pos[s] = len(self.prompts[id])
            self._emit(s, sample(logits.tolist(), self.temperature, self.top_k))
        # Advance every slot.
        var h = len(self.slot_request)
        if self.active() > 0:
            var tokens = List[Int](capacity=h)
            var positions = List[Int](capacity=h)
            for s in range(h):
                var id = self.slot_request[s]
                if id >= 0:
                    tokens.append(self.outputs[id][len(self.outputs[id]) - 1])
                    positions.append(self.slot_pos[s])
                else:
                    tokens.append(0)
                    positions.append(1)
            var all = self.model.decode_batch(tokens, positions).tolist()
            var v = self.model.config.vocab_size
            for s in range(h):
                if self.slot_request[s] < 0:
                    continue
                var row = List[Float64](capacity=v)
                for j in range(v):
                    row.append(all[s * v + j])
                self.slot_pos[s] += 1
                self._emit(s, sample(row, self.temperature, self.top_k))
        return self.active() > 0 or len(self.waiting) > 0

    def _emit(mut self, s: Int, token: Int):
        """Appends `token` to the request in slot `s`, freeing the slot
        if that completes it."""
        var id = self.slot_request[s]
        self.outputs[id].append(token)
        if (
            len(self.outputs[id]) >= self.limits[id]
            or (self.stop_at_eot and token == BPETokenizer.EOT)
            or self.slot_pos[s] >= self.model.max_context
        ):
            self.done[id] = True
            self.slot_request[s] = -1
