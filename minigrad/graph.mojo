"""The lazy operation graph (`minigrad.graph`).

Every tensor operation appends a `Node` to one process-wide arena, the
`Graph`. Nodes are addressed by integer id, name their inputs by id, and
are hash-consed: building the same operation on the same inputs twice
returns the same id, so common subexpressions are shared as they are
built. The exceptions are `BUFFER` and `CONTIGUOUS` nodes, each of which
is its own array: `contiguous()` is how to get a tensor that is distinct
from every other tensor with the same value. Reference counts (held by `UOp` handles and by parent nodes) free
a node's slot for reuse when nothing refers to it.

Realizing a node rewrites it in place into a `BUFFER` node that owns
memory; nodes that used it as an input then read the buffer, with no
substitution pass. A buffer's contents never change after that, which is
what makes sharing nodes sound: there is no in-place assignment.
"""

from std.ffi import _Global
from std.memory.alloc import unsafe_alloc
from std.os import abort

from minigrad.dtype import DT

comptime BytePtr = Pointer[Byte, MutUntrackedOrigin]

# ---- operation codes

comptime OP_BUFFER = 0
"""A realized array; `arg[0]` is its buffer id."""
comptime OP_CONST = 1
"""A scalar constant; `arg[0]` holds the value's bits."""

comptime OP_RESHAPE = 2
comptime OP_PERMUTE = 3
comptime OP_EXPAND = 4
comptime OP_PAD = 5
comptime OP_SHRINK = 6
comptime OP_FLIP = 7
comptime OP_WINDOW = 8
"""Sliding windows over the last dimensions; `arg` is the number of
window dimensions, then their kernel sizes, strides, and dilations."""

comptime OP_NEG = 10
comptime OP_EXP = 11
comptime OP_LOG = 12
comptime OP_SIN = 13
comptime OP_SQRT = 14
comptime OP_RECIP = 15
comptime OP_TRUNC = 16
comptime OP_CAST = 17

comptime OP_ADD = 20
comptime OP_SUB = 21
comptime OP_MUL = 22
comptime OP_DIV = 23
comptime OP_MAX = 24
comptime OP_MOD = 25
comptime OP_POW = 26
comptime OP_CMPLT = 27
comptime OP_CMPNE = 28
comptime OP_CMPEQ = 29
comptime OP_AND = 30
comptime OP_OR = 31
comptime OP_XOR = 32

comptime OP_WHERE = 40

comptime OP_REDUCE = 50
"""A reduction keeping dimensions; `arg[0]` is `REDUCE_SUM` or
`REDUCE_MAX` and `arg[1:]` are the reduced axes."""

comptime OP_CONTIGUOUS = 60
comptime OP_DETACH = 62
comptime OP_CONV = 63
"""The value of `s0`, which other operations compute as the convolution
of `s1` by the weight `s2`; `arg` is the groups, stride, dilation, and
padding. The scheduler looks through it. It exists for `gradient`, which
differentiates it as a convolution instead of differentiating `s0`."""

comptime REDUCE_SUM = 0
comptime REDUCE_MAX = 1


def is_movement(op: Int) -> Bool:
    """Whether `op` only re-indexes its input."""
    return op >= OP_RESHAPE and op <= OP_WINDOW


def is_unary(op: Int) -> Bool:
    """Whether `op` is an elementwise operation of one input."""
    return op >= OP_NEG and op <= OP_CAST


def is_binary(op: Int) -> Bool:
    """Whether `op` is an elementwise operation of two inputs."""
    return op >= OP_ADD and op <= OP_XOR


def is_elementwise(op: Int) -> Bool:
    """Whether `op` computes each output element from the same element of
    its inputs."""
    return (op >= OP_NEG and op <= OP_WHERE) or op == OP_DETACH


def op_name(op: Int) -> String:
    """The name of an operation code, for printing."""
    var names: List[String] = [
        "BUFFER",
        "CONST",
        "RESHAPE",
        "PERMUTE",
        "EXPAND",
        "PAD",
        "SHRINK",
        "FLIP",
        "WINDOW",
        "?",
        "NEG",
        "EXP",
        "LOG",
        "SIN",
        "SQRT",
        "RECIP",
        "TRUNC",
        "CAST",
        "?",
        "?",
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "MAX",
        "MOD",
        "POW",
        "CMPLT",
        "CMPNE",
        "CMPEQ",
        "AND",
        "OR",
        "XOR",
    ]
    if op >= 0 and op <= OP_XOR:
        return names[op]
    if op == OP_WHERE:
        return "WHERE"
    if op == OP_REDUCE:
        return "REDUCE"
    if op == OP_CONTIGUOUS:
        return "CONTIGUOUS"
    if op == OP_DETACH:
        return "DETACH"
    if op == OP_CONV:
        return "CONV"
    return "?"


# ---- nodes and buffers


struct Node(Movable):
    """One operation in the graph."""

    var op: Int
    """An `OP_*` code; -1 for a free slot."""
    var dtype: DT
    """The element type of the result."""
    var rc: Int
    """References from `UOp` handles and from parent nodes."""
    var s0: Int
    """The first input's id, or -1."""
    var s1: Int
    """The second input's id, or -1."""
    var s2: Int
    """The third input's id, or -1."""
    var shape: List[Int]
    """The shape of the result."""
    var arg: List[Int]
    """Operation-specific integers (axes, padding, constant bits, ...)."""

    def __init__(
        out self,
        op: Int,
        dtype: DT,
        var shape: List[Int],
        s0: Int,
        s1: Int,
        s2: Int,
        var arg: List[Int],
    ):
        self.op = op
        self.dtype = dtype
        self.rc = 0
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.shape = shape^
        self.arg = arg^


struct NodeKey(Copyable, Equatable, Hashable):
    """The identity of a node for hash-consing: its operation, element
    type, inputs, and argument, flattened."""

    var data: List[Int]

    def __init__(out self, var data: List[Int]):
        self.data = data^


struct Buffer(Movable):
    """A block of memory holding a realized array."""

    var ptr: BytePtr
    """The first byte."""
    var nbytes: Int
    """The size in bytes (0 for a free slot)."""
    var rc: Int
    """The number of `BUFFER` nodes that share this memory."""

    def __init__(out self, ptr: BytePtr, nbytes: Int):
        self.ptr = ptr
        self.nbytes = nbytes
        self.rc = 0


struct GradSlot(Movable):
    """A tensor that requires a gradient, as `backward` sees it."""

    var node: Int
    """The tensor's current node id, or -1 for a free slot."""
    var grad: Int
    """The node id of the accumulated gradient, or -1."""

    def __init__(out self, node: Int):
        self.node = node
        self.grad = -1


struct Stats(Copyable):
    """Counters for inspecting what the engine did."""

    var nodes_created: Int
    """Nodes appended to the graph (hash-cons misses)."""
    var nodes_reused: Int
    """Hash-cons hits."""
    var kernels_run: Int
    """Kernels executed."""
    var kernels_compiled: Int
    """Kernels lowered to an execution plan (plan cache misses)."""
    var schedules: Int
    """Calls to the scheduler."""
    var gemm_kernels: Int
    """Kernels run by the tuned matrix-multiply kernel."""
    var schedule_hits: Int
    """Calls to the scheduler answered from the schedule cache."""
    var ns_group: Int
    """Nanoseconds spent sorting the graph and choosing kernel roots."""
    var ns_build: Int
    """Nanoseconds spent building kernels from the graph."""
    var ns_execute: Int
    """Nanoseconds spent running kernels (including plan lookups)."""

    def __init__(out self):
        self.nodes_created = 0
        self.nodes_reused = 0
        self.kernels_run = 0
        self.kernels_compiled = 0
        self.schedules = 0
        self.gemm_kernels = 0
        self.schedule_hits = 0
        self.ns_group = 0
        self.ns_build = 0
        self.ns_execute = 0


comptime POOL_LIMIT_BYTES = 1 << 31
"""Freed buffer memory kept for reuse before it goes back to the system."""


struct Graph(Movable):
    """The arena of nodes and buffers behind every tensor."""

    var nodes: List[Node]
    var free_nodes: List[Int]
    var table: Dict[NodeKey, Int]
    var buffers: List[Buffer]
    var free_buffers: List[Int]
    var pool: Dict[Int, List[BytePtr]]
    """Freed allocations by size, reused by `alloc_buffer`."""
    var pool_bytes: Int
    var stamp: List[Int]
    """Per-node visit marks for traversals; see `next_epoch`."""
    var epoch: Int
    var pos: List[Int]
    """Per-node index into the last `toposort` result; valid for nodes
    whose `stamp` is the current `epoch`."""
    var grad_slots: List[GradSlot]
    var free_slots: List[Int]
    var training: Bool
    """Whether layers such as dropout are in training mode."""
    var rng_state: UInt64
    var stats: Stats

    def __init__(out self):
        self.nodes = List[Node]()
        self.free_nodes = List[Int]()
        self.table = Dict[NodeKey, Int]()
        self.buffers = List[Buffer]()
        self.free_buffers = List[Int]()
        self.pool = Dict[Int, List[BytePtr]]()
        self.pool_bytes = 0
        self.stamp = List[Int]()
        self.epoch = 0
        self.pos = List[Int]()
        self.grad_slots = List[GradSlot]()
        self.free_slots = List[Int]()
        self.training = False
        self.rng_state = 0x9E3779B97F4A7C15
        self.stats = Stats()

    # ---- nodes

    @staticmethod
    def _key(
        op: Int, dtype: DT, s0: Int, s1: Int, s2: Int, arg: List[Int]
    ) -> NodeKey:
        var data = List[Int](capacity=5 + len(arg))
        data.append(op)
        data.append(dtype.code)
        data.append(s0)
        data.append(s1)
        data.append(s2)
        for a in arg:
            data.append(a)
        return NodeKey(data^)

    def add(
        mut self,
        op: Int,
        dtype: DT,
        var shape: List[Int],
        s0: Int,
        s1: Int,
        s2: Int,
        var arg: List[Int],
    ) -> Int:
        """Returns the id of the node for this operation, creating it if
        no identical node is alive.

        The result's reference count is not incremented: wrap it in a
        `UOp` at once.

        Args:
            op: The operation code.
            dtype: The element type of the result.
            shape: The shape of the result.
            s0: The first input's id, or -1.
            s1: The second input's id, or -1.
            s2: The third input's id, or -1.
            arg: Operation-specific integers.

        Returns:
            The node id.
        """
        if op != OP_BUFFER and op != OP_CONTIGUOUS:
            var key = Self._key(op, dtype, s0, s1, s2, arg)
            var found = self.table.get(key)
            if found:
                self.stats.nodes_reused += 1
                return found.value()
            var id = self._new_slot(Node(op, dtype, shape^, s0, s1, s2, arg^))
            self.table[key^] = id
            return id
        return self._new_slot(Node(op, dtype, shape^, s0, s1, s2, arg^))

    def _new_slot(mut self, var node: Node) -> Int:
        self.stats.nodes_created += 1
        if node.s0 >= 0:
            self.nodes[node.s0].rc += 1
        if node.s1 >= 0:
            self.nodes[node.s1].rc += 1
        if node.s2 >= 0:
            self.nodes[node.s2].rc += 1
        if len(self.free_nodes) > 0:
            var id = self.free_nodes.pop()
            self.nodes[id] = node^
            return id
        self.nodes.append(node^)
        self.stamp.append(0)
        self.pos.append(0)
        return len(self.nodes) - 1

    def incref(mut self, id: Int):
        """Adds a reference to node `id`."""
        self.nodes[id].rc += 1

    def decref(mut self, id: Int):
        """Drops a reference to node `id`, freeing it and any inputs that
        become unreferenced."""
        self.nodes[id].rc -= 1
        if self.nodes[id].rc > 0:
            return
        var stack = List[Int]()
        stack.append(id)
        while len(stack) > 0:
            var n = stack.pop()
            self._unlink(n, stack)
            self.nodes[n].op = -1
            self.nodes[n].shape.clear()
            self.nodes[n].arg.clear()
            self.free_nodes.append(n)

    def _unlink(mut self, n: Int, mut dead: List[Int]):
        """Removes node `n` from the hash-cons table and drops what it
        refers to, appending inputs that died to `dead`."""
        var op = self.nodes[n].op
        if op == OP_BUFFER:
            self.release_buffer(self.nodes[n].arg[0])
        elif op != OP_CONTIGUOUS:
            var key = Self._key(
                op,
                self.nodes[n].dtype,
                self.nodes[n].s0,
                self.nodes[n].s1,
                self.nodes[n].s2,
                self.nodes[n].arg,
            )
            _ = self.table.pop(key, -1)
        for k in range(3):
            var s = self.nodes[n].s0
            if k == 1:
                s = self.nodes[n].s1
            elif k == 2:
                s = self.nodes[n].s2
            if s >= 0:
                self.nodes[s].rc -= 1
                if self.nodes[s].rc == 0:
                    dead.append(s)
        self.nodes[n].s0 = -1
        self.nodes[n].s1 = -1
        self.nodes[n].s2 = -1

    def become_buffer(mut self, n: Int, buffer: Int):
        """Rewrites node `n` in place into a `BUFFER` node sharing
        `buffer`: the node has been realized."""
        var dead = List[Int]()
        self._unlink(n, dead)
        self.buffers[buffer].rc += 1
        self.nodes[n].op = OP_BUFFER
        self.nodes[n].arg.clear()
        self.nodes[n].arg.append(buffer)
        while len(dead) > 0:
            var d = dead.pop()
            self._unlink(d, dead)
            self.nodes[d].op = -1
            self.nodes[d].shape.clear()
            self.nodes[d].arg.clear()
            self.free_nodes.append(d)

    def next_epoch(mut self) -> Int:
        """Starts a traversal: returns a mark no entry of `stamp` holds."""
        self.epoch += 1
        return self.epoch

    def toposort(mut self, roots: List[Int]) -> List[Int]:
        """The nodes reachable from `roots`, each after its inputs.

        Until the next traversal, `pos[n]` is the index of node `n` in
        the result, for the nodes `n` with `stamp[n] == epoch`.

        Args:
            roots: The node ids to start from.

        Returns:
            Node ids in topological order.
        """
        var e = self.next_epoch()
        var order = List[Int]()
        var stack = List[Int]()
        for r in roots:
            if self.stamp[r] == e:
                continue
            stack.append(r)
            while len(stack) > 0:
                var n = stack[len(stack) - 1]
                if self.stamp[n] == e:
                    _ = stack.pop()
                    continue
                # Push unvisited inputs; emit the node once all are done.
                var pending = False
                for k in range(3):
                    var s = self.nodes[n].s0
                    if k == 1:
                        s = self.nodes[n].s1
                    elif k == 2:
                        s = self.nodes[n].s2
                    if s >= 0 and self.stamp[s] != e:
                        stack.append(s)
                        pending = True
                if not pending:
                    self.stamp[n] = e
                    self.pos[n] = len(order)
                    order.append(n)
                    _ = stack.pop()
        return order^

    # ---- buffers

    def alloc_buffer(mut self, nbytes: Int) -> Int:
        """Allocates at least `nbytes` of uninitialized, 64-byte-aligned
        memory, padded by 64 bytes.

        Returns:
            The buffer id, with a reference count of zero.
        """
        # Whole cache lines, plus one the SIMD loops may read or write
        # past the last element.
        var size = ((nbytes + 63) // 64) * 64 + 64
        var pooled: Optional[BytePtr] = None
        if size in self.pool:
            try:
                ref bucket = self.pool[size]
                if len(bucket) > 0:
                    pooled = bucket.pop()
                    self.pool_bytes -= size
            except:
                pass
        var ptr = pooled.value() if pooled else unsafe_alloc[Byte](
            size, alignment=64
        )
        if len(self.free_buffers) > 0:
            var id = self.free_buffers.pop()
            self.buffers[id].ptr = ptr
            self.buffers[id].nbytes = size
            self.buffers[id].rc = 0
            return id
        self.buffers.append(Buffer(ptr, size))
        return len(self.buffers) - 1

    def release_buffer(mut self, id: Int):
        """Drops a reference to buffer `id`, recycling its memory when it
        was the last."""
        self.buffers[id].rc -= 1
        if self.buffers[id].rc > 0:
            return
        var size = self.buffers[id].nbytes
        var ptr = self.buffers[id].ptr
        self.buffers[id].nbytes = 0
        self.free_buffers.append(id)
        if self.pool_bytes + size > POOL_LIMIT_BYTES:
            ptr.unsafe_free()
            return
        self.pool_bytes += size
        if size not in self.pool:
            self.pool[size] = List[BytePtr]()
        try:
            self.pool[size].append(ptr)
        except:
            ptr.unsafe_free()

    def buffer_ptr(self, node: Int) -> BytePtr:
        """The memory of `BUFFER` node `node`."""
        return self.buffers[self.nodes[node].arg[0]].ptr

    # ---- gradient slots

    def new_grad_slot(mut self, node: Int) -> Int:
        """Registers a tensor that requires a gradient."""
        if len(self.free_slots) > 0:
            var s = self.free_slots.pop()
            self.grad_slots[s].node = node
            self.grad_slots[s].grad = -1
            return s
        self.grad_slots.append(GradSlot(node))
        return len(self.grad_slots) - 1

    def free_grad_slot(mut self, slot: Int):
        """Unregisters a gradient slot, dropping its gradient."""
        var g = self.grad_slots[slot].grad
        self.grad_slots[slot].node = -1
        self.grad_slots[slot].grad = -1
        self.free_slots.append(slot)
        if g >= 0:
            self.decref(g)

    def live_nodes(self) -> Int:
        """The number of nodes in use."""
        return len(self.nodes) - len(self.free_nodes)

    def live_buffers(self) -> Int:
        """The number of buffers in use."""
        return len(self.buffers) - len(self.free_buffers)


def _init_graph() -> Graph:
    return Graph()


comptime _GRAPH = _Global["minigrad_graph", _init_graph]


@always_inline
def graph() -> Pointer[Graph, MutUntrackedOrigin]:
    """The process-wide graph. Not thread safe: build tensors on one
    thread."""
    try:
        return _GRAPH.get_or_create_ptr()
    except:
        abort("minigrad: cannot create the graph")
