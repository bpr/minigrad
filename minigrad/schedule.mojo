"""Turning the lazy graph into kernels and running them
(`minigrad.schedule`).

`realize` takes the nodes whose values are wanted and:

1. sorts the unrealized graph below them;
2. decides which nodes get their own buffer (*kernel roots*): the
   requested nodes, every reduction, `CONTIGUOUS` nodes, the operands of
   a contraction (so the tuned matrix-multiply kernel can run it), and
   any elementwise node that more than one kernel would read (computed
   once, not once per reader);
3. builds one `Kernel` per root by walking down from it. Elementwise
   nodes fuse into the kernel's program. Movement nodes are collected on
   the way down and applied, innermost first, to the identity view of the
   buffer at the bottom, so every load is one strided view;
4. runs the kernels in topological order, each through its cached
   plan, into a new buffer;
5. rewrites each root in place into a `BUFFER` node.

A reshape that is not expressible as one strided view of what lies below
it (see `View.reshape`), or sliding windows over a padded view (see
`View.window`), cannot fuse: the input of that movement is added to the
roots and scheduling starts again.
"""

from std.time import perf_counter_ns

from minigrad.dtype import DT
from minigrad.graph import (
    BytePtr,
    NodeKey,
    graph,
    is_binary,
    is_movement,
    OP_BUFFER,
    OP_CAST,
    OP_CONST,
    OP_CONTIGUOUS,
    OP_CONV,
    OP_DETACH,
    OP_EXPAND,
    OP_FLIP,
    OP_MUL,
    OP_PAD,
    OP_PERMUTE,
    OP_REDUCE,
    OP_RESHAPE,
    OP_SHRINK,
    OP_WHERE,
    OP_WINDOW,
    REDUCE_SUM,
)
from minigrad.runtime.blockops import fill_block
from minigrad.runtime.kernel import (
    Instr,
    K_CONST,
    K_LOAD,
    Kernel,
    CachedKernel,
    JitStep,
    Load,
    STEP_KERNEL,
    Plan,
    SCHEDULE_CACHE_LIMIT,
    make_plan,
    run_plan,
    runtime,
)
from minigrad.uop import UOp
from minigrad.view import View, prod

comptime _MULTI = -2
"""Owner mark: read by more than one kernel."""


struct _Scheduler(Movable):
    var order: List[Int]
    """The unrealized graph below the roots, inputs first."""
    var is_root: List[Bool]
    var owner: List[Int]
    var chain_parent: List[Int]
    """Movement chains as a tree: chain `c` applies the movement node
    `chain_node[c]` first, then chain `chain_parent[c]`. Chain 0 is
    empty."""
    var chain_node: List[Int]
    var chain_index: Dict[Int, Int]
    var memo: Dict[Int, Int]
    var kernels: List[Kernel]
    var keep: List[UOp]
    """Buffers created for padded constants, alive until execution."""
    var fail_node: Int
    """The node to force as a root after a reshape failed to fuse."""

    def __init__(out self):
        self.order = List[Int]()
        self.is_root = List[Bool]()
        self.owner = List[Int]()
        self.chain_parent = List[Int]()
        self.chain_node = List[Int]()
        self.chain_index = Dict[Int, Int]()
        self.memo = Dict[Int, Int]()
        self.kernels = List[Kernel]()
        self.keep = List[UOp]()
        self.fail_node = -1
        self.chain_parent.append(-1)
        self.chain_node.append(-1)

    def group(mut self, roots: List[Int], forced: List[Int]):
        """Sorts the graph below `roots` and marks the kernel roots."""
        var g = graph()
        self.order = g[].toposort(roots)
        var n = len(self.order)
        self.is_root = List[Bool](length=n, fill=False)
        self.owner = List[Int](length=n, fill=-1)
        for r in roots:
            self.is_root[g[].pos[r]] = True
        for f in forced:
            self.is_root[g[].pos[f]] = True
        for i in range(n):
            var op = g[].nodes[self.order[i]].op
            if op == OP_REDUCE or op == OP_CONTIGUOUS:
                self.is_root[i] = True
        # A sum of products whose operands each skip an output dimension
        # is a contraction: give both operands buffers, so the kernel is
        # two loads (each perhaps converted) and the tuned gemm can run it.
        for i in range(n):
            var node = self.order[i]
            if g[].nodes[node].op != OP_REDUCE or (
                g[].nodes[node].arg[0] != REDUCE_SUM
            ):
                continue
            var m = g[].nodes[node].s0
            if g[].nodes[m].op != OP_MUL or not g[].nodes[m].dtype.is_float():
                continue
            if not self._is_contraction(node, m):
                continue
            for j in range(2):
                var s = g[].nodes[m].s0 if j == 0 else g[].nodes[m].s1
                while is_movement(g[].nodes[s].op):
                    s = g[].nodes[s].s0
                if g[].nodes[s].op == OP_CAST:
                    # A buffer stored in another type (bfloat16 weights,
                    # say) stays as it is: gemm converts as it reads.
                    var t = g[].nodes[s].s0
                    while is_movement(g[].nodes[t].op):
                        t = g[].nodes[t].s0
                    if g[].nodes[t].op == OP_BUFFER:
                        continue
                var sop = g[].nodes[s].op
                if sop != OP_BUFFER and sop != OP_CONST:
                    self.is_root[g[].pos[s]] = True
        # Parents before inputs: each node learns which kernel reads it.
        for ii in range(n):
            var i = n - 1 - ii
            var node = self.order[i]
            var op = g[].nodes[node].op
            if op == OP_BUFFER or op == OP_CONST:
                continue
            if not self.is_root[i]:
                # Movement and CONV nodes pass the mark on to what they
                # are made from.
                if (
                    self.owner[i] == _MULTI
                    and not is_movement(op)
                    and op != OP_CONV
                ):
                    self.is_root[i] = True
            var c = i if self.is_root[i] else self.owner[i]
            # A CONV node's other inputs are for the gradient only.
            for j in range(1 if op == OP_CONV else 3):
                var s = g[].nodes[node].s0
                if j == 1:
                    s = g[].nodes[node].s1
                elif j == 2:
                    s = g[].nodes[node].s2
                if s < 0:
                    continue
                var sop = g[].nodes[s].op
                if sop == OP_BUFFER or sop == OP_CONST:
                    continue
                var p = g[].pos[s]
                if self.is_root[p]:
                    continue
                if self.owner[p] == -1:
                    self.owner[p] = c
                elif self.owner[p] != c:
                    self.owner[p] = _MULTI

    def _broadcast_dims(self, node: Int) -> List[Bool]:
        """For each dimension of `node`, whether it is broadcast: every
        element along it is the same. Looks through permutations to the
        first `EXPAND`."""
        var g = graph()
        var n = len(g[].nodes[node].shape)
        var flags = List[Bool](length=n, fill=False)
        var op = g[].nodes[node].op
        if op == OP_EXPAND:
            var s = g[].nodes[node].s0
            for d in range(n):
                flags[d] = g[].nodes[s].shape[d] == 1 and (
                    g[].nodes[node].shape[d] != 1
                )
        elif op == OP_PERMUTE:
            var inner = self._broadcast_dims(g[].nodes[node].s0)
            for d in range(n):
                flags[d] = inner[g[].nodes[node].arg[d]]
        return flags^

    def _is_contraction(self, reduce: Int, mul: Int) -> Bool:
        """Whether `reduce`, a sum of the product `mul`, has the shape
        `C[b,i,j] = sum_k A[b,i,k] B[b,k,j]`: some output dimension is
        broadcast in exactly one operand. With a single row (`i` empty)
        it is a matrix-vector product; dimensions neither operand
        broadcasts (`b`) make a batch of products."""
        var g = graph()
        var bx = self._broadcast_dims(g[].nodes[mul].s0)
        var by = self._broadcast_dims(g[].nodes[mul].s1)
        var n = len(g[].nodes[mul].shape)
        var reduced = List[Bool](length=n, fill=False)
        for j in range(1, len(g[].nodes[reduce].arg)):
            reduced[g[].nodes[reduce].arg[j]] = True
        var only_x = False
        var only_y = False
        for d in range(n):
            if reduced[d] or g[].nodes[mul].shape[d] == 1:
                continue
            if bx[d] and not by[d]:
                only_y = True
            elif by[d] and not bx[d]:
                only_x = True
        # Dimensions neither operand broadcasts are a batch of products.
        return only_x or only_y

    def _chain(mut self, parent: Int, node: Int) -> Int:
        """The chain that applies movement `node`, then chain `parent`."""
        var key = (parent << 32) | node
        var found = self.chain_index.get(key)
        if found:
            return found.value()
        self.chain_parent.append(parent)
        self.chain_node.append(node)
        var c = len(self.chain_node) - 1
        self.chain_index[key] = c
        return c

    def _leaf_view(mut self, leaf: Int, chain: Int) -> Optional[View]:
        """The view through which a kernel reads `leaf` after the
        movements of `chain`, or `None` (setting `fail_node`) when a
        reshape does not fuse."""
        var g = graph()
        var v = View.identity(g[].nodes[leaf].shape)
        var c = chain
        while c != 0:
            var m = self.chain_node[c]
            var op = g[].nodes[m].op
            if op == OP_RESHAPE:
                var r = v.reshape(g[].nodes[m].shape)
                if not r:
                    self.fail_node = g[].nodes[m].s0
                    return None
                v = r.value().copy()
            elif op == OP_PERMUTE:
                v = v.permute(g[].nodes[m].arg)
            elif op == OP_EXPAND:
                v = v.expand(g[].nodes[m].shape)
            elif op == OP_PAD:
                v = v.pad(g[].nodes[m].arg)
            elif op == OP_SHRINK:
                v = v.shrink(g[].nodes[m].arg)
            elif op == OP_FLIP:
                v = v.flip(g[].nodes[m].arg)
            elif op == OP_WINDOW:
                var nk = g[].nodes[m].arg[0]
                var kernel = List[Int]()
                var stride = List[Int]()
                var dilation = List[Int]()
                for j in range(nk):
                    kernel.append(g[].nodes[m].arg[1 + j])
                    stride.append(g[].nodes[m].arg[1 + nk + j])
                    dilation.append(g[].nodes[m].arg[1 + 2 * nk + j])
                var r = v.window(kernel, stride, dilation)
                if not r:
                    self.fail_node = g[].nodes[m].s0
                    return None
                v = r.value().copy()
            c = self.chain_parent[c]
        return v^

    def _load(
        mut self, mut k: Kernel, node: Int, var view: View, dt: DT
    ) -> Int:
        """Appends (or finds) the instruction loading `node` through
        `view`."""
        for i in range(len(k.instrs)):
            if k.instrs[i].op == K_LOAD:
                ref l = k.loads[k.instrs[i].a]
                if l.node == node and l.view == view:
                    return i
        k.loads.append(Load(node, view^, dt))
        k.instrs.append(Instr(K_LOAD, dt, dt, len(k.loads) - 1, -1, -1))
        return len(k.instrs) - 1

    def _emit(mut self, mut k: Kernel, node: Int, chain: Int, top: Bool) -> Int:
        """Appends the instructions computing `node` seen through `chain`
        and returns the index of its value, or -1 if a reshape does not
        fuse."""
        var g = graph()
        var key = (node << 24) | chain
        if not top:
            var found = self.memo.get(key)
            if found:
                return found.value()
        var op = g[].nodes[node].op
        var dt = g[].nodes[node].dtype
        var idx = -1
        if op == OP_BUFFER or (not top and self.is_root[g[].pos[node]]):
            var v = self._leaf_view(node, chain)
            if not v:
                return -1
            idx = self._load(k, node, v.value().copy(), dt)
        elif op == OP_CONST:
            var v = self._leaf_view(node, chain)
            if not v:
                return -1
            var bits = g[].nodes[node].arg[0]
            if v.value().has_mask():
                # A padded constant: read it from a one-element buffer
                # through the masked view.
                var buf = UOp.new_buffer(dt, List[Int]())
                fill_block(dt, graph()[].buffer_ptr(buf.id), bits, 1)
                idx = self._load(k, buf.id, v.value().copy(), dt)
                self.keep.append(buf)
            else:
                for i in range(len(k.instrs)):
                    if (
                        k.instrs[i].op == K_CONST
                        and k.instrs[i].a == bits
                        and k.instrs[i].dtype == dt
                    ):
                        idx = i
                if idx < 0:
                    k.instrs.append(Instr(K_CONST, dt, dt, bits, -1, -1))
                    idx = len(k.instrs) - 1
        elif is_movement(op):
            idx = self._emit(
                k, g[].nodes[node].s0, self._chain(chain, node), False
            )
        elif (
            op == OP_CONTIGUOUS
            or op == OP_DETACH
            or op == OP_REDUCE
            or op == OP_CONV
        ):
            idx = self._emit(k, g[].nodes[node].s0, chain, False)
        else:
            var s0 = g[].nodes[node].s0
            var s1 = g[].nodes[node].s1
            var s2 = g[].nodes[node].s2
            var a = self._emit(k, s0, chain, False)
            if a < 0:
                return -1
            var b = -1
            var c = -1
            if s1 >= 0:
                b = self._emit(k, s1, chain, False)
                if b < 0:
                    return -1
            if s2 >= 0:
                c = self._emit(k, s2, chain, False)
                if c < 0:
                    return -1
            k.instrs.append(Instr(op, dt, graph()[].nodes[s0].dtype, a, b, c))
            idx = len(k.instrs) - 1
        if idx >= 0 and not top:
            self.memo[key] = idx
        return idx

    def build(mut self) -> Bool:
        """Builds a kernel for every root. Returns False, with
        `fail_node` set, if a reshape did not fuse."""
        var g = graph()
        for i in range(len(self.order)):
            if not self.is_root[i]:
                continue
            var node = self.order[i]
            var op = g[].nodes[node].op
            var dt = g[].nodes[node].dtype
            var shape = g[].nodes[node].shape.copy()
            if op == OP_REDUCE:
                shape = g[].nodes[g[].nodes[node].s0].shape.copy()
            var k = Kernel(node, shape^, dt)
            if op == OP_REDUCE:
                k.reduce_op = g[].nodes[node].arg[0]
                for j in range(1, len(g[].nodes[node].arg)):
                    k.reduce_axes.append(g[].nodes[node].arg[j])
            self.memo.clear()
            if self._emit(k, node, 0, True) < 0:
                return False
            self.kernels.append(k^)
        return True

    def execute(mut self) raises -> Optional[List[CachedKernel]]:
        """Runs the kernels, in topological order, and turns their roots
        into buffers.

        Returns:
            The schedule to cache, or `None` if it cannot be replayed
            (it made buffers for padded constants, which are not in the
            graph).
        """
        var g = graph()
        var rt = runtime()
        var cacheable = len(self.keep) == 0
        var cached = List[CachedKernel]()
        var e = g[].epoch
        var ins = List[Int]()
        for ki in range(len(self.kernels)):
            ref k = self.kernels[ki]
            var plan_id = -1
            if prod(k.shape) > 0:
                var key = k.key()
                plan_id = rt[].index.get(key, -1)
                if plan_id < 0:
                    rt[].plans.append(make_plan(k))
                    plan_id = len(rt[].plans) - 1
                    rt[].index[key^] = plan_id
                    g[].stats.kernels_compiled += 1
            var inputs = List[Int]()
            for l in k.loads:
                inputs.append(g[].pos[l.node] if g[].stamp[l.node] == e else -1)
                if g[].stamp[l.node] != e:
                    cacheable = False
            var label = String("")
            if rt[].debug >= 2:
                var red = String("")
                if k.reduce_op >= 0:
                    red = String(t" reduce {k.reduce_axes}")
                label = String(
                    t"{k.dtype} {k.shape}{red}  loads {len(k.loads)} instrs"
                    t" {len(k.instrs)}"
                )
            var ck = CachedKernel(
                plan_id,
                g[].pos[k.root],
                inputs^,
                k.out_size() * k.dtype.itemsize(),
                k.instrs.copy(),
                label^,
            )
            ins.clear()
            for l in k.loads:
                ins.append(g[].nodes[l.node].arg[0])
            _run(ck, k.root, ins)
            cached.append(ck^)
        self.keep.clear()
        if cacheable:
            return cached^
        return None


def _run(ck: CachedKernel, root: Int, in_bufs: List[Int]) raises:
    """Runs one kernel, reading buffers `in_bufs`, into a new buffer, and
    makes `root` that buffer. A `MiniJit` capture records it."""
    var g = graph()
    var rt = runtime()
    var out_buf = g[].alloc_buffer(ck.nbytes)
    if rt[].recording:
        for b in in_bufs:
            g[].hold(b)
        g[].hold(out_buf)
        rt[].tape.append(
            JitStep(
                STEP_KERNEL, ck.plan, ck.instrs.copy(), in_bufs.copy(), out_buf
            )
        )
    if ck.plan >= 0:
        var t0 = perf_counter_ns() if rt[].debug >= 2 else 0
        ref plan = rt[].plans[ck.plan]
        var ins = List[BytePtr](capacity=len(in_bufs))
        for b in in_bufs:
            ins.append(g[].buffers[b].ptr)
        run_plan(plan, g[].buffers[out_buf].ptr, ins, ck.instrs)
        g[].stats.kernels_run += 1
        if plan.gemm:
            g[].stats.gemm_kernels += 1
        if rt[].debug >= 2:
            var kind = String("gemm  ") if plan.gemm else String("blocks")
            print(
                t"{kind} {Float64(perf_counter_ns() - t0) / 1e3} us "
                t" {ck.label}  loops {plan.sizes}"
            )
    g[].become_buffer(root, out_buf)


def _signature(order: List[Int], roots: List[Int]) -> NodeKey:
    """The structure of the graph `order` (a topological order) holds:
    each node's operation, element type, shape, argument, and inputs by
    position, without the identity of its buffers. Graphs with equal
    signatures get the same schedule."""
    var g = graph()
    var d = List[Int](capacity=12 * len(order))
    for n in order:
        ref node = g[].nodes[n]
        d.append(node.op)
        d.append(node.dtype.code)
        d.append(g[].pos[node.s0] if node.s0 >= 0 else -1)
        d.append(g[].pos[node.s1] if node.s1 >= 0 else -1)
        d.append(g[].pos[node.s2] if node.s2 >= 0 else -1)
        d.append(len(node.shape))
        for s in node.shape:
            d.append(s)
        if node.op == OP_BUFFER:
            d.append(-1)  # which buffer does not matter
        else:
            d.append(len(node.arg))
            for a in node.arg:
                d.append(a)
    d.append(-2)
    for r in roots:
        d.append(g[].pos[r])
    return NodeKey(d^)


def realize(roots: List[UOp]) raises:
    """Computes the values of `roots`: afterwards each is a buffer.

    Args:
        roots: The nodes to realize.

    """
    var g = graph()
    var ids = List[Int]()
    for r in roots:
        if not r.is_realized() and r.id not in ids:
            ids.append(r.id)
    if len(ids) == 0:
        return
    g[].stats.schedules += 1
    var rt = runtime()
    var t0 = perf_counter_ns()
    # A graph of a structure scheduled before replays that schedule:
    # only which buffers it reads differs.
    var key: Optional[NodeKey] = None
    if rt[].cache_schedules:
        var order = g[].toposort(ids)
        var sig = _signature(order, ids)
        var found = rt[].schedule_index.get(sig, -1)
        if found >= 0:
            var t1 = perf_counter_ns()
            g[].stats.ns_group += Int(t1 - t0)
            g[].stats.schedule_hits += 1
            var ins = List[Int]()
            for i in range(len(rt[].schedules[found])):
                ref ck = rt[].schedules[found][i]
                ins.clear()
                for p in ck.inputs:
                    ins.append(g[].nodes[order[p]].arg[0])
                _run(ck, order[ck.root], ins)
            g[].stats.ns_execute += Int(perf_counter_ns() - t1)
            return
        key = sig^
    var forced = List[Int]()
    while True:
        var s = _Scheduler()
        var t1 = perf_counter_ns()
        s.group(ids, forced)
        var t2 = perf_counter_ns()
        var built = s.build()
        var t3 = perf_counter_ns()
        g[].stats.ns_group += Int(t2 - t0)
        g[].stats.ns_build += Int(t3 - t2)
        if built:
            var cached = s.execute()
            g[].stats.ns_execute += Int(perf_counter_ns() - t3)
            if cached and key:
                if len(rt[].schedules) >= SCHEDULE_CACHE_LIMIT:
                    rt[].schedules.clear()
                    rt[].schedule_index.clear()
                rt[].schedules.append(cached.value().copy())
                rt[].schedule_index[key.value().copy()] = (
                    len(rt[].schedules) - 1
                )
            return
        forced.append(s.fail_node)
        t0 = t1
