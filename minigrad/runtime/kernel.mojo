"""Kernels, their execution plans, and the CPU executor
(`minigrad.runtime.kernel`).

A `Kernel` is what the scheduler fuses out of the graph: one elementwise
expression over strided loads, optionally reduced, written to one output
buffer. It is a short program in SSA form and knows nothing about loops.

`make_plan` lowers a kernel to a `Plan`: it drops size-1 dimensions,
merges dimensions that every load walks contiguously, picks what to
vectorize over (one long dimension, or several short trailing ones
flattened through offset tables, whichever a cost estimate prefers), and
assigns block temporaries to instructions. A kernel shaped like a
contraction is planned for `minigrad.runtime.gemm` instead. Plans depend
only on the kernel's structure, not on which buffers it reads, so they
are cached by `Kernel.key()` and a training loop lowers each kernel once.

`run_plan` executes a plan: an odometer over the outer dimensions, and
for each position the instructions run block by block along the vector
dimension (see `minigrad.runtime.blockops`).
"""

from std.ffi import _Global
from std.memory import unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.os import abort, getenv

from minigrad.dtype import DT
from minigrad.graph import (
    BytePtr,
    NodeKey,
    OP_CAST,
    OP_CMPEQ,
    OP_CMPLT,
    OP_CMPNE,
    OP_MUL,
    OP_WHERE,
    REDUCE_SUM,
    is_binary,
    is_unary,
)
from minigrad.runtime.blockops import (
    BLOCK,
    MASKED,
    PAD_BYTES,
    accumulate_block,
    binary_block,
    cast_block,
    compare_block,
    copy_block,
    fill_block,
    fill_identity,
    gather_block,
    gather_table_block,
    reduce_into_block,
    segment_reduce_block,
    splat_block,
    unary_block,
    where_block,
)
from minigrad.runtime.gemm import gemm, gemm_threads
from minigrad.runtime.pool import run_parallel
from minigrad.view import View, prod

comptime K_LOAD = 100
"""Instruction: read load `a` of the kernel."""
comptime K_CONST = 101
"""Instruction: the constant whose `CONST` argument is `a`."""

comptime SLOT_BYTES = 2176
"""Bytes per block temporary: `BLOCK` 8-byte elements plus padding,
rounded up to a multiple of 64."""


struct Instr(Copyable):
    """One instruction of a kernel, in SSA form: operands are the indices
    of earlier instructions."""

    var op: Int
    """An elementwise `OP_*` code, `K_LOAD`, or `K_CONST`."""
    var dtype: DT
    """The element type of the result."""
    var src_dtype: DT
    """The element type of the first operand (it differs from `dtype`
    for casts and comparisons)."""
    var a: Int
    var b: Int
    var c: Int

    def __init__(
        out self, op: Int, dtype: DT, src_dtype: DT, a: Int, b: Int, c: Int
    ):
        self.op = op
        self.dtype = dtype
        self.src_dtype = src_dtype
        self.a = a
        self.b = b
        self.c = c


struct Load(Copyable):
    """A read of a buffer through a view whose shape is the kernel's
    iteration shape."""

    var node: Int
    """The graph node to read: a `BUFFER` by the time the kernel runs."""
    var view: View
    var dtype: DT

    def __init__(out self, node: Int, var view: View, dtype: DT):
        self.node = node
        self.view = view^
        self.dtype = dtype


struct Kernel(Movable):
    """A fused computation with one output buffer."""

    var root: Int
    """The graph node this kernel realizes."""
    var shape: List[Int]
    """The iteration shape: the output shape before any reduction."""
    var reduce_op: Int
    """`REDUCE_SUM`, `REDUCE_MAX`, or -1 for no reduction."""
    var reduce_axes: List[Int]
    """The dimensions of `shape` that are reduced."""
    var dtype: DT
    """The element type of the output."""
    var loads: List[Load]
    var instrs: List[Instr]
    """The program; the last instruction is the value."""

    def __init__(out self, root: Int, var shape: List[Int], dtype: DT):
        self.root = root
        self.shape = shape^
        self.reduce_op = -1
        self.reduce_axes = List[Int]()
        self.dtype = dtype
        self.loads = List[Load]()
        self.instrs = List[Instr]()

    def out_size(self) -> Int:
        """The number of elements in the output."""
        var n = prod(self.shape)
        for a in self.reduce_axes:
            n //= self.shape[a]
        return n

    def key(self) -> NodeKey:
        """The kernel's structure, without the identity of the buffers it
        reads or the values of its constants: kernels with equal keys
        share one plan, so a constant that changes every step (a decayed
        learning rate, say) does not build a new plan every step."""
        var d = List[Int]()
        d.append(len(self.shape))
        for s in self.shape:
            d.append(s)
        d.append(self.reduce_op)
        d.append(len(self.reduce_axes))
        for a in self.reduce_axes:
            d.append(a)
        d.append(self.dtype.code)
        d.append(len(self.loads))
        for l in self.loads:
            d.append(l.dtype.code)
            d.append(l.view.offset)
            for s in l.view.strides:
                d.append(s)
            d.append(len(l.view.mask))
            for m in l.view.mask:
                d.append(m)
        d.append(len(self.instrs))
        for i in self.instrs:
            d.append(i.op)
            d.append(i.dtype.code)
            d.append(i.src_dtype.code)
            d.append(0 if i.op == K_CONST else i.a)
            d.append(i.b)
            d.append(i.c)
        return NodeKey(d^)


struct PlanLoad(Copyable):
    """A load, re-expressed over the plan's loop dimensions."""

    var dtype: DT
    var offset: Int
    """The buffer element read at loop index zero."""
    var strides: List[Int]
    """Elements per step of each loop dimension; the vector dimension is
    last."""
    var mask: List[Int]
    """Empty, or `(lo, hi)` per loop dimension."""
    var zero: Bool
    """Whether nothing is valid: the load reads as zeros."""
    var persistent: Bool
    """Whether the load is constant along the vector dimension, so its
    block is filled once per outer position."""
    var table: List[Int]
    """Empty, or (when the vector dimension is several dimensions
    flattened) the offset of each of its elements from the element at
    its start, `MASKED` where masked."""

    def __init__(out self, dtype: DT, offset: Int):
        self.dtype = dtype
        self.offset = offset
        self.strides = List[Int]()
        self.mask = List[Int]()
        self.zero = False
        self.persistent = False
        self.table = List[Int]()


struct Plan(Movable):
    """A kernel lowered to loops over blocks."""

    var sizes: List[Int]
    """Loop dimension sizes; the last is the vector dimension."""
    var out_strides: List[Int]
    """Output elements per step of each loop dimension (0 if reduced)."""
    var vec_reduced: Bool
    """Whether the vector dimension is reduced: each run of blocks folds
    into one output element."""
    var group: Int
    """When trailing dimensions are flattened into the vector dimension
    and the innermost of them are reduced, the number of consecutive
    vector elements that fold into one output element; else 1."""
    var reduce_op: Int
    var dtype: DT
    var out_size: Int
    var loads: List[PlanLoad]
    var instrs: List[Instr]
    var slots: List[Int]
    """The block temporary each instruction writes."""
    var nslots: Int
    var gemm: Bool
    """Whether the kernel is a contraction `C[i,j] = sum_k A[i,k] B[k,j]`
    run by `minigrad.runtime.gemm` instead of the block loops; the
    fields below are then its operands and offset tables."""
    var gemm_a: Int
    """The load that is operand A."""
    var gemm_b: Int
    var gemm_bdt: DT
    """How operand B is stored: the plan's type, or a narrower one that
    gemm converts as it reads (bfloat16 weights, say)."""
    var ci: List[Int]
    var cj: List[Int]
    var ai: List[Int]
    var ak: List[Int]
    var bk: List[Int]
    var bj: List[Int]
    var cb: List[Int]
    """For a batch of contractions (dimensions shared by both operands
    and the output), the offset of each one in C; `[0]` for one."""
    var ab: List[Int]
    var bb: List[Int]

    def __init__(out self, dtype: DT, reduce_op: Int, out_size: Int):
        self.sizes = List[Int]()
        self.out_strides = List[Int]()
        self.vec_reduced = False
        self.group = 1
        self.reduce_op = reduce_op
        self.dtype = dtype
        self.out_size = out_size
        self.loads = List[PlanLoad]()
        self.instrs = List[Instr]()
        self.slots = List[Int]()
        self.nslots = 0
        self.gemm = False
        self.gemm_a = 0
        self.gemm_b = 1
        self.gemm_bdt = dtype
        self.ci = List[Int]()
        self.cj = List[Int]()
        self.ai = List[Int]()
        self.ak = List[Int]()
        self.bk = List[Int]()
        self.bj = List[Int]()
        self.cb = List[Int]()
        self.ab = List[Int]()
        self.bb = List[Int]()


def _offsets(
    groups: List[Int], gsize: List[Int], strides: List[Int], base: Int
) -> List[Int]:
    """`base + sum(index[g] * strides[g])` for every index over the
    dimensions `groups`, in row-major order."""
    var out = List[Int]()
    out.append(base)
    for g in groups:
        var n = len(out)
        var next = List[Int](capacity=n * gsize[g])
        for i in range(n):
            for j in range(gsize[g]):
                next.append(out[i] + j * strides[g])
        out = next^
    return out^


def _contraction_conversion(k: Kernel) -> Int:
    """Whether `k` is a sum of the products of two loads, each perhaps
    converted to the kernel's type on the way, that gemm can run.

    Returns:
        -2 if not; else the load gemm must convert (stored as `float16`,
        `bfloat16`, or `int8`, in a `float32` kernel), or -1 for none.
    """
    if k.reduce_op != REDUCE_SUM or len(k.loads) != 2:
        return -2
    if k.dtype != DT.float32 and k.dtype != DT.float64:
        return -2
    if k.loads[0].view.has_mask() or k.loads[1].view.has_mask():
        return -2
    var n = len(k.instrs)
    ref mul = k.instrs[n - 1]
    if mul.op != OP_MUL:
        return -2
    var converted = -1
    var seen = 0
    for operand in [mul.a, mul.b]:
        ref ins = k.instrs[operand]
        if ins.op == K_LOAD:
            if k.loads[ins.a].dtype != k.dtype:
                return -2
            seen |= 1 << ins.a
        elif ins.op == OP_CAST and k.instrs[ins.a].op == K_LOAD:
            var load = k.instrs[ins.a].a
            var stored = k.loads[load].dtype
            var ok = k.dtype == DT.float32 and (
                stored.is_half() or stored == DT.int8
            )
            if not ok or converted >= 0:
                return -2
            converted = load
            seen |= 1 << load
        else:
            return -2
    # Both loads, and nothing but the loads, the casts, and the product.
    if seen != 3 or n != 3 + (1 if converted >= 0 else 0):
        return -2
    return converted


def make_plan(k: Kernel) -> Plan:
    """Lowers a kernel to loops.

    Args:
        k: The kernel.

    Returns:
        Its execution plan.
    """
    var n = len(k.shape)
    var nl = len(k.loads)
    var plan = Plan(k.dtype, k.reduce_op, k.out_size())
    var reduced = List[Bool](length=n, fill=False)
    for a in k.reduce_axes:
        reduced[a] = True
    var ostride = List[Int](length=n, fill=0)
    var acc = 1
    for i in range(n):
        var d = n - 1 - i
        if not reduced[d]:
            ostride[d] = acc
            acc *= k.shape[d]

    # Groups of merged dimensions, in order. Per group: size, whether
    # reduced, output stride; per (load, group): stride and mask.
    var gsize = List[Int]()
    var gred = List[Bool]()
    var gostride = List[Int]()
    var lstride = List[List[Int]]()
    var lmask = List[List[Int]]()
    for _ in range(nl):
        lstride.append(List[Int]())
        lmask.append(List[Int]())
    for d in range(n):
        var size = k.shape[d]
        if size == 1:
            continue
        var ng = len(gsize)
        var merge = ng > 0 and gred[ng - 1] == reduced[d]
        if merge:
            for l in range(nl):
                ref v = k.loads[l].view
                if lstride[l][ng - 1] != v.strides[d] * size:
                    merge = False
                if lmask[l][2 * (ng - 1)] != 0 or (
                    lmask[l][2 * (ng - 1) + 1] != gsize[ng - 1]
                ):
                    merge = False
                if v.has_mask() and (
                    v.mask[2 * d] != 0 or v.mask[2 * d + 1] != size
                ):
                    merge = False
        if merge:
            gsize[ng - 1] *= size
            gostride[ng - 1] = ostride[d]
            for l in range(nl):
                lstride[l][ng - 1] = k.loads[l].view.strides[d]
                lmask[l][2 * (ng - 1) + 1] = gsize[ng - 1]
        else:
            gsize.append(size)
            gred.append(reduced[d])
            gostride.append(ostride[d])
            for l in range(nl):
                ref v = k.loads[l].view
                lstride[l].append(v.strides[d])
                lmask[l].append(v.mask[2 * d] if v.has_mask() else 0)
                lmask[l].append(v.mask[2 * d + 1] if v.has_mask() else size)
    if len(gsize) == 0:
        # A single element: one loop of length 1.
        gsize.append(1)
        gred.append(False)
        gostride.append(1)
        for l in range(nl):
            lstride[l].append(0)
            lmask[l].append(0)
            lmask[l].append(1)
    var ng = len(gsize)

    # A sum of products of two loads, where each output dimension
    # belongs to one operand only, is a contraction: hand it to gemm.
    var converted = _contraction_conversion(k)
    if converted >= -1:
        var gi = List[Int]()
        var gj = List[Int]()
        var gk = List[Int]()
        var gb = List[Int]()
        for g in range(ng):
            if gred[g]:
                gk.append(g)
            elif lstride[1][g] == 0:
                gi.append(g)
            elif lstride[0][g] == 0:
                gj.append(g)
            else:
                gb.append(g)  # shared by both operands and C: a batch
        if len(gk) > 0 and (len(gi) > 0 or len(gj) > 0):
            var a = 0
            var b = 1
            var ni = 1
            var nj = 1
            for g in gi:
                ni *= gsize[g]
            for g in gj:
                nj *= gsize[g]
            # gemm converts operand B as it reads it; otherwise it
            # vectorizes along j, so make that the longer side.
            if converted == 0 or (converted < 0 and ni > nj):
                a = 1
                b = 0
                var t = gi^
                gi = gj^
                gj = t^
            plan.gemm = True
            plan.gemm_a = a
            plan.gemm_b = b
            plan.gemm_bdt = k.loads[b].dtype
            plan.ci = _offsets(gi, gsize, gostride, 0)
            plan.cj = _offsets(gj, gsize, gostride, 0)
            plan.ai = _offsets(gi, gsize, lstride[a], k.loads[a].view.offset)
            plan.ak = _offsets(gk, gsize, lstride[a], 0)
            plan.bk = _offsets(gk, gsize, lstride[b], k.loads[b].view.offset)
            plan.bj = _offsets(gj, gsize, lstride[b], 0)
            plan.cb = _offsets(gb, gsize, gostride, 0)
            plan.ab = _offsets(gb, gsize, lstride[a], 0)
            plan.bb = _offsets(gb, gsize, lstride[b], 0)
            return plan^

    # What to vectorize over. Either one dimension (a reduced one, or
    # the innermost output one), or the trailing dimensions flattened
    # into one: output dimensions, then reduced ones, so that runs of the
    # flat index fold into consecutive output elements. Costs are rough
    # nanoseconds: a fixed price per run of the vector loop, and a price
    # per element that grows with loads that are not contiguous.
    var total = 1
    for g in range(ng):
        total *= gsize[g]
    var ni_ = len(k.instrs)
    var per_run = 30 + 6 * ni_
    var best = -1
    var best_cost = -1
    for g in range(ng):
        if not gred[g] and gostride[g] != 1:
            continue
        var gathers = 0
        for l in range(nl):
            if (lstride[l][g] != 0 and lstride[l][g] != 1) or (
                lmask[l][2 * g] != 0 or lmask[l][2 * g + 1] != gsize[g]
            ):
                gathers += 1
        var cost = (total // gsize[g]) * per_run + total * (
            15 * ni_ + 100 * gathers
        ) // 100
        if best < 0 or cost <= best_cost:
            best = g
            best_cost = cost
    var first = ng
    var flat_size = 1
    var seen_output = False
    while first > 0 and flat_size < BLOCK:
        var g = first - 1
        if gred[g] and seen_output:
            break
        if flat_size * gsize[g] > 65536:
            break
        if not gred[g]:
            seen_output = True
        first = g
        flat_size *= gsize[g]
    var flat = False
    var group = 1
    # Per load, its offset table over the flattened dimensions.
    var tables = List[List[Int]]()
    if ng - first >= 2:
        for g in range(first, ng):
            if gred[g]:
                group *= gsize[g]
        var ntable = 0
        var index = List[Int](length=ng, fill=0)
        for l in range(nl):
            var table = List[Int](capacity=flat_size)
            for g in range(first, ng):
                index[g] = 0
            var identity = True
            var constant = True
            for j in range(flat_size):
                var off = 0
                var ok = True
                for g in range(first, ng):
                    off += index[g] * lstride[l][g]
                    if index[g] < lmask[l][2 * g] or (
                        index[g] >= lmask[l][2 * g + 1]
                    ):
                        ok = False
                if not ok:
                    off = MASKED
                if off != j:
                    identity = False
                if off != 0:
                    constant = False
                table.append(off)
                var g = ng - 1
                while g >= first:
                    index[g] += 1
                    if index[g] < gsize[g]:
                        break
                    index[g] = 0
                    g -= 1
            if identity or constant:
                table.clear()
                table.append(1 if identity else 0)
            else:
                ntable += 1
            tables.append(table^)
        var cost = (total // flat_size) * per_run + total * (
            15 * ni_ + 120 * ntable + (50 if group > 1 else 0)
        ) // 100
        flat = cost < best_cost

    if flat:
        for g in range(first):
            plan.sizes.append(gsize[g])
            plan.out_strides.append(gostride[g])
        plan.sizes.append(flat_size)
        plan.out_strides.append(1)
        plan.group = group
        for l in range(nl):
            ref v = k.loads[l].view
            var pl = PlanLoad(k.loads[l].dtype, v.offset)
            if v.has_mask():
                for d in range(n):
                    if v.mask[2 * d] >= v.mask[2 * d + 1]:
                        pl.zero = True
            var masked = False
            for g in range(first):
                if lmask[l][2 * g] != 0 or lmask[l][2 * g + 1] != gsize[g]:
                    masked = True
            for g in range(first):
                pl.strides.append(lstride[l][g])
                if masked:
                    pl.mask.append(lmask[l][2 * g])
                    pl.mask.append(lmask[l][2 * g + 1])
            if masked:
                pl.mask.append(0)
                pl.mask.append(flat_size)
            if len(tables[l]) == 1 and flat_size != 1:
                # Contiguous (stride 1) or constant (stride 0) along the
                # flattened dimensions: no table needed.
                pl.strides.append(tables[l][0])
                pl.persistent = tables[l][0] == 0
            else:
                pl.strides.append(0)
                pl.table = tables[l].copy()
            plan.loads.append(pl^)
    else:
        for g in range(ng):
            if g != best:
                plan.sizes.append(gsize[g])
                plan.out_strides.append(gostride[g])
        plan.sizes.append(gsize[best])
        plan.out_strides.append(gostride[best])
        plan.vec_reduced = gred[best]

        for l in range(nl):
            ref v = k.loads[l].view
            var pl = PlanLoad(k.loads[l].dtype, v.offset)
            var masked = False
            for g in range(ng):
                if lmask[l][2 * g] != 0 or lmask[l][2 * g + 1] != gsize[g]:
                    masked = True
            if v.has_mask():
                for d in range(n):
                    if v.mask[2 * d] >= v.mask[2 * d + 1]:
                        pl.zero = True
            for g in range(ng):
                if g != best:
                    pl.strides.append(lstride[l][g])
                    if masked:
                        pl.mask.append(lmask[l][2 * g])
                        pl.mask.append(lmask[l][2 * g + 1])
            pl.strides.append(lstride[l][best])
            if masked:
                pl.mask.append(lmask[l][2 * best])
                pl.mask.append(lmask[l][2 * best + 1])
            pl.persistent = lstride[l][best] == 0 and (
                not masked
                or (
                    lmask[l][2 * best] == 0
                    and lmask[l][2 * best + 1] == gsize[best]
                )
            )
            plan.loads.append(pl^)

    # Block temporaries: constants and persistent loads keep theirs for
    # the whole run; the others are reused once their value is dead.
    var ni = len(k.instrs)
    var last_use = List[Int](length=ni, fill=-1)
    for i in range(ni):
        ref ins = k.instrs[i]
        if ins.op != K_LOAD and ins.op != K_CONST:
            last_use[ins.a] = i
            if ins.b >= 0:
                last_use[ins.b] = i
            if ins.c >= 0:
                last_use[ins.c] = i
    last_use[ni - 1] = ni
    var free = List[Int]()
    var nslots = 0
    for i in range(ni):
        ref ins = k.instrs[i]
        plan.instrs.append(ins.copy())
        var pinned = ins.op == K_CONST or (
            ins.op == K_LOAD and plan.loads[ins.a].persistent
        )
        if pinned or len(free) == 0:
            plan.slots.append(nslots)
            nslots += 1
        else:
            plan.slots.append(free.pop())
        if ins.op != K_LOAD and ins.op != K_CONST:
            for j in range(3):
                var o = ins.a
                if j == 1:
                    o = ins.b
                elif j == 2:
                    o = ins.c
                if o < 0 or last_use[o] != i:
                    continue
                ref oi = k.instrs[o]
                var opinned = oi.op == K_CONST or (
                    oi.op == K_LOAD and plan.loads[oi.a].persistent
                )
                if not opinned:
                    last_use[o] = -2  # released once
                    free.append(plan.slots[o])
    plan.nslots = nslots
    return plan^


struct Runtime(Movable):
    """The plan cache and scratch memory of the CPU executor."""

    var plans: List[Plan]
    var index: Dict[NodeKey, Int]
    var scratch: BytePtr
    """Block temporaries: `scratch_slots` of `SLOT_BYTES` each."""
    var scratch_slots: Int
    var zeros: BytePtr
    """One block of zero bytes."""
    var threads: Int
    """The most threads a kernel may use."""
    var debug: Int
    """The value of the environment variable `MINIGRAD_DEBUG`: at 2 and
    above, every kernel run is printed with its time."""

    def __init__(out self):
        self.plans = List[Plan]()
        self.index = Dict[NodeKey, Int]()
        self.scratch_slots = 32
        self.scratch = unsafe_alloc[Byte](
            self.scratch_slots * SLOT_BYTES, alignment=64
        )
        self.zeros = unsafe_alloc[Byte](SLOT_BYTES, alignment=64)
        unsafe_memset_zero(self.zeros, SLOT_BYTES)
        self.threads = gemm_threads()
        self.debug = 0
        var s = getenv("MINIGRAD_DEBUG")
        if s:
            try:
                self.debug = atol(s)
            except:
                pass

    def reserve(mut self, nslots: Int):
        """Makes room for `nslots` block temporaries."""
        if nslots <= self.scratch_slots:
            return
        self.scratch.unsafe_free()
        self.scratch_slots = nslots * 2
        self.scratch = unsafe_alloc[Byte](
            self.scratch_slots * SLOT_BYTES, alignment=64
        )


def _init_runtime() -> Runtime:
    return Runtime()


comptime _RUNTIME = _Global["minigrad_runtime", _init_runtime]


@always_inline
def runtime() -> Pointer[Runtime, MutUntrackedOrigin]:
    """The process-wide CPU runtime."""
    try:
        return _RUNTIME.get_or_create_ptr()
    except:
        abort("minigrad: cannot create the runtime")


comptime GEMM_SPLIT_WORK = 4_000_000
"""Multiply-adds of one product below which a batch of products is
spread over threads by product rather than within each."""


@fieldwise_init
struct _Batch(Copyable):
    """What a task of a batch of contractions needs: addresses of the
    plan and the buffers."""

    var plan: Int
    var c: Int
    var a: Int
    var b: Int
    var threads: Int


def _batch_task(ctx: Int, i: Int):
    """Contraction `i` of a batch."""
    var job = Pointer[_Batch, MutUntrackedOrigin](unsafe_from_address=ctx)
    var plan = Pointer[Plan, MutUntrackedOrigin](unsafe_from_address=job[].plan)
    var isz = plan[].dtype.itemsize()
    var c = BytePtr(unsafe_from_address=job[].c + plan[].cb[i] * isz)
    var a = BytePtr(unsafe_from_address=job[].a + plan[].ab[i] * isz)
    var bsz = plan[].gemm_bdt.itemsize()
    var b = BytePtr(unsafe_from_address=job[].b + plan[].bb[i] * bsz)
    if plan[].dtype == DT.float64:
        _gemm[DType.float64, DType.float64](c, a, b, plan, job[].threads)
    elif plan[].gemm_bdt == DT.float16:
        _gemm[DType.float32, DType.float16](c, a, b, plan, job[].threads)
    elif plan[].gemm_bdt == DT.bfloat16:
        _gemm[DType.float32, DType.bfloat16](c, a, b, plan, job[].threads)
    elif plan[].gemm_bdt == DT.int8:
        _gemm[DType.float32, DType.int8](c, a, b, plan, job[].threads)
    else:
        _gemm[DType.float32, DType.float32](c, a, b, plan, job[].threads)


@always_inline
def _gemm[
    dt: DType, bdt: DType
](
    c: BytePtr,
    a: BytePtr,
    b: BytePtr,
    plan: Pointer[Plan, MutUntrackedOrigin],
    threads: Int,
):
    gemm[dt, bdt](
        c,
        plan[].ci,
        plan[].cj,
        a,
        plan[].ai,
        plan[].ak,
        b,
        plan[].bk,
        plan[].bj,
        threads,
    )


def run_plan(
    plan: Plan, out_ptr: BytePtr, ins: List[BytePtr], instrs: List[Instr]
):
    """Executes a plan.

    Args:
        plan: The plan.
        out_ptr: The output buffer, of `plan.out_size` elements.
        ins: The buffer each of the plan's loads reads.
        instrs: The instructions of the kernel being run, which supply
            the values of its constants (the plan may have been built
            from a kernel with other values).
    """
    var rt = runtime()
    if plan.gemm:
        unsafe_memset_zero(out_ptr, plan.out_size * plan.dtype.itemsize())
        var batch = _Batch(
            Int(Pointer(to=plan)),
            Int(out_ptr),
            Int(ins[plan.gemm_a]),
            Int(ins[plan.gemm_b]),
            rt[].threads,
        )
        var nb = len(plan.cb)
        var work = len(plan.ci) * len(plan.cj) * len(plan.ak)
        if nb > 1 and work < GEMM_SPLIT_WORK:
            # Many small products: one per task, each on one thread.
            batch.threads = 1
            run_parallel[_batch_task](Int(Pointer(to=batch)), nb)
        else:
            for b in range(nb):
                _batch_task(Int(Pointer(to=batch)), b)
        # Tasks read `batch` through an address: keep it alive until here.
        _ = batch^
        return
    rt[].reserve(plan.nslots)
    var scratch = rt[].scratch
    var zeros = rt[].zeros
    var nd = len(plan.sizes)
    var nouter = nd - 1
    var length = plan.sizes[nd - 1]
    var nl = len(plan.loads)
    var ni = len(plan.instrs)
    var osz = plan.dtype.itemsize()
    var rop = plan.reduce_op
    var group = plan.group

    if rop >= 0:
        fill_identity(plan.dtype, rop, out_ptr, plan.out_size)

    var idx = List[Int](length=nouter, fill=0)
    var cur = List[Int](length=nl, fill=0)
    var valid = List[Bool](length=nl, fill=True)
    for l in range(nl):
        cur[l] = plan.loads[l].offset
    var out_off = 0
    var regs = List[BytePtr](length=ni, fill=zeros)

    # Constants fill their block once.
    for i in range(ni):
        ref ins_i = plan.instrs[i]
        if ins_i.op == K_CONST:
            var p = scratch.unsafe_offset(plan.slots[i] * SLOT_BYTES)
            fill_block(ins_i.dtype, p, instrs[i].a, min(length, BLOCK))
            regs[i] = p

    while True:
        # Per outer position: which loads are inside their masks, and
        # the blocks of loads that are constant along the vector.
        for l in range(nl):
            ref ld = plan.loads[l]
            var ok = not ld.zero
            if ok and len(ld.mask) > 0:
                for d in range(nouter):
                    if idx[d] < ld.mask[2 * d] or idx[d] >= ld.mask[2 * d + 1]:
                        ok = False
            valid[l] = ok
        for i in range(ni):
            ref ins_i = plan.instrs[i]
            if ins_i.op == K_LOAD and plan.loads[ins_i.a].persistent:
                var l = ins_i.a
                if valid[l]:
                    var p = scratch.unsafe_offset(plan.slots[i] * SLOT_BYTES)
                    splat_block(
                        ins_i.dtype,
                        ins[l].unsafe_offset(cur[l] * ins_i.dtype.itemsize()),
                        p,
                        min(length, BLOCK),
                    )
                    regs[i] = p
                else:
                    regs[i] = zeros

        var j0 = 0
        while j0 < length:
            var n = min(BLOCK, length - j0)
            if group > 1:
                # Whole groups per block, or a block inside one group.
                if group <= BLOCK:
                    n = min(BLOCK // group * group, length - j0)
                else:
                    n = min(BLOCK, group - j0 % group)
            for i in range(ni):
                ref ins_i = plan.instrs[i]
                var op = ins_i.op
                if op == K_CONST:
                    continue
                var dst = scratch.unsafe_offset(plan.slots[i] * SLOT_BYTES)
                if op == K_LOAD:
                    var l = ins_i.a
                    ref ld = plan.loads[l]
                    if ld.persistent:
                        continue
                    if not valid[l]:
                        regs[i] = zeros
                        continue
                    var isz = ins_i.dtype.itemsize()
                    if len(ld.table) > 0:
                        gather_table_block(
                            ins_i.dtype,
                            ins[l].unsafe_offset(cur[l] * isz),
                            Pointer[Int, MutUntrackedOrigin](
                                unsafe_from_address=Int(ld.table.unsafe_ptr())
                                + 8 * j0
                            ),
                            dst,
                            n,
                        )
                        regs[i] = dst
                        continue
                    var vs = ld.strides[nouter]
                    var base = cur[l] + j0 * vs
                    var lo = 0
                    var hi = n
                    if len(ld.mask) > 0:
                        lo = min(max(ld.mask[2 * nouter] - j0, 0), n)
                        hi = min(max(ld.mask[2 * nouter + 1] - j0, 0), n)
                    if lo == 0 and hi == n:
                        if vs == 1:
                            regs[i] = ins[l].unsafe_offset(base * isz)
                        else:
                            gather_block(
                                ins_i.dtype,
                                ins[l].unsafe_offset(base * isz),
                                vs,
                                dst,
                                n,
                            )
                            regs[i] = dst
                    else:
                        unsafe_memset_zero(dst, n * isz)
                        if hi > lo:
                            gather_block(
                                ins_i.dtype,
                                ins[l].unsafe_offset((base + lo * vs) * isz),
                                vs,
                                dst.unsafe_offset(lo * isz),
                                hi - lo,
                            )
                        regs[i] = dst
                    continue
                # The last instruction of an unreduced kernel writes the
                # output directly when whole vectors fit exactly.
                if i == ni - 1 and rop < 0 and (n & 63) == 0:
                    dst = out_ptr.unsafe_offset((out_off + j0) * osz)
                if op == OP_CAST:
                    cast_block(
                        ins_i.src_dtype, ins_i.dtype, regs[ins_i.a], dst, n
                    )
                elif op == OP_WHERE:
                    where_block(
                        ins_i.dtype,
                        regs[ins_i.a],
                        regs[ins_i.b],
                        regs[ins_i.c],
                        dst,
                        n,
                    )
                elif op == OP_CMPLT or op == OP_CMPNE or op == OP_CMPEQ:
                    compare_block(
                        ins_i.src_dtype,
                        op,
                        regs[ins_i.a],
                        regs[ins_i.b],
                        dst,
                        n,
                    )
                elif is_binary(op):
                    binary_block(
                        ins_i.dtype, op, regs[ins_i.a], regs[ins_i.b], dst, n
                    )
                else:
                    unary_block(ins_i.dtype, op, regs[ins_i.a], dst, n)
                regs[i] = dst

            var result = regs[ni - 1]
            if rop < 0:
                var o = out_ptr.unsafe_offset((out_off + j0) * osz)
                if result != o:
                    copy_block(plan.dtype, result, o, n)
            elif group > 1:
                var o = out_ptr.unsafe_offset((out_off + j0 // group) * osz)
                if group <= BLOCK:
                    segment_reduce_block(
                        plan.dtype, rop, result, group, o, n // group
                    )
                else:
                    reduce_into_block(plan.dtype, rop, result, o, n)
            elif plan.vec_reduced:
                reduce_into_block(
                    plan.dtype,
                    rop,
                    result,
                    out_ptr.unsafe_offset(out_off * osz),
                    n,
                )
            else:
                accumulate_block(
                    plan.dtype,
                    rop,
                    result,
                    out_ptr.unsafe_offset((out_off + j0) * osz),
                    n,
                )
            j0 += n

        # Advance the odometer over the outer dimensions.
        var d = nouter - 1
        while d >= 0:
            idx[d] += 1
            if idx[d] < plan.sizes[d]:
                for l in range(nl):
                    cur[l] += plan.loads[l].strides[d]
                out_off += plan.out_strides[d]
                break
            idx[d] = 0
            var back = plan.sizes[d] - 1
            for l in range(nl):
                cur[l] -= back * plan.loads[l].strides[d]
            out_off -= back * plan.out_strides[d]
            d -= 1
        if d < 0:
            break
