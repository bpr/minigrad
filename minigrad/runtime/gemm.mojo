"""A tuned matrix-multiply kernel (`minigrad.runtime.gemm`).

`gemm` computes a contraction of two operands,

    C[ci[i] + cj[j]] += sum_k A[ai[i] + ak[k]] * B[bk[k] + bj[j]]

where every operand is addressed through *offset tables*: the element for
row `i` and column `j` lives at `row_offset[i] + col_offset[j]`. A plain
matrix product is the special case of arithmetic tables, but so is a
product of transposed or strided matrices, a batch of rows, and a
convolution (rows are output positions, `k` runs over the receptive
field, and the tables encode the sliding window): the planner in
`minigrad.runtime.kernel` builds the tables from the views of a kernel
shaped like a contraction.

The implementation is the classic blocking of Goto and BLIS: blocks of
the operands are packed into contiguous panels through the tables, sized
to stay in cache, and a microkernel accumulates an `MR` by `NR` tile of C
in vector registers. Columns of C are independent, so `gemm` splits them
over threads; each thread packs its own blocks.
"""

from std.ffi import external_call
from std.os import getenv
from std.memory import unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.sys import num_physical_cores, simd_width_of, size_of

from minigrad.graph import BytePtr

comptime MR = 8
"""Rows of C per microkernel tile."""
comptime NV = 2
"""SIMD vectors of C columns per microkernel tile."""
comptime KC = 256
"""Terms of the sum per packed block: a B panel then fits in L1."""
comptime MC = 96
"""Rows of C per packed A block (a multiple of `MR`)."""
comptime NC_VECTORS = 256
"""SIMD vectors of C columns per packed B block: with `KC` terms, about
4 MB of packed B per thread."""

comptime PARALLEL_WORK = 32_000_000
"""Multiply-adds below which `gemm` stays on the calling thread: about a
quarter of a millisecond of work, several times what starting the
threads costs."""


@always_inline
def _micro[
    dt: DType
](
    ap: Pointer[Scalar[dt], MutUntrackedOrigin],
    bp: Pointer[Scalar[dt], MutUntrackedOrigin],
    kc: Int,
    tile: Pointer[Scalar[dt], MutUntrackedOrigin],
):
    """`tile[r, :] = sum_k ap[k, r] * bp[k, :]` for an `MR` by `NV * W`
    tile; `ap` is `kc` by `MR`, `bp` is `kc` by `NV * W`."""
    comptime W = simd_width_of[dt]()
    var acc = Array[SIMD[dt, W], MR * NV](fill=SIMD[dt, W](0))
    for k in range(kc):
        var pa = ap.unsafe_offset(k * MR)
        var pb = bp.unsafe_offset(k * NV * W)
        comptime for v in range(NV):
            var bv = pb.unsafe_load[width=W, alignment=1](v * W)
            comptime for r in range(MR):
                acc[r * NV + v] = SIMD[dt, W](pa[unsafe_offset=r]).fma(
                    bv, acc[r * NV + v]
                )
    comptime for i in range(MR * NV):
        tile.unsafe_store(i * W, acc[i])


@always_inline
def _micro_rows[
    dt: DType
](
    a: Pointer[Scalar[dt], MutUntrackedOrigin],
    rows: Pointer[Int, MutUntrackedOrigin],
    bp: Pointer[Scalar[dt], MutUntrackedOrigin],
    kc: Int,
    tile: Pointer[Scalar[dt], MutUntrackedOrigin],
):
    """`_micro` reading A in place: row `r` of the tile's A is the `kc`
    consecutive elements at `a + rows[r]`."""
    comptime W = simd_width_of[dt]()
    var acc = Array[SIMD[dt, W], MR * NV](fill=SIMD[dt, W](0))
    var ptrs = Array[Pointer[Scalar[dt], MutUntrackedOrigin], MR](fill=a)
    comptime for r in range(MR):
        ptrs[r] = a.unsafe_offset(rows[unsafe_offset=r])
    for k in range(kc):
        var pb = bp.unsafe_offset(k * NV * W)
        comptime for v in range(NV):
            var bv = pb.unsafe_load[width=W, alignment=1](v * W)
            comptime for r in range(MR):
                acc[r * NV + v] = SIMD[dt, W](ptrs[r][unsafe_offset=k]).fma(
                    bv, acc[r * NV + v]
                )
    comptime for i in range(MR * NV):
        tile.unsafe_store(i * W, acc[i])


comptime _IntPtr = Pointer[Int, MutUntrackedOrigin]


@always_inline
def _is_unit(offsets: _IntPtr, n: Int) -> Bool:
    """Whether the `n` offsets are consecutive."""
    for i in range(1, n):
        if offsets[unsafe_offset=i] != offsets[unsafe_offset=i - 1] + 1:
            return False
    return True


@fieldwise_init
struct _Job(Copyable):
    """One thread's share of a `gemm`: rows `[i_lo, i_hi)` and columns
    `[j_lo, j_hi)` of C. Pointers are held as addresses so the struct
    has no parameters."""

    var c: Int
    var a: Int
    var b: Int
    var ci: Int
    var cj: Int
    var ai: Int
    var ak: Int
    var bk: Int
    var bj: Int
    var nk: Int
    var i_lo: Int
    var i_hi: Int
    var j_lo: Int
    var j_hi: Int


def _block[dt: DType](job: _Job):
    """Adds the contraction into the job's rows and columns of C."""
    comptime W = simd_width_of[dt]()
    comptime NR = NV * W
    comptime NC = NC_VECTORS * W
    comptime P = Pointer[Scalar[dt], MutUntrackedOrigin]
    var ni = job.i_hi - job.i_lo
    var nk = job.nk
    var c = P(unsafe_from_address=job.c)
    var a = P(unsafe_from_address=job.a)
    var b = P(unsafe_from_address=job.b)
    var ci = _IntPtr(unsafe_from_address=job.ci)
    var cj = _IntPtr(unsafe_from_address=job.cj)
    var ai = _IntPtr(unsafe_from_address=job.ai)
    var bj = _IntPtr(unsafe_from_address=job.bj)
    var kc_max = min(KC, nk)
    var nc_max = min(NC, (job.j_hi - job.j_lo + NR - 1) // NR * NR)
    var bpack = unsafe_alloc[Scalar[dt]](kc_max * nc_max + 64, alignment=64)
    var apack = unsafe_alloc[Scalar[dt]](
        (min(MC, ni) + MR) * kc_max + 64, alignment=64
    )
    var tile = unsafe_alloc[Scalar[dt]](MR * NR, alignment=64)
    # For a tile with fewer than MR rows read in place: where its rows
    # are in A, the missing ones reading zeros.
    var zeros = unsafe_alloc[Scalar[dt]](kc_max + 64, alignment=64)
    unsafe_memset_zero(zeros, kc_max + 64)
    var edge = unsafe_alloc[Int](MR, alignment=64)
    var jc = job.j_lo
    while jc < job.j_hi:
        var nc = min(NC, job.j_hi - jc)
        var jpanels = (nc + NR - 1) // NR
        var kp = 0
        while kp < nk:
            var kc = min(KC, nk - kp)
            var ak = _IntPtr(unsafe_from_address=job.ak).unsafe_offset(kp)
            var bk = _IntPtr(unsafe_from_address=job.bk).unsafe_offset(kp)
            # Pack B[kp:kp+kc, jc:jc+nc] into panels of NR columns, kc by
            # NR each, padding the last panel with zeros.
            for p in range(jpanels):
                var dst = bpack.unsafe_offset(p * kc * NR)
                var j0 = jc + p * NR
                var nr = min(NR, jc + nc - j0)
                if nr == NR and _is_unit(bj.unsafe_offset(j0), NR):
                    var first = bj[unsafe_offset=j0]
                    for k in range(kc):
                        var src = b.unsafe_offset(bk[unsafe_offset=k] + first)
                        var d = dst.unsafe_offset(k * NR)
                        comptime for v in range(NV):
                            d.unsafe_store(
                                v * W,
                                src.unsafe_load[width=W, alignment=1](v * W),
                            )
                else:
                    if nr < NR:
                        unsafe_memset_zero(dst, kc * NR)
                    for k in range(kc):
                        var row = b.unsafe_offset(bk[unsafe_offset=k])
                        var d = dst.unsafe_offset(k * NR)
                        for j in range(nr):
                            d[unsafe_offset=j] = row[
                                unsafe_offset=bj[unsafe_offset=j0 + j]
                            ]
            # When the terms are consecutive in A, its rows are read in
            # place; otherwise blocks of A are packed like B.
            var ak_unit = _is_unit(ak, kc)
            var ak0 = ak[unsafe_offset=0]
            var ic = job.i_lo
            while ic < job.i_hi:
                var mc = min(MC, job.i_hi - ic)
                var panels = (mc + MR - 1) // MR
                if not ak_unit:
                    # Pack A[ic:ic+mc, kp:kp+kc] into panels of MR rows,
                    # kc by MR each, padding the last panel with zeros.
                    for p in range(panels):
                        var dst = apack.unsafe_offset(p * kc * MR)
                        for r in range(MR):
                            var i = ic + p * MR + r
                            if i < ic + mc:
                                var row = a.unsafe_offset(ai[unsafe_offset=i])
                                for k in range(kc):
                                    dst[unsafe_offset=k * MR + r] = row[
                                        unsafe_offset=ak[unsafe_offset=k]
                                    ]
                            else:
                                for k in range(kc):
                                    dst[unsafe_offset=k * MR + r] = 0
                for jp in range(jpanels):
                    var j0 = jc + jp * NR
                    var nr = min(NR, jc + nc - j0)
                    var bpanel = bpack.unsafe_offset(jp * kc * NR)
                    var cj_unit = nr == NR and _is_unit(
                        cj.unsafe_offset(j0), NR
                    )
                    for p in range(panels):
                        var rows = min(MR, mc - p * MR)
                        if not ak_unit:
                            _micro[dt](
                                apack.unsafe_offset(p * kc * MR),
                                bpanel,
                                kc,
                                tile,
                            )
                        elif rows == MR:
                            _micro_rows[dt](
                                a.unsafe_offset(ak0),
                                ai.unsafe_offset(ic + p * MR),
                                bpanel,
                                kc,
                                tile,
                            )
                        else:
                            for r in range(MR):
                                edge[unsafe_offset=r] = (
                                    ai[unsafe_offset=ic + p * MR + r] + ak0 if r
                                    < rows else 0
                                )
                            # Offsets from `zeros`, so missing rows read it.
                            var base = Int(a) - Int(zeros)
                            for r in range(rows):
                                edge[unsafe_offset=r] += (
                                    base // size_of[Scalar[dt]]()
                                )
                            _micro_rows[dt](zeros, edge, bpanel, kc, tile)
                        for r in range(rows):
                            var crow = c.unsafe_offset(
                                ci[unsafe_offset=ic + p * MR + r]
                            )
                            var trow = tile.unsafe_offset(r * NR)
                            if cj_unit:
                                var dst = crow.unsafe_offset(
                                    cj[unsafe_offset=j0]
                                )
                                comptime for v in range(NV):
                                    dst.unsafe_store(
                                        v * W,
                                        dst.unsafe_load[width=W, alignment=1](
                                            v * W
                                        )
                                        + trow.unsafe_load[
                                            width=W, alignment=1
                                        ](v * W),
                                    )
                            else:
                                for j in range(nr):
                                    crow[
                                        unsafe_offset=cj[unsafe_offset=j0 + j]
                                    ] += trow[unsafe_offset=j]
                ic += mc
            kp += kc
        jc += nc
    bpack.unsafe_free()
    apack.unsafe_free()
    tile.unsafe_free()
    zeros.unsafe_free()
    edge.unsafe_free()


def _thread_entry[
    dt: DType
](arg: OpaquePointer[MutAnyOrigin]) -> OpaquePointer[MutAnyOrigin]:
    """The entry point of a worker thread: runs the job at `arg`."""
    _block[dt](arg.unsafe_bitcast[_Job]()[])
    return arg


def gemm_threads() -> Int:
    """The number of threads a large `gemm` uses: the physical cores, or
    the environment variable `MINIGRAD_THREADS` if it is set."""
    var s = getenv("MINIGRAD_THREADS")
    if s:
        try:
            var n = atol(s)
            if n > 0:
                return n
        except:
            pass
    return num_physical_cores()


def gemm[
    dt: DType
](
    c: BytePtr,
    ci: List[Int],
    cj: List[Int],
    a: BytePtr,
    ai: List[Int],
    ak: List[Int],
    b: BytePtr,
    bk: List[Int],
    bj: List[Int],
    threads: Int,
):
    """Adds a contraction of A and B into C; see the module docstring.

    The microkernel vectorizes along the columns `j`: callers should
    make J the longer side. Threads split whichever of the rows and
    columns makes more tiles.

    Args:
        c: The output; the elements addressed must be initialized.
        ci: The offset of each of the I rows of C, in elements.
        cj: The offset of each of the J columns of C.
        a: The first operand.
        ai: The offset of each of the I rows of A.
        ak: The offset of each of the K terms in A.
        b: The second operand.
        bk: The offset of each of the K terms in B.
        bj: The offset of each of the J columns of B.
        threads: The most threads to use.
    """
    comptime W = simd_width_of[dt]()
    comptime NR = NV * W
    var ni = len(ci)
    var nj = len(cj)
    var nk = len(ak)
    if ni == 0 or nj == 0 or nk == 0:
        return
    var itiles = ni // MR
    var jtiles = nj // NR
    var by_rows = itiles > jtiles
    var workers = 1
    if ni * nj * nk >= PARALLEL_WORK:
        workers = max(1, min(threads, itiles if by_rows else jtiles))
    var jobs = List[_Job](capacity=workers)
    var tids = List[UInt64](length=workers, fill=0)
    for w in range(workers):
        # Whole tiles per thread, so no two threads share an element.
        var i_lo = 0
        var i_hi = ni
        var j_lo = 0
        var j_hi = nj
        if by_rows:
            i_lo = (ni * w // workers) // MR * MR
            if w < workers - 1:
                i_hi = (ni * (w + 1) // workers) // MR * MR
        else:
            j_lo = (nj * w // workers) // NR * NR
            if w < workers - 1:
                j_hi = (nj * (w + 1) // workers) // NR * NR
        jobs.append(
            _Job(
                Int(c),
                Int(a),
                Int(b),
                Int(ci.unsafe_ptr()),
                Int(cj.unsafe_ptr()),
                Int(ai.unsafe_ptr()),
                Int(ak.unsafe_ptr()),
                Int(bk.unsafe_ptr()),
                Int(bj.unsafe_ptr()),
                nk,
                i_lo,
                i_hi,
                j_lo,
                j_hi,
            )
        )
    if workers == 1:
        _block[dt](jobs[0])
    else:
        var entry = _thread_entry[dt]
        for w in range(1, workers):
            _ = external_call["pthread_create", Int32](
                Pointer(to=tids[w]),
                Int(0),
                entry,
                Pointer(to=jobs[w]).unsafe_bitcast[NoneType](),
            )
        _block[dt](jobs[0])
        for w in range(1, workers):
            _ = external_call["pthread_join", Int32](tids[w], Int(0))
