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
in vector registers. Rows and columns of C are independent, so `gemm`
splits the longer side over the thread pool; each thread packs its own
blocks. Up to `GEMV_ROWS` rows, `gemm` instead streams B once, without
packing it: a matrix-vector product is limited by memory, not
arithmetic.
"""

from std.os import getenv
from std.memory import unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.sys import num_physical_cores, simd_width_of, size_of

from minigrad.graph import BytePtr
from minigrad.runtime.pool import run_parallel

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

comptime PARALLEL_WORK = 2_000_000
"""Multiply-adds below which `gemm` stays on the calling thread (about
10 microseconds of work)."""


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


def _block[dt: DType, bdt: DType](job: _Job):
    """Adds the contraction into the job's rows and columns of C. B is
    stored as `bdt` and converted to `dt` as it is packed."""
    comptime W = simd_width_of[dt]()
    comptime NR = NV * W
    comptime NC = NC_VECTORS * W
    comptime P = Pointer[Scalar[dt], MutUntrackedOrigin]
    var ni = job.i_hi - job.i_lo
    var nk = job.nk
    var c = P(unsafe_from_address=job.c)
    var a = P(unsafe_from_address=job.a)
    var b = Pointer[Scalar[bdt], MutUntrackedOrigin](unsafe_from_address=job.b)
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
                                src.unsafe_load[width=W, alignment=1](
                                    v * W
                                ).cast[dt](),
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
                            ].cast[dt]()
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


def _block_task[dt: DType, bdt: DType](ctx: Int, i: Int):
    """Pool task `i`: job `i` of the array at `ctx`."""
    var jobs = Pointer[_Job, MutUntrackedOrigin](unsafe_from_address=ctx)
    _block[dt, bdt](jobs[unsafe_offset=i])


comptime GEMV_PARALLEL_WORK = 1_000_000
"""Multiply-adds from which a matrix-vector product uses threads."""

comptime GEMV_ROWS = 3
"""Rows of C up to which `gemm` streams B once instead of tiling. The
streaming loop keeps each row's sums in memory, so its arithmetic slows
as rows are added; from 4 rows the tiled kernel, which packs B and keeps
an 8 by 32 tile in registers, is faster (measured on GPT-2's MLP: 16
rows took 280 µs streamed and 110 µs tiled)."""

comptime CHUNKS_PER_THREAD = 4
"""Pieces a product splits into per thread, when splitting is free."""


@always_inline
def _dot[
    dt: DType, bdt: DType
](
    x: Pointer[Scalar[dt], MutUntrackedOrigin],
    y: Pointer[Scalar[bdt], MutUntrackedOrigin],
    n: Int,
) -> Scalar[dt]:
    """`sum(x[k] * y[k])` over `n` contiguous elements, `y` converted
    from `bdt`."""
    comptime W = simd_width_of[dt]()
    var a0 = SIMD[dt, W](0)
    var a1 = SIMD[dt, W](0)
    var a2 = SIMD[dt, W](0)
    var a3 = SIMD[dt, W](0)
    var k = 0
    while k + 4 * W <= n:
        a0 = x.unsafe_load[width=W, alignment=1](k).fma(
            y.unsafe_load[width=W, alignment=1](k).cast[dt](), a0
        )
        a1 = x.unsafe_load[width=W, alignment=1](k + W).fma(
            y.unsafe_load[width=W, alignment=1](k + W).cast[dt](), a1
        )
        a2 = x.unsafe_load[width=W, alignment=1](k + 2 * W).fma(
            y.unsafe_load[width=W, alignment=1](k + 2 * W).cast[dt](), a2
        )
        a3 = x.unsafe_load[width=W, alignment=1](k + 3 * W).fma(
            y.unsafe_load[width=W, alignment=1](k + 3 * W).cast[dt](), a3
        )
        k += 4 * W
    while k + W <= n:
        a0 = x.unsafe_load[width=W, alignment=1](k).fma(
            y.unsafe_load[width=W, alignment=1](k).cast[dt](), a0
        )
        k += W
    var s = ((a0 + a1) + (a2 + a3)).reduce_add()
    while k < n:
        s += x[unsafe_offset=k] * y[unsafe_offset=k].cast[dt]()
        k += 1
    return s


def _gemv[dt: DType, bdt: DType](job: _Job):
    """Columns `[j_lo, j_hi)` of C for a few rows, reading B once.

    `job.a` is A packed: `ni` rows of `nk` contiguous elements. B is read
    by columns when its terms are contiguous (a dot product per column),
    by rows when its columns are (a scaled row added per term), and
    element by element otherwise.
    """
    comptime W = simd_width_of[dt]()
    comptime P = Pointer[Scalar[dt], MutUntrackedOrigin]
    var ni = job.i_hi - job.i_lo
    var nk = job.nk
    var c = P(unsafe_from_address=job.c)
    var a = P(unsafe_from_address=job.a)
    var b = Pointer[Scalar[bdt], MutUntrackedOrigin](unsafe_from_address=job.b)
    var ci = _IntPtr(unsafe_from_address=job.ci)
    var cj = _IntPtr(unsafe_from_address=job.cj)
    var bk = _IntPtr(unsafe_from_address=job.bk)
    var bj = _IntPtr(unsafe_from_address=job.bj)
    var nj = job.j_hi - job.j_lo
    var bj_lo = bj.unsafe_offset(job.j_lo)
    var cj_lo = cj.unsafe_offset(job.j_lo)
    if _is_unit(bk, nk):
        var k0 = bk[unsafe_offset=0]
        for j in range(nj):
            var col = b.unsafe_offset(k0 + bj_lo[unsafe_offset=j])
            for i in range(ni):
                c[
                    unsafe_offset=ci[unsafe_offset=i] + cj_lo[unsafe_offset=j]
                ] += _dot[dt, bdt](a.unsafe_offset(i * nk), col, nk)
        return
    if _is_unit(bj_lo, nj):
        var acc = unsafe_alloc[Scalar[dt]](ni * nj + 2 * W, alignment=64)
        unsafe_memset_zero(acc, ni * nj + 2 * W)
        var j0 = bj_lo[unsafe_offset=0]
        for k in range(nk):
            var row = b.unsafe_offset(bk[unsafe_offset=k] + j0)
            for i in range(ni):
                var s = SIMD[dt, W](a[unsafe_offset=i * nk + k])
                var dst = acc.unsafe_offset(i * nj)
                var j = 0
                while j + W <= nj:
                    dst.unsafe_store(
                        j,
                        s.fma(
                            row.unsafe_load[width=W, alignment=1](j).cast[dt](),
                            dst.unsafe_load[width=W, alignment=1](j),
                        ),
                    )
                    j += W
                while j < nj:
                    dst[unsafe_offset=j] += (
                        s[0] * row[unsafe_offset=j].cast[dt]()
                    )
                    j += 1
        for i in range(ni):
            var crow = c.unsafe_offset(ci[unsafe_offset=i])
            for j in range(nj):
                crow[unsafe_offset=cj_lo[unsafe_offset=j]] += acc[
                    unsafe_offset=i * nj + j
                ]
        acc.unsafe_free()
        return
    for i in range(ni):
        var crow = c.unsafe_offset(ci[unsafe_offset=i])
        var arow = a.unsafe_offset(i * nk)
        for j in range(nj):
            var s = Scalar[dt](0)
            var base = b.unsafe_offset(bj_lo[unsafe_offset=j])
            for k in range(nk):
                s += (
                    arow[unsafe_offset=k]
                    * base[unsafe_offset=bk[unsafe_offset=k]].cast[dt]()
                )
            crow[unsafe_offset=cj_lo[unsafe_offset=j]] += s


def _gemv_task[dt: DType, bdt: DType](ctx: Int, i: Int):
    """Pool task `i`: job `i` of the array at `ctx`."""
    var jobs = Pointer[_Job, MutUntrackedOrigin](unsafe_from_address=ctx)
    _gemv[dt, bdt](jobs[unsafe_offset=i])


def _gemv_all[
    dt: DType, bdt: DType
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
    """`gemm` for at most `GEMV_ROWS` rows: packs A, then splits the
    columns over the pool, each thread streaming its share of B once."""
    comptime W = simd_width_of[dt]()
    var ni = len(ci)
    var nj = len(cj)
    var nk = len(ak)
    var pa = a.unsafe_bitcast[Scalar[dt]]()
    var apack = unsafe_alloc[Scalar[dt]](ni * nk + W, alignment=64)
    for i in range(ni):
        var row = pa.unsafe_offset(ai[i])
        for k in range(nk):
            apack[unsafe_offset=i * nk + k] = row[unsafe_offset=ak[k]]
    var workers = 1
    if threads > 1 and ni * nj * nk >= GEMV_PARALLEL_WORK:
        workers = max(1, min(CHUNKS_PER_THREAD * threads, nj // (4 * W)))
    var jobs = List[_Job](capacity=workers)
    for w in range(workers):
        # Column ranges in whole vectors, so threads write whole lines.
        var lo = (nj * w // workers) // W * W
        var hi = nj if w == workers - 1 else (
            (nj * (w + 1) // workers) // W * W
        )
        jobs.append(
            _Job(
                Int(c),
                Int(apack),
                Int(b),
                Int(ci.unsafe_ptr()),
                Int(cj.unsafe_ptr()),
                Int(apack),
                Int(ak.unsafe_ptr()),
                Int(bk.unsafe_ptr()),
                Int(bj.unsafe_ptr()),
                nk,
                0,
                ni,
                lo,
                hi,
            )
        )
    run_parallel[_gemv_task[dt, bdt]](Int(jobs.unsafe_ptr()), workers)
    # The tasks read `jobs` through an address: keep it alive until here.
    _ = jobs^
    apack.unsafe_free()


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
    dt: DType, bdt: DType = dt
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

    B may be stored in a narrower type `bdt` (16-bit floats, `int8`): it
    is converted to `dt` as it is read, never copied whole.

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
    if ni <= GEMV_ROWS:
        _gemv_all[dt, bdt](c, ci, cj, a, ai, ak, b, bk, bj, threads)
        return
    var itiles = ni // MR
    var jtiles = nj // NR
    # Each thread packs the B columns it computes, so splitting by
    # columns packs B once; split by rows only when there are too few
    # column tiles to go round.
    var by_rows = jtiles < min(threads, itiles)
    # Chunks of whole tiles, claimed by the pool's threads as they come
    # free: by columns, several per thread, so a thread the system
    # preempts holds up only a small share; by rows, one per thread, as
    # each row chunk packs all of B.
    var workers = 1
    # `threads == 1` means this call is itself a pool task: run_parallel
    # is not reentrant, so it must stay one chunk.
    if threads > 1 and ni * nj * nk >= PARALLEL_WORK:
        if by_rows:
            workers = max(1, min(threads, itiles))
        else:
            workers = max(1, min(CHUNKS_PER_THREAD * threads, jtiles))
    var jobs = List[_Job](capacity=workers)
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
    run_parallel[_block_task[dt, bdt]](Int(jobs.unsafe_ptr()), workers)
    # The tasks read `jobs` through an address: keep it alive until here.
    _ = jobs^
