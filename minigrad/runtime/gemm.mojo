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
from std.sys import (
    num_physical_cores,
    simd_byte_width,
    simd_width_of,
    size_of,
)
from std.sys.intrinsics import PrefetchOptions, prefetch

from minigrad.graph import BytePtr
from minigrad.runtime.pool import run_parallel

comptime MR = 8 if simd_byte_width() >= 64 else 4
"""Rows of C per microkernel tile. The tile's `MR * NV` accumulators, the
`NV` vectors of B, and A's broadcast must fit in the vector registers:
32 with AVX-512, 16 with AVX2, where an 8-row tile spills and runs at
60% of the speed of a 4-row one."""
comptime NV = 2
"""SIMD vectors of C columns per microkernel tile."""
comptime KC = 256
"""Terms of the sum per packed block: a B panel then fits in L1."""
comptime MC = 96
"""Rows of C per packed A block (a multiple of `MR`)."""
comptime NC_VECTORS = 256
"""SIMD vectors of C columns per packed B block: with `KC` terms, about
4 MB of packed B per thread."""

comptime PREFETCH_ROWS = 32
"""Rows ahead `_micro_direct` prefetches B. Its walk down a column strip
of B, a few lines per row, defeats the hardware prefetcher when B is in
memory rather than cache."""

comptime DIRECT_PANELS = 2
"""Row panels up to which a job reads B in place instead of packing it.
Packing pays for itself by reusing a packed panel for many row panels;
for a few rows (a batch of decode steps) it only adds a pass over B."""

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
def _micro_direct[
    dt: DType, bdt: DType
](
    a: Pointer[Scalar[dt], MutUntrackedOrigin],
    rows: _IntPtr,
    b: Pointer[Scalar[bdt], MutUntrackedOrigin],
    bk: _IntPtr,
    kc: Int,
    tile: Pointer[Scalar[dt], MutUntrackedOrigin],
):
    """`_micro_rows` reading B in place as well: row `k` of the tile's B
    is the `NV * W` consecutive elements at `b + bk[k]`, converted from
    `bdt` as they are loaded."""
    comptime W = simd_width_of[dt]()
    var acc = Array[SIMD[dt, W], MR * NV](fill=SIMD[dt, W](0))
    var ptrs = Array[Pointer[Scalar[dt], MutUntrackedOrigin], MR](fill=a)
    comptime for r in range(MR):
        ptrs[r] = a.unsafe_offset(rows[unsafe_offset=r])
    # The lines of the panel's row PREFETCH_ROWS ahead.
    comptime LINES = (NV * W * size_of[Scalar[bdt]]() + 63) // 64
    comptime STEP = 64 // size_of[Scalar[bdt]]()
    for k in range(kc):
        var pb = b.unsafe_offset(bk[unsafe_offset=k])
        if k + PREFETCH_ROWS < kc:
            var ahead = b.unsafe_offset(bk[unsafe_offset=k + PREFETCH_ROWS])
            comptime for l in range(LINES):
                prefetch[PrefetchOptions().for_read().high_locality()](
                    ahead.unsafe_offset(l * STEP)
                )
        comptime for v in range(NV):
            var bv = pb.unsafe_load[width=W, alignment=1](v * W).cast[dt]()
            comptime for r in range(MR):
                acc[r * NV + v] = SIMD[dt, W](ptrs[r][unsafe_offset=k]).fma(
                    bv, acc[r * NV + v]
                )
    comptime for i in range(MR * NV):
        tile.unsafe_store(i * W, acc[i])


@always_inline
def _is_unit(offsets: _IntPtr, n: Int) -> Bool:
    """Whether the `n` offsets are consecutive."""
    for i in range(1, n):
        if offsets[unsafe_offset=i] != offsets[unsafe_offset=i - 1] + 1:
            return False
    return True


comptime MIN_RUN = 16
"""Shortest run of consecutive terms of A that a block of terms is cut
to, so that A is read in place rather than packed."""

comptime LONG_RUN = 64
"""Runs of terms of A at least this long are always cut at; shorter ones
only when C's columns are consecutive. A cut adds an update of C per
tile: a few vector loads and stores, or, when C's columns are scattered
(a weight gradient laid out by output channel), one per element, which
costs more than packing A."""


@always_inline
def _run_length(offsets: _IntPtr, n: Int) -> Int:
    """How many of the first `n` offsets are consecutive."""
    var i = 1
    while (
        i < n and offsets[unsafe_offset=i] == offsets[unsafe_offset=i - 1] + 1
    ):
        i += 1
    return i


def _block_terms(ak: _IntPtr, kp: Int, nk: Int, c_unit: Bool) -> Int:
    """How many terms, from term `kp`, the next block of `_block` takes:
    `KC`, or fewer to end where a run of consecutive terms of A ends, so
    that A is read in place (a weight gradient's terms run over one
    image's positions)."""
    var kc = min(KC, nk - kp)
    var run = _run_length(ak.unsafe_offset(kp), kc)
    if run == kc or run < MIN_RUN:
        return kc
    # Inside a run a block already split, the run is longer than KC.
    var long = run >= LONG_RUN or (
        kp > 0 and ak[unsafe_offset=kp] == ak[unsafe_offset=kp - 1] + 1
    )
    return run if long or c_unit else kc


def _reads_a_in_place(ak: _IntPtr, nk: Int) -> Bool:
    """Whether every block `_block` takes has consecutive terms in A, for
    any layout of C: then it never packs A."""
    var kp = 0
    while kp < nk:
        var kc = _block_terms(ak, kp, nk, False)
        if not _is_unit(ak.unsafe_offset(kp), kc):
            return False
        kp += kc
    return True


def _max(offsets: _IntPtr, n: Int) -> Int:
    """The largest of the `n` offsets."""
    var m = offsets[unsafe_offset=0]
    for i in range(1, n):
        m = max(m, offsets[unsafe_offset=i])
    return m


@always_inline
def _pack_runs[
    dt: DType, bdt: DType, NR: Int
](
    dst: Pointer[Scalar[dt], MutUntrackedOrigin],
    b: Pointer[Scalar[bdt], MutUntrackedOrigin],
    bk: _IntPtr,
    kc: Int,
    bj: _IntPtr,
    nr: Int,
    last_b: Int,
    runs: _IntPtr,
) -> Bool:
    """Packs the `kc` by `nr` panel of B at rows `bk` and columns `bj`
    into `dst` (`NR` elements per row) by runs of consecutive columns,
    with vector loads: a convolution's window over rows of an image is a
    few runs. Returns False, packing nothing, when the runs average under
    two columns (a transposed B), which the caller packs element by
    element.

    A run shorter than a vector is loaded and stored as a whole vector;
    the excess lands where later runs, later rows, or later panels are
    stored, so `dst` must be packed in order and have a vector of slack
    at the end. The load reads past the run only when that stays at or
    below `last_b`, B's last element; otherwise the run is copied
    element by element."""
    comptime W = simd_width_of[dt]()
    # Runs as (first column's offset, position in the panel, length).
    var n = 0
    var j = 0
    while j < nr:
        var first = bj[unsafe_offset=j]
        var length = 1
        while (
            j + length < nr and bj[unsafe_offset=j + length] == first + length
        ):
            length += 1
        runs[unsafe_offset=3 * n] = first
        runs[unsafe_offset=3 * n + 1] = j
        runs[unsafe_offset=3 * n + 2] = length
        n += 1
        j += length
    if 2 * n > nr:
        return False
    for k in range(kc):
        var row_offset = bk[unsafe_offset=k]
        var row = b.unsafe_offset(row_offset)
        var d = dst.unsafe_offset(k * NR)
        for q in range(n):
            var first = runs[unsafe_offset=3 * q]
            var at = runs[unsafe_offset=3 * q + 1]
            var length = runs[unsafe_offset=3 * q + 2]
            var src = row.unsafe_offset(first)
            var to = d.unsafe_offset(at)
            if length >= W:
                # Whole vectors, the last one overlapping the one before.
                var t = 0
                while t + W <= length:
                    to.unsafe_store(
                        t, src.unsafe_load[width=W, alignment=1](t).cast[dt]()
                    )
                    t += W
                if t < length:
                    to.unsafe_store(
                        length - W,
                        src.unsafe_load[width=W, alignment=1](length - W).cast[
                            dt
                        ](),
                    )
            elif row_offset + first + W - 1 <= last_b:
                to.unsafe_store(
                    0, src.unsafe_load[width=W, alignment=1](0).cast[dt]()
                )
            else:
                for t in range(length):
                    to[unsafe_offset=t] = src[unsafe_offset=t].cast[dt]()
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


comptime DOT_COLS = 3
"""Columns of B per tile of `_block_dots`: `MR * DOT_COLS` accumulators
(24 with AVX-512, 12 with AVX2), which leaves registers for the
operands."""


@always_inline
def _dot_tile[
    dt: DType, bdt: DType, NJ: Int
](
    rows_a: Array[Pointer[Scalar[dt], MutUntrackedOrigin], MR],
    cols_b: Array[Pointer[Scalar[bdt], MutUntrackedOrigin], NJ],
    nk: Int,
    out_tile: Pointer[Scalar[dt], MutUntrackedOrigin],
):
    """`out_tile[r * NJ + q] = sum_k rows_a[r][k] * cols_b[q][k]`: `MR`
    by `NJ` dot products over `nk` consecutive terms, B converted from
    `bdt` as it is loaded."""
    comptime W = simd_width_of[dt]()
    var acc = Array[SIMD[dt, W], MR * NJ](fill=SIMD[dt, W](0))
    var k = 0
    while k + W <= nk:
        var bv = Array[SIMD[dt, W], NJ](fill=SIMD[dt, W](0))
        comptime for q in range(NJ):
            bv[q] = cols_b[q].unsafe_load[width=W, alignment=1](k).cast[dt]()
        comptime for r in range(MR):
            var av = rows_a[r].unsafe_load[width=W, alignment=1](k)
            comptime for q in range(NJ):
                acc[r * NJ + q] = av.fma(bv[q], acc[r * NJ + q])
        k += W
    comptime for r in range(MR):
        comptime for q in range(NJ):
            var t = acc[r * NJ + q].reduce_add()
            for kk in range(k, nk):
                t += (
                    rows_a[r][unsafe_offset=kk]
                    * cols_b[q][unsafe_offset=kk].cast[dt]()
                )
            out_tile[unsafe_offset=r * NJ + q] = t


def _block_dots[dt: DType, bdt: DType](job: _Job):
    """`_block` for a few rows when the terms are consecutive in both A
    and B (B stored `(out, in)`, as in `x @ w.T`): tiles of dot products
    vectorized along the terms, both operands read in place."""
    comptime P = Pointer[Scalar[dt], MutUntrackedOrigin]
    comptime PB = Pointer[Scalar[bdt], MutUntrackedOrigin]
    var nk = job.nk
    var c = P(unsafe_from_address=job.c)
    var a = P(unsafe_from_address=job.a)
    var b = PB(unsafe_from_address=job.b)
    var ci = _IntPtr(unsafe_from_address=job.ci)
    var cj = _IntPtr(unsafe_from_address=job.cj)
    var ai = _IntPtr(unsafe_from_address=job.ai)
    var bj = _IntPtr(unsafe_from_address=job.bj)
    var ak0 = _IntPtr(unsafe_from_address=job.ak)[unsafe_offset=0]
    var bk0 = _IntPtr(unsafe_from_address=job.bk)[unsafe_offset=0]
    # Missing rows of a short tile read zeros.
    var zeros = unsafe_alloc[Scalar[dt]](nk + 64, alignment=64)
    unsafe_memset_zero(zeros, nk + 64)
    var tile = unsafe_alloc[Scalar[dt]](MR * DOT_COLS + MR, alignment=64)
    var i = job.i_lo
    while i < job.i_hi:
        var rows = min(MR, job.i_hi - i)
        var ra = Array[P, MR](fill=zeros)
        for r in range(rows):
            ra[r] = a.unsafe_offset(ai[unsafe_offset=i + r] + ak0)
        var j = job.j_lo
        while j < job.j_hi:
            var nj = min(DOT_COLS, job.j_hi - j)
            if nj == DOT_COLS:
                var cb = Array[PB, DOT_COLS](fill=b)
                comptime for q in range(DOT_COLS):
                    cb[q] = b.unsafe_offset(bk0 + bj[unsafe_offset=j + q])
                _dot_tile[dt, bdt, DOT_COLS](ra, cb, nk, tile)
            else:
                # Leftover columns one at a time, through the scratch
                # area after the tile.
                var one = tile.unsafe_offset(MR * DOT_COLS)
                for q in range(nj):
                    var cb = Array[PB, 1](
                        fill=b.unsafe_offset(bk0 + bj[unsafe_offset=j + q])
                    )
                    _dot_tile[dt, bdt, 1](ra, cb, nk, one)
                    for r in range(MR):
                        tile[unsafe_offset=r * DOT_COLS + q] = one[
                            unsafe_offset=r
                        ]
            for r in range(rows):
                var crow = c.unsafe_offset(ci[unsafe_offset=i + r])
                for q in range(nj):
                    crow[unsafe_offset=cj[unsafe_offset=j + q]] += tile[
                        unsafe_offset=r * DOT_COLS + q
                    ]
            j += nj
        i += rows
    zeros.unsafe_free()
    tile.unsafe_free()


def _block[dt: DType, bdt: DType](job: _Job):
    """Adds the contraction into the job's rows and columns of C. B is
    stored as `bdt` and converted to `dt` as it is packed."""
    comptime W = simd_width_of[dt]()
    comptime NR = NV * W
    comptime NC = NC_VECTORS * W
    comptime P = Pointer[Scalar[dt], MutUntrackedOrigin]
    var ni = job.i_hi - job.i_lo
    var nk = job.nk
    if (
        (ni + MR - 1) // MR <= DIRECT_PANELS
        and _is_unit(_IntPtr(unsafe_from_address=job.ak), nk)
        and _is_unit(_IntPtr(unsafe_from_address=job.bk), nk)
    ):
        _block_dots[dt, bdt](job)
        return
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
    # Few rows read in place (A's terms consecutive): B panels whose
    # columns are consecutive are read in place too, not packed.
    var direct = (ni + MR - 1) // MR <= DIRECT_PANELS and _is_unit(
        _IntPtr(unsafe_from_address=job.ak), nk
    )
    # A panel of B that is not one run of consecutive columns is packed
    # run by run (`_pack_runs`); a short run's vector load may read past
    # it, but never past B's last element, `last_b`.
    var last_b = _max(_IntPtr(unsafe_from_address=job.bk), nk) + _max(
        bj.unsafe_offset(job.j_lo), job.j_hi - job.j_lo
    )
    var runs = unsafe_alloc[Int](3 * NR, alignment=64)
    var c_unit = _is_unit(cj.unsafe_offset(job.j_lo), job.j_hi - job.j_lo)
    var jc = job.j_lo
    while jc < job.j_hi:
        var nc = min(NC, job.j_hi - jc)
        var jpanels = (nc + NR - 1) // NR
        var kp = 0
        while kp < nk:
            var kc = _block_terms(
                _IntPtr(unsafe_from_address=job.ak),
                kp,
                nk,
                c_unit,
            )
            var ak = _IntPtr(unsafe_from_address=job.ak).unsafe_offset(kp)
            var bk = _IntPtr(unsafe_from_address=job.bk).unsafe_offset(kp)
            # Pack B[kp:kp+kc, jc:jc+nc] into panels of NR columns, kc by
            # NR each, padding the last panel with zeros.
            for p in range(jpanels):
                var dst = bpack.unsafe_offset(p * kc * NR)
                var j0 = jc + p * NR
                var nr = min(NR, jc + nc - j0)
                var unit = nr == NR and _is_unit(bj.unsafe_offset(j0), NR)
                if direct and unit:
                    continue
                if unit:
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
                    if _pack_runs[dt, bdt, NR](
                        dst, b, bk, kc, bj.unsafe_offset(j0), nr, last_b, runs
                    ):
                        continue
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
                    # Term by term: A's terms are not consecutive here, so
                    # its rows usually are (a transpose), and a panel's
                    # MR values of one term are one load.
                    for p in range(panels):
                        var dst = apack.unsafe_offset(p * kc * MR)
                        var i0 = ic + p * MR
                        var rows = min(MR, ic + mc - i0)
                        if rows == MR and _is_unit(ai.unsafe_offset(i0), MR):
                            var src = a.unsafe_offset(ai[unsafe_offset=i0])
                            for k in range(kc):
                                dst.unsafe_store(
                                    k * MR,
                                    src.unsafe_load[width=MR, alignment=1](
                                        ak[unsafe_offset=k]
                                    ),
                                )
                        else:
                            if rows < MR:
                                unsafe_memset_zero(dst, kc * MR)
                            for k in range(kc):
                                var src = a.unsafe_offset(ak[unsafe_offset=k])
                                var d = dst.unsafe_offset(k * MR)
                                for r in range(rows):
                                    d[unsafe_offset=r] = src[
                                        unsafe_offset=ai[unsafe_offset=i0 + r]
                                    ]
                for jp in range(jpanels):
                    var j0 = jc + jp * NR
                    var nr = min(NR, jc + nc - j0)
                    var bpanel = bpack.unsafe_offset(jp * kc * NR)
                    var cj_unit = nr == NR and _is_unit(
                        cj.unsafe_offset(j0), NR
                    )
                    var in_place = (
                        direct
                        and nr == NR
                        and _is_unit(bj.unsafe_offset(j0), NR)
                    )
                    for p in range(panels):
                        var rows = min(MR, mc - p * MR)
                        if in_place:
                            # Rows of A by offset from `zeros`, missing
                            # rows reading zeros, as below.
                            var base = (Int(a) - Int(zeros)) // size_of[
                                Scalar[dt]
                            ]()
                            for r in range(MR):
                                edge[unsafe_offset=r] = (
                                    ai[unsafe_offset=ic + p * MR + r]
                                    + ak0
                                    + base
                                ) if r < rows else 0
                            _micro_direct[dt, bdt](
                                zeros,
                                edge,
                                b.unsafe_offset(bj[unsafe_offset=j0]),
                                bk,
                                kc,
                                tile,
                            )
                        elif not ak_unit:
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
    runs.unsafe_free()


def _block_task[dt: DType, bdt: DType](ctx: Int, i: Int):
    """Pool task `i`: job `i` of the array at `ctx`."""
    var jobs = Pointer[_Job, MutUntrackedOrigin](unsafe_from_address=ctx)
    _block[dt, bdt](jobs[unsafe_offset=i])


comptime GEMV_PARALLEL_BYTES = 256 * 1024
"""Bytes of B from which a matrix-vector product uses threads: it is
limited by reading B, which one thread does at about half the memory's
bandwidth."""

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


comptime STRIP = 8
"""Vectors of columns per strip of `_gemv`'s row loop, their sums held in
registers."""


@always_inline
def _gemv_strip[
    dt: DType, bdt: DType, V: Int
](
    arow: Pointer[Scalar[dt], MutUntrackedOrigin],
    b: Pointer[Scalar[bdt], MutUntrackedOrigin],
    bk: _IntPtr,
    nk: Int,
    crow: Pointer[Scalar[dt], MutUntrackedOrigin],
    cj: _IntPtr,
    j: Int,
):
    """Adds `sum_k arow[k] * b[bk[k] + v]` for the `V * W` consecutive
    columns at `b` into C's row `crow`, at columns `j ..` of `cj`."""
    comptime W = simd_width_of[dt]()
    var acc = Array[SIMD[dt, W], V](fill=SIMD[dt, W](0))
    for k in range(nk):
        var s = SIMD[dt, W](arow[unsafe_offset=k])
        var row = b.unsafe_offset(bk[unsafe_offset=k])
        comptime for v in range(V):
            acc[v] = s.fma(
                row.unsafe_load[width=W, alignment=1](v * W).cast[dt](), acc[v]
            )
    comptime for v in range(V):
        comptime for l in range(W):
            crow[unsafe_offset=cj[unsafe_offset=j + v * W + l]] += acc[v][l]


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
        # Strips of columns, their sums in registers across all the
        # terms: a running sum in memory would cost a load and a store
        # per vector of B, four times as many per byte for int8.
        var j0 = bj_lo[unsafe_offset=0]
        for i in range(ni):
            var arow = a.unsafe_offset(i * nk)
            var crow = c.unsafe_offset(ci[unsafe_offset=i])
            var j = 0
            comptime for e in range(4):
                comptime V = STRIP >> e
                while j + V * W <= nj:
                    _gemv_strip[dt, bdt, V](
                        arow, b.unsafe_offset(j0 + j), bk, nk, crow, cj_lo, j
                    )
                    j += V * W
            while j < nj:
                var sum = Scalar[dt](0)
                var col = b.unsafe_offset(j0 + j)
                for k in range(nk):
                    sum += (
                        arow[unsafe_offset=k]
                        * col[unsafe_offset=bk[unsafe_offset=k]].cast[dt]()
                    )
                crow[unsafe_offset=cj_lo[unsafe_offset=j]] += sum
                j += 1
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
    # Column ranges in whole vectors of C and whole cache lines of B (64
    # int8 columns), so that no two threads read the same line of B.
    comptime UNIT = max(W, 64 // size_of[Scalar[bdt]]())
    var workers = 1
    if threads > 1 and nk * nj * size_of[Scalar[bdt]]() >= GEMV_PARALLEL_BYTES:
        workers = max(
            1, min(CHUNKS_PER_THREAD * threads, nj // max(4 * W, UNIT))
        )
    var jobs = List[_Job](capacity=workers)
    for w in range(workers):
        var lo = (nj * w // workers) // UNIT * UNIT
        var hi = nj if w == workers - 1 else (
            (nj * (w + 1) // workers) // UNIT * UNIT
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
        # Every column chunk packs all of A when it cannot read A in
        # place: then one chunk per thread.
        var chunks = CHUNKS_PER_THREAD if _reads_a_in_place(
            _IntPtr(unsafe_from_address=Int(ak.unsafe_ptr())), nk
        ) else 1
        if by_rows:
            workers = max(1, min(threads, itiles))
        else:
            workers = max(1, min(chunks * threads, jtiles))
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
