"""A persistent thread pool (`minigrad.runtime.pool`).

`run_parallel[f](ctx, n)` runs `f(ctx, i)` for `i` in `0..n` on the
pool's threads and the calling thread, and returns when all are done.
The threads are started on first use. Between jobs they spin briefly,
then sleep on a futex, so an idle process uses no CPU and a burst of
small jobs pays no wake-up latency. At exit the Mojo runtime destroys
the pool, which stops and joins the threads: under `mojo run` the
program's code is unmapped after that, and a thread still spinning in it
would crash the process.

Tasks are claimed with a compare-and-swap on a counter tagged with the
job's generation, so a thread that wakes late cannot claim a task of a
job that has already moved on.

`MINIGRAD_THREADS` caps the threads, the caller included (1: no pool).
"""

from std.atomic import Atomic
from std.ffi import _get_global, external_call
from std.memory.alloc import unsafe_alloc
from std.os import abort, getenv
from std.sys import num_physical_cores
from std.time import perf_counter_ns

comptime TaskFn = def(Int, Int) thin -> None
"""A task: `f(ctx, i)` for task index `i`."""

comptime _SPIN_NS = 2_000_000
"""How long a thread polls for the next job before it sleeps. Waking a
sleeping thread can take most of a millisecond, longer than many jobs,
and the gaps between a program's jobs (building the next graph) are
often a few hundred microseconds."""


def _noop(ctx: Int, i: Int):
    pass


struct _State:
    """The memory the threads share. Allocated once and never freed:
    threads read it until the process ends."""

    var gen: Int32
    """The current job; changes when a job is posted."""
    var sleepers: Int32
    """Threads asleep on `gen`."""
    var _pad0: SIMD[DType.int64, 8]
    var next: Int64
    """`gen << 32 | the next task to claim`."""
    var _pad1: SIMD[DType.int64, 8]
    var done: Int64
    """Tasks of the current job completed."""
    var _pad2: SIMD[DType.int64, 8]
    var ntasks: Int64
    var ctx: Int
    var task: TaskFn
    var stop: Int32
    """Set at exit: threads return."""
    var workers: Int
    var tids: Pointer[UInt64, MutUntrackedOrigin]


comptime _StatePtr = Pointer[_State, MutUntrackedOrigin]


def _futex(addr: Pointer[Int32, MutUntrackedOrigin], op: Int, val: Int):
    _ = external_call["syscall", Int, num_fixed_args=1](
        202, Int(addr), op, val, 0, 0, 0
    )


def _run_tasks(s: _StatePtr, gen: Int64):
    """Claims and runs tasks of job `gen` until none are left."""
    var f = s[].task
    var ctx = s[].ctx
    var n = s[].ntasks
    while True:
        var v = Atomic[Int64].load(Pointer(to=s[].next))
        if (v >> 32) != gen or (v & 0xFFFFFFFF) >= n:
            return
        if Atomic[Int64].compare_exchange(Pointer(to=s[].next), v, v + 1):
            f(ctx, Int(v & 0xFFFFFFFF))
            _ = Atomic[Int64].fetch_add(Pointer(to=s[].done), 1)


def _worker(arg: OpaquePointer[MutAnyOrigin]) -> OpaquePointer[MutAnyOrigin]:
    var s = _StatePtr(unsafe_from_address=Int(arg))
    var seen = Atomic[Int32].load(Pointer(to=s[].gen))
    while True:
        var spins = 0
        var deadline = Int(perf_counter_ns()) + _SPIN_NS
        var g = Atomic[Int32].load(Pointer(to=s[].gen))
        if Atomic[Int32].load(Pointer(to=s[].stop)) != 0:
            return arg
        while g == seen:
            spins += 1
            if (spins & 1023) == 0 and Int(perf_counter_ns()) > deadline:
                _ = Atomic[Int32].fetch_add(Pointer(to=s[].sleepers), 1)
                # Sleeps only if the generation is still `seen`.
                _futex(
                    Pointer(to=s[].gen), 128, Int(seen)
                )  # FUTEX_WAIT_PRIVATE
                _ = Atomic[Int32].fetch_add(Pointer(to=s[].sleepers), -1)
                deadline = Int(perf_counter_ns()) + _SPIN_NS
            g = Atomic[Int32].load(Pointer(to=s[].gen))
        seen = g
        if Atomic[Int32].load(Pointer(to=s[].stop)) != 0:
            return arg
        _run_tasks(s, Int64(g))


comptime _Opaque = Optional[Pointer[NoneType, UntrackedOrigin[mut=True]]]


def _threads() -> Int:
    var threads = num_physical_cores()
    var env = getenv("MINIGRAD_THREADS")
    if env:
        try:
            var n = atol(env)
            if n > 0:
                threads = n
        except:
            pass
    return threads


def _pool_init() -> _Opaque:
    """Starts the threads; the global's value is the `_State`."""
    var workers = max(0, _threads() - 1)
    var s = unsafe_alloc[_State](1, alignment=64)
    s[].gen = 0
    s[].sleepers = 0
    s[].next = 0
    s[].done = 0
    s[].ntasks = 0
    s[].ctx = 0
    s[].task = _noop
    s[].stop = 0
    s[].workers = workers
    s[].tids = unsafe_alloc[UInt64](max(1, workers))
    var entry = _worker
    for w in range(workers):
        _ = external_call["pthread_create", Int32](
            s[].tids.unsafe_offset(w),
            Int(0),
            entry,
            OpaquePointer[MutAnyOrigin](unsafe_from_address=Int(s)),
        )
    return s.unsafe_bitcast[NoneType]()


def _pool_destroy(p: _Opaque):
    """Stops and joins the threads. The `_State` is left allocated."""
    if case .Some(ptr) = p:
        var s = _StatePtr(unsafe_from_address=Int(ptr))
        Atomic[Int32].store(Pointer(to=s[].stop), 1)
        _ = Atomic[Int32].fetch_add(Pointer(to=s[].gen), 1)
        _futex(Pointer(to=s[].gen), 129, 1 << 30)  # FUTEX_WAKE_PRIVATE
        for w in range(s[].workers):
            _ = external_call["pthread_join", Int32](
                s[].tids[unsafe_offset=w], Int(0)
            )


@always_inline
def _state() -> _StatePtr:
    var p = _get_global["minigrad_pool", _pool_init, _pool_destroy]()
    if not p:
        abort("minigrad: cannot start the thread pool")
    return _StatePtr(unsafe_from_address=Int(p.value()))


def pool_threads() -> Int:
    """The threads a job runs on, the caller included."""
    return _state()[].workers + 1


def run_parallel[task: TaskFn](ctx: Int, n: Int):
    """Runs `task(ctx, i)` for `i` in `0..n`, in parallel, and waits.

    Not reentrant: `task` must not call `run_parallel`. Call it from one
    thread at a time.

    Parameters:
        task: The task.

    Args:
        ctx: Passed to every task (an address, typically).
        n: The number of tasks.
    """
    if n <= 0:
        return
    if n == 1:
        task(ctx, 0)
        return
    var s = _state()
    if s[].workers == 0:
        for i in range(n):
            task(ctx, i)
        return
    var gen = Int64(s[].gen) + 1
    s[].task = task
    s[].ctx = ctx
    s[].ntasks = Int64(n)
    Atomic[Int64].store(Pointer(to=s[].done), 0)
    Atomic[Int64].store(Pointer(to=s[].next), (gen & 0x7FFFFFFF) << 32)
    var g32 = Int32(gen & 0x7FFFFFFF)
    Atomic[Int32].store(Pointer(to=s[].gen), g32)
    if Atomic[Int32].load(Pointer(to=s[].sleepers)) > 0:
        _futex(Pointer(to=s[].gen), 129, 1 << 30)  # FUTEX_WAKE_PRIVATE
    _run_tasks(s, Int64(g32))
    while Atomic[Int64].load(Pointer(to=s[].done)) < Int64(n):
        pass
