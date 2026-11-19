"""The matrix-multiply kernel alone (`minigrad.runtime.gemm`), float32.

    mojo build -I . -o build/gemm_bench benchmarks/gemm_bench.mojo
    build/gemm_bench [REPEATS]

For each product, on one thread and on `gemm_threads()` threads, prints
the median and best time over REPEATS runs (default 21) and the best
rate in GFLOPS.
"""

from std.memory import unsafe_memset_zero
from std.memory.alloc import unsafe_alloc
from std.sys import argv
from std.time import perf_counter_ns

from minigrad.graph import BytePtr
from minigrad.runtime.gemm import gemm, gemm_threads


def _arange(n: Int, step: Int) -> List[Int]:
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(i * step)
    return out^


def _time(
    name: String,
    n: Int,
    k: Int,
    m: Int,
    transposed_b: Bool,
    threads: Int,
    repeats: Int,
) raises:
    """Times C (n, m) = A (n, k) times B, where B is (k, m), or (m, k)
    read transposed."""
    var a = unsafe_alloc[Float32](n * k)
    var b = unsafe_alloc[Float32](k * m)
    var c = unsafe_alloc[Float32](n * m)
    for i in range(n * k):
        a[unsafe_offset=i] = Float32(i % 7) * 0.1
    for i in range(k * m):
        b[unsafe_offset=i] = Float32(i % 5) * 0.1
    var bk = _arange(k, 1) if transposed_b else _arange(k, m)
    var bj = _arange(m, k) if transposed_b else _arange(m, 1)
    var times = List[Int]()
    for _ in range(repeats):
        unsafe_memset_zero(c, n * m)
        var t0 = perf_counter_ns()
        gemm[DType.float32](
            BytePtr(unsafe_from_address=Int(c)),
            _arange(n, m),
            _arange(m, 1),
            BytePtr(unsafe_from_address=Int(a)),
            _arange(n, k),
            _arange(k, 1),
            BytePtr(unsafe_from_address=Int(b)),
            bk,
            bj,
            threads,
        )
        times.append(Int(perf_counter_ns() - t0))
    sort(times)
    var best = Float64(times[0])
    print(
        t"{name}  threads {threads}  median"
        t" {Float64(times[len(times) // 2]) / 1e3} us  best {best / 1e3} us "
        t" {2.0 * Float64(n * k * m) / best} GFLOPS"
    )
    a.unsafe_free()
    b.unsafe_free()
    c.unsafe_free()


def main() raises:
    var args = argv()
    var repeats = atol(args[1]) if len(args) > 1 else 21
    var all = gemm_threads()
    _time("64 x 784 times 784 x 128", 64, 784, 128, False, 1, repeats)
    for threads in [1, all]:
        _time(
            "512 x 784 times 784 x 512", 512, 784, 512, False, threads, repeats
        )
    _time(
        "512 x 784 times (512 x 784) transposed",
        512,
        784,
        512,
        True,
        all,
        repeats,
    )
    for threads in [1, all]:
        _time(
            "1024 x 1024 times 1024 x 1024",
            1024,
            1024,
            1024,
            False,
            threads,
            repeats,
        )
