"""Time per training step of a two-layer MLP on random data.

    mojo run -I . benchmarks/mlp_step.mojo [BATCH] [HIDDEN] [STEPS]

Prints the median step time: forward, cross-entropy, backward, and an
SGD update with momentum, including reading the loss. The matching
tinygrad program is benchmarks/tinygrad/mlp_step.py.
"""

from std.sys import argv
from std.time import perf_counter_ns

from minigrad.graph import graph
from minigrad.nn import Linear, SGD, collect
from minigrad.tensor import Tensor


def main() raises:
    var args = argv()
    var batch = atol(args[1]) if len(args) > 1 else 64
    var hidden = atol(args[2]) if len(args) > 2 else 128
    var steps = atol(args[3]) if len(args) > 3 else 200
    Tensor.manual_seed(0)
    var x = Tensor.rand(batch, 784)
    var y = Tensor.randint([batch], 0, 10)
    var l1 = Linear(784, hidden)
    var l2 = Linear(hidden, 10)
    var opt = SGD(collect(l1, l2), lr=0.01, momentum=0.9)
    var times = List[Int]()
    var loss = 0.0
    var total = 0
    for i in range(steps):
        if i == steps // 2:
            # Time the second half only: plans are built by then.
            graph()[].stats.ns_group = 0
            graph()[].stats.ns_build = 0
            graph()[].stats.ns_execute = 0
            total = 0
        var t0 = perf_counter_ns()
        opt.zero_grad()
        var l = l2(l1(x).relu()).sparse_categorical_crossentropy(y)
        l.backward()
        opt.step([l])
        loss = l.item()
        var dt = Int(perf_counter_ns() - t0)
        times.append(dt)
        total += dt
    sort(times)
    var st = graph()[].stats.copy()
    var half = Float64(steps - steps // 2) * 1e3
    print(
        "per step (us): graph construction",
        Float64(total - st.ns_group - st.ns_build - st.ns_execute) / half,
        " grouping",
        Float64(st.ns_group) / half,
        " kernel building",
        Float64(st.ns_build) / half,
        " execution",
        Float64(st.ns_execute) / half,
    )
    print("batch", batch, "hidden", hidden, "steps", steps, "final loss", loss)
    print(
        "median step:",
        Float64(times[len(times) // 2]) / 1e3,
        "us   min:",
        Float64(times[0]) / 1e3,
        "us   first:",
        "(includes plan building)",
    )
    print(
        "kernels per step:",
        st.kernels_run // steps,
        "  plans built in total:",
        st.kernels_compiled,
    )
