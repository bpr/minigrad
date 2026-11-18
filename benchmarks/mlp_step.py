"""benchmarks/tinygrad/mlp_step.py on minigrad's Python bindings: the
same program with minigrad's imports, to measure what Python adds over
benchmarks/mlp_step.mojo.

    pixi run python-build
    PYTHONPATH=python python benchmarks/mlp_step.py [BATCH] [HIDDEN] [STEPS] [jit]
"""
import sys, time
from minigrad import Tensor, TinyJit, nn

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 64
hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 128
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
use_jit = len(sys.argv) > 4 and sys.argv[4] == "jit"

Tensor.manual_seed(0)
x = Tensor.rand(batch, 784).realize()
y = Tensor.randint(batch, low=0, high=10).realize()
l1, l2 = nn.Linear(784, hidden), nn.Linear(hidden, 10)
opt = nn.optim.SGD(nn.state.get_parameters([l1, l2]), lr=0.01, momentum=0.9)

@Tensor.train()
def step():
  opt.zero_grad()
  loss = l2(l1(x).relu()).sparse_categorical_crossentropy(y).backward()
  return loss.realize(*opt.schedule_step())

if use_jit: step = TinyJit(step)

times = []
for _ in range(steps):
  t0 = time.perf_counter_ns()
  loss = step().item()
  times.append(time.perf_counter_ns() - t0)
times = sorted(times[steps // 2:])  # the second half: plans are built by then
print(f"minigrad python jit {use_jit} batch {batch} hidden {hidden} steps {steps} final loss {loss}")
print(f"median step: {times[len(times)//2]/1e3:.1f} us   min: {times[0]/1e3:.1f} us")
