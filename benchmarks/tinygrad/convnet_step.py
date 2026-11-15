"""The tinygrad counterpart of benchmarks/convnet_step.mojo.

    DEV=CPU python3 benchmarks/tinygrad/convnet_step.py [BATCH] [STEPS] [jit]

Run from a tinygrad checkout (or with it on PYTHONPATH).
"""
import sys, time
from tinygrad import Tensor, TinyJit, nn, Device, Context

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 512
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
use_jit = len(sys.argv) > 3 and sys.argv[3] == "jit"

Tensor.manual_seed(0)
x = Tensor.rand(batch, 1, 28, 28).realize()
y = Tensor.randint(batch, low=0, high=10).realize()
layers = [
  nn.Conv2d(1, 32, 5), Tensor.relu,
  nn.Conv2d(32, 32, 5), Tensor.relu,
  nn.BatchNorm(32), Tensor.max_pool2d,
  nn.Conv2d(32, 64, 3), Tensor.relu,
  nn.Conv2d(64, 64, 3), Tensor.relu,
  nn.BatchNorm(64), Tensor.max_pool2d,
  lambda t: t.flatten(1), nn.Linear(576, 10)]
opt = nn.optim.Adam(nn.state.get_parameters(layers))

@Context(TRAINING=1)
def step():
  opt.zero_grad()
  loss = x.sequential(layers).sparse_categorical_crossentropy(y).backward()
  return loss.realize(*opt.schedule_step())

if use_jit: step = TinyJit(step)

times = []
for _ in range(steps):
  t0 = time.perf_counter_ns()
  loss = step().item()
  times.append(time.perf_counter_ns() - t0)
first = times[0]
times.sort()
print(f"device {Device.DEFAULT} jit {use_jit} batch {batch} steps {steps} final loss {loss}")
print(f"median step: {times[len(times)//2]/1e6:.1f} ms   min: {times[0]/1e6:.1f} ms   first step: {first/1e6:.1f} ms")
