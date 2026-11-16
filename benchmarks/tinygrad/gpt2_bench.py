"""tinygrad's GPT-2 (examples/gpt2.py) on minigrad's weights: the
reference for checking minigrad's output and the baseline for
benchmarks/gpt2_bench.mojo.

    cd <tinygrad checkout>
    PYTHONPATH=. DEV=CPU python3 <minigrad>/benchmarks/tinygrad/gpt2_bench.py \
        <minigrad>/data/gpt2/model.safetensors ACTION TOKEN_IDS...

ACTION is `logits` (print the last position's logits, rounded, for the
given ids), `greedy N` (the ids of N greedy tokens after the given ids),
or `bench N` (time the prompt and N greedy decode steps, with TinyJit).
Token ids rather than text, so tinygrad's run needs no tiktoken.
"""
import os, sys, time
os.environ.setdefault("MAX_CONTEXT", "1024")
sys.path.insert(0, os.getcwd())
from tinygrad import Tensor, Variable
from tinygrad.nn.state import safe_load, load_state_dict
from examples.gpt2 import Transformer, MODEL_PARAMS, MAX_CONTEXT

path, action = sys.argv[1], sys.argv[2]
args = [int(a) for a in sys.argv[3:]]

weights = {k: v.to(None) for k, v in safe_load(path).items() if not k.endswith(".attn.bias")}
for k in list(weights):
  if k.endswith(("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")):
    weights[k] = weights[k].T
weights["lm_head.weight"] = weights["wte.weight"]
model = Transformer(**MODEL_PARAMS["gpt2"])
load_state_dict(model, weights, verbose=False)

def logits_of(ids):
  # forward() returns the chosen ids; capture the head's output on the way.
  captured, head = {}, model.lm_head
  def capture(x):
    captured["logits"] = head(x)
    return captured["logits"]
  model.lm_head = capture
  model.forward(Tensor([ids]), Variable("start_pos", 0, MAX_CONTEXT - 1).bind(0), 0.0)
  model.lm_head = head
  return captured["logits"][0, -1].tolist()

def step(toks, start_pos):
  t = Variable("tokens", 0, 50256).bind(toks[0]) if len(toks) == 1 else Tensor([toks])
  return model(t, Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT - 1).bind(start_pos), 0.0).tolist()[0]

if action == "logits":
  print(" ".join(f"{v:.4f}" for v in logits_of(args)))
elif action == "greedy":
  n, ids = args[0], args[1:]
  out, pos = [], 0
  for _ in range(n):
    t = step(ids[pos:] if pos == 0 else [out[-1]], pos)
    pos = len(ids) + len(out)
    out.append(t)
  print(" ".join(map(str, out)))
elif action == "bench":
  n, ids = args[0], args[1:]
  t0 = time.perf_counter()
  tok = step(ids, 0)
  t1 = time.perf_counter()
  tok = step(ids, 0)  # again, with its kernels compiled
  t2 = time.perf_counter()
  pos, times = len(ids), []
  for _ in range(n):
    s = time.perf_counter()
    tok = step([tok], pos)
    times.append(time.perf_counter() - s)
    pos += 1
  first = times[0]
  times.sort()
  print(f"prompt of {len(ids)} tokens: {(t1-t0)*1e3:.1f} ms cold, {(t2-t1)*1e3:.1f} ms warm; first decode step {first*1e3:.1f} ms; "
        f"decode median {times[len(times)//2]*1e3:.2f} ms, min {times[0]*1e3:.2f} ms over {n} steps")
