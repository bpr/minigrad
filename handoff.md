# Handoff: benchmarks to take on a quiet machine

Give this file to Claude in a session on a quiet machine. Read
CLAUDE.md, AGENTS.md and `benchmarks/README.md` first. Wait until
`/proc/loadavg` is below 1 before timing anything, and report medians.

## What needs measuring, and why

The last three changes were timed only under load, or not at all:

1. **`336d3bc`: the cast builder converts before movements**
   (`UOp.cast` in `minigrad/uop.mojo`). It fixed the Python MNIST step
   (about 240 ms down to 116 ms, both measured at a load average near 4).
   The change touches every graph that casts a broadcast, so check that
   the Mojo benchmarks did not regress: MLP step, convnet step, GPT-2
   decode (float32 and `--dtype bfloat16`/`int8`, which depend on the
   cast staying a cast of a buffer), MNIST end to end.
2. **Python MNIST, end to end**: `examples/beautiful_mnist.py` has never
   finished on a quiet machine. A run on 2026-11-17 was stopped when the
   machine ran low on memory, during the first test-accuracy
   evaluation: `get_test_acc` runs all 10,000 test images in one batch
   (as tinygrad's file does), which takes several GB. Watch memory
   (`free -g`); if it fails again, say so rather than changing the
   example silently.
3. **The Python MLP table** (`benchmarks/README.md`, "From Python") was
   taken quietly before `336d3bc`; rerun it to confirm.

Baseline: `96ca3e1` (the commit before the cast change). Quiet
numbers taken at `96ca3e1`, to compare against:

| Benchmark | Quiet result before the cast change |
|---|---|
| `mlp_step 8 32 3000` | 75 µs median |
| `mlp_step 64 128 3000` | 116 µs |
| `mlp_step 512 512 300` | 720 µs |
| `convnet_step 64 20` | 9.8 ms |
| `convnet_step 512 20` | about 90 ms |
| `gpt2_bench bench 100` (13-token prompt) | decode median 8.65 ms |
| `examples/mnist.mojo` total | 10.0 s |
| Python `mlp_step.py 8 32 3000` / `jit` | 94 µs / 20 µs |
| Python `mlp_step.py 64 128 3000` / `jit` | 148 to 152 µs / 81 to 109 µs |
| Python `mlp_step.py 512 512 300` / `jit` | 805 µs / 730 to 1,000 µs |
| Python `beautiful_mnist.py` step (under load) | 116 ms, vs 107 ms for `mnist.mojo` |

## How to run it

Data must be present: `data/mnist` (`bash examples/get_mnist.sh`) and
`data/gpt2` (`bash examples/get_gpt2.sh`).

```sh
cd ~/src/mojo/minigrad
OUT=$(mktemp -d)                   # binaries and results
git worktree add $OUT/base 96ca3e1 # the baseline tree
pixi install && pixi run python-build

for v in base new; do
  src=$([ $v = base ] && echo $OUT/base || echo .)
  for p in benchmarks/mlp_step benchmarks/convnet_step benchmarks/gpt2_bench examples/mnist; do
    (cd $src && pixi run mojo build -I . -o $OUT/${v}_$(basename $p) $p.mojo)
  done
done
```

Then, once the load is below 1, run each pair interleaved (base, new,
base, new) so a burst of load hits both, two rounds:

```sh
IDS="464 5440 318 2068 2491 286 257 1110 13 632 373 257 922"
PY=.pixi/envs/default/bin/python3
until awk '{exit !($1<1.0)}' /proc/loadavg; do sleep 20; done
for r in 1 2; do for v in base new; do
  for a in "8 32 3000" "64 128 3000" "512 512 300"; do timeout 300 $OUT/${v}_mlp_step $a | grep median; done
  for a in 64 512; do timeout 300 $OUT/${v}_convnet_step $a 20 | grep median; done
  for d in "" "--dtype bfloat16" "--dtype int8"; do timeout 300 $OUT/${v}_gpt2_bench $d bench 100 $IDS | grep decode; done
done; done
for a in "8 32 3000" "8 32 3000 jit" "64 128 3000" "64 128 3000 jit" "512 512 300" "512 512 300 jit"; do
  PYTHONPATH=python timeout 300 $PY benchmarks/mlp_step.py $a | grep median
done
for v in base new; do timeout 600 $OUT/${v}_mnist | grep total; done
PYTHONPATH=python timeout 900 $PY examples/beautiful_mnist.py | tail -3
```

Note the load average before and after each block (`cat /proc/loadavg`)
and throw away a round whose load rose above 1.

## Afterwards

- Update `benchmarks/README.md` if numbers moved: the MLP, convnet,
  GPT-2 and MNIST tables, and the "From Python" section. Replace its
  "load average of 4" MNIST sentence with the quiet numbers, and add the
  Python MNIST end-to-end total and accuracy if the run finished.
- If `new` is slower than `base` anywhere beyond noise, find out why
  before anything else: `MINIGRAD_DEBUG=2` prints every kernel with its
  time, so diff the two kernel lists.
- Remove the baseline worktree: `git worktree remove $OUT/base`. An
  older one may also exist (`git worktree list`, at `87cb536` under
  `/tmp/claude-1000/...`); remove it too.
- Update `commit_msg.txt` (CLAUDE.md's rule), and delete this file once
  the numbers are in.

## Status (2026-11-17, second session)

Run on an i7-12700H laptop (20 threads, no AVX-512), not the 9800X3D
the README's tables come from, so absolute numbers were not copied into
the tables. On this machine the load average rises to 5-10 during the
run from minigrad's own pool threads alone (the machine is otherwise
96% idle), so the "below 1" rule cannot hold during a run here.

- Cast change: no regression. Interleaved base/new pairs agree within
  noise for the MLP, convnet, GPT-2 float32/bfloat16, and MNIST (27.5 s
  base, 27.0 s new). int8 decode looked 20% slower in two rounds; four
  more interleaved pairs gave new 5.5/5.6/6.6/6.6 ms against base
  6.4/6.5/6.6/6.6 ms. Same kernel and plan counts.
- Python MNIST finished: 26.8 s, 97.87%, peak RSS 3.6 GB (added to the
  README as a separate-machine note).
- Python MLP here: 181/84, 711/380, 4120/4133 µs (plain/jit).
- The GPT-2 decode breakdown was wrong since decode moved to `MiniJit`
  (`c412173`): replays did not count in `stats`, so "running kernels"
  showed about 0.5 ms and 13 kernels, and replayed kernel time was
  reported as "building the graph". Replays now count; the line reads
  309 kernels and 14.7 ms of kernels (int8: 6.5 ms) here, with about
  0.15 ms of host work. The README's "Where a decode step goes" table
  predates `MiniJit`: retake it on the 9800X3D.
- Still to do on the 9800X3D: the quiet tables above, then delete this
  file.
