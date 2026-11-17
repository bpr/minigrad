# CLAUDE.md

> **Note:** Project architecture and coding standards are defined in
> [AGENTS.md](./AGENTS.md).

@AGENTS.md

This file holds what is specific to Claude Code: how to work in this
repository. Commands and everything about the code itself are in
AGENTS.md.

## Workflow

- Run single test files while iterating (`pixi run test tests/test_x.mojo`);
  run `pixi run test` before calling a change done.
- Build to a binary and run it under `timeout` rather than `mojo run`
  when a program might hang: a bug in graph code tends to loop, not
  crash.
- After a change to the planner or executor, rerun
  `benchmarks/mlp_step.mojo` and `examples/mnist.mojo` and compare with
  `benchmarks/README.md`; update the table if the numbers moved.
- **Keep `commit_msg.txt` current.** After every change to the tree,
  however small and even mid-task, overwrite `commit_msg.txt` (repo
  root) with a message under 20 words (never over 100) covering
  exactly what `git status` shows as uncommitted; check `git log -1`
  first, as the user commits between turns. Never end a turn with it
  stale.
- A new deviation from tinygrad goes in
  `docs/differences-from-tinygrad.md`; a design that was tried and
  dropped goes in `docs/plan.md`.

## Where to look

| Question | Where |
|---|---|
| Commands, layout, conventions, Mojo traps | AGENTS.md |
| How a computation flows, and why it is built this way | `docs/architecture.md` |
| What is next, and what was rejected | `docs/plan.md` |
| Where minigrad differs from tinygrad | `docs/differences-from-tinygrad.md` |
| Benchmark numbers and how to reproduce them | `benchmarks/README.md` |
| The goals and the bar for success | `~/markdown/tinygrad-mojo.md` |
