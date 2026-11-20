#!/bin/bash
# Link molina, the linear algebra package minigrad.linalg uses, into
# build/deps, so code compiles with `-I build/deps`. molina lives in the
# MojoDEC repository; MOLINA_PATH points at its package directory
# (default: ../MojoDEC/molina, a sibling checkout). It is compiled from
# source with this workspace's Mojo: minigrad imports only molina.dense
# and molina.real_array, which need no MAX (the package as a whole does).
# Run from the repository root: pixi run molina
set -e
MOLINA_PATH=${MOLINA_PATH:-../MojoDEC/molina}
if [ ! -f "$MOLINA_PATH/dense.mojo" ]; then
    echo "error: molina not found at $MOLINA_PATH (set MOLINA_PATH)" >&2
    exit 1
fi
mkdir -p build/deps
ln -sfn "$(cd "$MOLINA_PATH" && pwd)" build/deps/molina
