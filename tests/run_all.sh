#!/bin/bash
# Run every minigrad test file. Run from the repository root: pixi run test
# (or `pixi run test tests/test_view.mojo` for one file).
# MOJO_FLAGS is passed to `mojo run` (e.g. "-D ASSERT=none").
files=("$@")
if [ ${#files[@]} -eq 0 ]; then
    files=(tests/test_*.mojo)
fi
failed=()
for f in "${files[@]}"; do
    echo "=== $f"
    mojo run $MOJO_FLAGS -I . -I tests "$f" || failed+=("$f")
done
if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "All test files passed."
