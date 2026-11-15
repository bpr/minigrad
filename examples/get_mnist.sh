#!/bin/bash
# Download MNIST into data/mnist/ (about 55 MB unpacked), from the mirror
# tinygrad uses. Run from the repository root: bash examples/get_mnist.sh
set -e
dir=data/mnist
mkdir -p "$dir"
base=https://storage.googleapis.com/cvdf-datasets/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    if [ ! -f "$dir/$f" ]; then
        echo "fetching $f"
        curl -fsSL "$base/$f.gz" | gunzip > "$dir/$f.part"
        mv "$dir/$f.part" "$dir/$f"
    fi
done
ls -l "$dir"
