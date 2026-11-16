#!/bin/bash
# Download GPT-2 (124M) weights and tokenizer into data/gpt2/ (about
# 550 MB), from Hugging Face, and a text to measure perplexity on. Run
# from the repository root:
#   bash examples/get_gpt2.sh
set -e
dir=data/gpt2
mkdir -p "$dir"
base=https://huggingface.co/openai-community/gpt2/resolve/main
for f in model.safetensors vocab.json merges.txt; do
    if [ ! -f "$dir/$f" ]; then
        echo "fetching $f"
        curl -fL --progress-bar -o "$dir/$f.part" "$base/$f"
        mv "$dir/$f.part" "$dir/$f"
    fi
done
# Text for measuring perplexity (public domain; tinygrad's examples use it).
if [ ! -f "$dir/tinyshakespeare.txt" ]; then
    echo "fetching tinyshakespeare.txt"
    curl -fsSL -o "$dir/tinyshakespeare.txt" \
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi
ls -l "$dir"
