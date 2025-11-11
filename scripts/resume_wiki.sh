#!/bin/bash

# Quick Resume Wiki Training
# Simple wrapper for continuing Wikipedia pretraining

# Use all defaults - perfect for quick continuation
./scripts/continue_wiki.sh \
    outputs/wiki_pretrained/model_checkpoint.bin \
    data/pretraining/wiki/wiki_corpus.txt \
    outputs/tokenizer_wiki.vocab \
    outputs/wiki_pretrained \
    0.00005 \
    5 \
    128
