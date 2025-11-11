# Configuration Files for LoopOS CLI

This directory contains JSON configuration files for running different computations on transformer models.
      "input_file": "...",             // Flattened corpus text file (single file)
## Configuration Schema

Each JSON configuration file should follow this structure:

```json
{
  "model": {
    "type": "transformer",           // Model architecture type
    "d_model": 512,                  // Model dimension
    "num_heads": 8,                  // Number of attention heads
    "num_layers": 6,                 // Number of transformer layers
    "d_ff": 2048,                    // Feed-forward dimension
    "vocab_size": 50000,             // Vocabulary size
    "num_classes": 10                // (Optional) For classification tasks
  },
  "computation": {
    "mode": "pretraining",           // "pretraining" or "posttraining"
    "method": "autoregressive",      // Specific method (see below)
    "description": "..."             // Human-readable description
  },
  "training": {
    "learning_rate": 0.0001,         // Learning rate
      "batch_size": 32,                // Batch size
      "max_length": 128,               // (Optional) Truncate/segment length
      "prefetch_batches": 3,           // (Optional) Async prefetch depth for streaming loader
      "num_workers": 2,                // (Optional) File/tokenization worker threads
      "shuffle": true,                 // (Optional) Shuffle dataset each epoch
      "num_epochs": 10                 // Number of epochs
      // Add method-specific parameters after the common ones
  },
  "data": {
    "input_file": "...",             // Input data file
    "output_dir": "...",             // Output directory
    // Method-specific data paths...
  }
}
```

## Available Methods

### Pre-training Methods

1. **autoregressive**: GPT-style next-token prediction
   - Config: `autoregressive_training.json`
   - Based on: Radford et al. (2018)

2. **masked_lm**: BERT-style masked language modeling
   - Config: `masked_lm_training.json`
   - Additional parameter: `mask_probability` (default: 0.15)
   - Based on: Devlin et al. (2018)

3. **contrastive**: Contrastive learning pre-training
   - Config: `contrastive_training.json`
   - Additional parameter: `temperature` (default: 0.07)
   - Based on: Chen et al. (2020), He et al. (2020)

### Post-training Methods

1. **fine_tuning**: Fine-tuning for downstream tasks
   - Config: `fine_tuning.json`
   - Requires: `pretrained_weights` path
   - Requires: `num_classes` in model config

2. **chain_of_thought**: Chain-of-thought reasoning
   - Config: `chain_of_thought.json`
   - Requires: `pretrained_weights` path
   - Requires: `reasoning_examples` data
   - Based on: Wei et al. (2022)

3. **rlhf**: Reinforcement Learning from Human Feedback
   - Config: `rlhf_training.json`
   - Requires: `pretrained_weights` path
   - Additional parameters: `clip_epsilon`, `kl_coefficient`
   - Based on: Ouyang et al. (2022)

## Optimized Profiles

These curated configurations pair with the optimized streaming data loader. Each file sets conservative defaults that the new auto-tuning logic can further adjust to your hardware at runtime.

- `autoregressive_tiny.json` / `autoregressive_quarter.json`: Trump dataset slices sized for quick local validation.
- `autoregressive_fast.json`: Compact 24M parameter GPT variant for iterative experiments (expects `wiki_fast_corpus.txt`).
- `wiki_test.json`: Small Wikipedia shard to verify the streaming loader end-to-end (`wiki_test_corpus.txt`).
- `wiki_cache_optimized.json` & `wiki_performance_tuned.json`: Laptop-friendly presets for larger corpora that still respect CPU and memory limits (`wiki_cache_corpus.txt`, `wiki_performance_corpus.txt`).
- `wiki_gpu_optimized.json`: Higher throughput profile for discrete GPUs (`wiki_gpu_corpus.txt`).

**Tip:** The loader auto-detects laptop-class CPUs and will clamp workers/prefetch depth when necessary. The explicit `prefetch_batches` and `num_workers` values in each config act as upper bounds that the autotuner can safely dial back.

## Preparing Wiki Corpora

1. Run `scripts/flatten_wiki_corpus.sh <source_dir> <output_file> [max_files]` to concatenate shards into a single text file.
   - Example full corpus: `./scripts/flatten_wiki_corpus.sh data/pretraining/wiki/fullEnglish data/pretraining/wiki/wiki_corpus.txt`
   - Example subset (first 100 files): `./scripts/flatten_wiki_corpus.sh data/pretraining/wiki/fullEnglish data/pretraining/wiki/wiki_subset_corpus.txt 100`
2. Point `data.input_file` in the desired configuration to the generated file.
3. Re-run the script whenever you refresh the source shards (it overwrites the target file).

## Usage

Run the LoopOS CLI with a configuration file:

```bash
./build/loop_cli --config configs/autoregressive_training.json
```

Or use the shorthand:

```bash
./build/loop_cli -c configs/masked_lm_training.json
```

List all available configurations:

```bash
./build/loop_cli --list-configs
```

Validate a configuration file:

```bash
./build/loop_cli --validate configs/fine_tuning.json
```
