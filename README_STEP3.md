# Step 3: Parallel Config Search

## Overview

Parallel config search evaluates multiple vLLM configurations concurrently to find the optimal configuration that maximizes QPS while meeting SLO constraints.

## Features

- **Parallel Evaluation**: Uses Python multiprocessing to evaluate N configs simultaneously
- **Automatic KV Blocks Calculation**: Calculates `total_kv_blocks` for each config
- **Binary Search Integration**: Calls `find_max_qps()` for each config
- **YAML Config Space**: Define config search space in YAML format
- **Multi-SLO Support**: Supports multiple simultaneous SLO constraints
- **Results Ranking**: Automatically ranks configs by max QPS
- **JSON Output**: Optionally save detailed results to JSON file

## Usage

### Basic Usage

```bash
python parallel_search.py --configs examples/configs_explicit.yaml
```

### With Trace File

```bash
python parallel_search.py -c examples/configs_explicit.yaml --trace traces/chat.csv
```

### Specify Number of Workers

```bash
# Use 4 parallel workers
python parallel_search.py -c examples/configs_explicit.yaml --num-workers 4

# Default: Uses all CPU cores
python parallel_search.py -c examples/configs_explicit.yaml
```

### Save Results to JSON

```bash
python parallel_search.py -c examples/configs_explicit.yaml --output results.json
```

### Custom Search Parameters

```bash
python parallel_search.py \
  -c examples/configs_explicit.yaml \
  --qps-min 1.0 \
  --qps-max 50.0 \
  --qps-granularity 0.1
```

## YAML Config Format

The tool supports **two formats** for defining config spaces:

### Format 1: Grid Search (Recommended)

Specify lists of values for each parameter. All combinations are automatically generated via Cartesian product.

**Example**: `examples/configs_grid_search.yaml`

```yaml
# Base configuration
model: meta-llama/llama-3.1-8b-instruct
hardware: H100
num_requests: 500

# SLO constraints (applied to all configs)
slos:
  - metric: e2e_p95_ms
    threshold_ms: 1000

# Grid search parameters - provide lists of values
tp: [1, 2]
batch_size: [128, 256, 512]
max_scheduled_tokens: [2048, 4096]
max_model_len: [4096, 8192]
gpu_memory_utilization: [0.85, 0.90, 0.95]
block_size: [16]

# Generates: 2 × 3 × 2 × 2 × 3 × 1 = 72 configs automatically
```

**Benefits**:
- ✅ Concise - define parameter ranges once
- ✅ Systematic - ensures all combinations tested
- ✅ Easy to modify - add/remove values in one place
- ✅ No duplicates or missing configs

### Format 2: Explicit Configs (Backward Compatible)

Manually specify each config. Use this when you want specific combinations only.

**Example**: `examples/configs_explicit.yaml`

```yaml
# Base configuration
model: meta-llama/llama-3.1-8b-instruct
hardware: H100
tp: 1
num_requests: 500

# SLO constraints (applied to all configs)
slos:
  - metric: e2e_p95_ms
    threshold_ms: 1000

# Explicit config list
configs:
  - batch_size: 128
    max_scheduled_tokens: 4096
    max_model_len: 4096
    gpu_memory_utilization: 0.90
    block_size: 16

  - batch_size: 256
    max_scheduled_tokens: 8192
    max_model_len: 8192
    gpu_memory_utilization: 0.90
    block_size: 16
```

**Use when**:
- You want specific combinations only (not all permutations)
- You need to exclude certain invalid combinations
- You're cherry-picking based on prior knowledge

### Required Fields

**Base Config:**
- `model`: HuggingFace model name
- `hardware`: GPU type (default: H100)
- `slos`: List of SLO constraints

**Each Config (Explicit Format):**
- `batch_size`: Max requests in batch
- `max_scheduled_tokens`: Max tokens per iteration
- `max_model_len`: Max sequence length
- `gpu_memory_utilization`: GPU memory target (0.0-1.0)
- `block_size`: KV cache block size (default: 16)

**Grid Search Parameters (Grid Format):**
- `tp`: Tensor parallelism size (can be list for sweep, default: [1])
- `batch_size`: Max requests in batch (list of values)
- `max_scheduled_tokens`: Max tokens per iteration (list of values)
- `max_model_len`: Max sequence length (list of values)
- `gpu_memory_utilization`: GPU memory target (list of values)
- `block_size`: KV cache block size (list of values, default: [16])

### Optional Fields

- `num_requests`: Requests per simulation (default: 500)
- `vllm_version`: vLLM Docker image version
- Workload parameters: `prefix_tokens`, `prompt_tokens`, `output_tokens`, etc.

## Output

### Console Output

The tool displays:
1. All configs ranked by max QPS
2. SLO compliance for each config
3. Best config highlighted with 🏆

Example output:
```
================================================================================
PARALLEL CONFIG SEARCH RESULTS
================================================================================

Evaluated 4 configs:
  Successful: 4
  Failed: 0

--------------------------------------------------------------------------------
All Configs (sorted by Max QPS):
--------------------------------------------------------------------------------

🏆 Config 2:
   Max QPS: 67.34
   batch_size: 256
   max_scheduled_tokens: 4096
   max_model_len: 8192
   gpu_memory_utilization: 0.90
   block_size: 16
   total_kv_blocks: 140,230
   SLO Metrics:
     ✅ e2e_p95_ms: 998.12 ms (SLO: 1000 ms)

 2. Config 3:
   Max QPS: 65.20
   batch_size: 256
   max_scheduled_tokens: 4096
   max_model_len: 8192
   gpu_memory_utilization: 0.95
   ...

================================================================================
BEST CONFIG
================================================================================

Config 2 achieves highest QPS: 67.34

Optimal Configuration:
  batch_size: 256
  max_scheduled_tokens: 4096
  max_model_len: 8192
  gpu_memory_utilization: 0.90
  block_size: 16
  total_kv_blocks: 140,230

SLO Compliance:
  ✅ e2e_p95_ms: 998.12 ms (SLO: 1000 ms)
```

### JSON Output (Optional)

When using `--output results.json`, saves detailed results:

```json
[
  {
    "config_id": 1,
    "config": {
      "model": "...",
      "batch_size": 256,
      "max_model_len": 8192,
      ...
    },
    "max_qps": 67.34,
    "metrics": {
      "e2e_p95_ms": 998.12,
      "ttft_p90_ms": 145.67,
      "total_kv_blocks": 140230,
      ...
    },
    "success": true
  },
  ...
]
```

## How It Works

1. **Load Config Space**: Reads YAML file with base config and list of configs
2. **Merge Configs**: Combines base parameters with each specific config
3. **Parallel Evaluation**:
   - Uses `multiprocessing.Pool` with N workers
   - Each worker:
     - Calculates `total_kv_blocks` using capacity planner
     - Runs binary search via `find_max_qps()`
     - Returns max QPS and metrics
4. **Rank Results**: Sorts configs by max QPS (descending)
5. **Display Best**: Shows all results and highlights optimal config

## Performance

- **Parallelization**: Evaluates N configs simultaneously (default: CPU count workers)
- **Binary Search**: Each config requires ~13-15 simulations (with 0.01 granularity)
- **Total Time**: Roughly `(time_per_simulation * 13) / num_workers`

Example:
- 4 configs, 4 workers, 15 seconds per simulation
- Total time: ~(15 * 13) / 4 ≈ **3-4 minutes**

## Tips

1. **Start Small**: Use `num_requests: 100` for faster testing
2. **Coarser Granularity**: Use `--qps-granularity 0.1` for faster search
3. **Worker Count**: More workers = faster, but ensure enough system resources
4. **Config Selection**: Focus on promising config ranges to reduce search space

## Integration with Steps 1 & 2

- **Step 1 (BLIS Runner)**: Used by `find_max_qps()` to run simulations
- **Step 2 (Binary Search)**: Called for each config to find max QPS
- **Capacity Planner**: Automatically calculates `total_kv_blocks` for each config

## Examples

See `examples/configs_explicit.yaml` for a complete example config space.

## Command Reference

```
python parallel_search.py --help

Required:
  -c, --configs PATH     Path to YAML file with config space

Optional:
  -t, --trace PATH       Path to trace file (CSV)
  -n, --num-workers N    Number of parallel workers (default: CPU count)
  -o, --output PATH      Save results to JSON file
  --qps-min FLOAT        Minimum QPS to search (default: 0.1)
  --qps-max FLOAT        Maximum QPS to search (default: 100.0)
  --qps-granularity F    QPS step size (default: 0.01)
```

## Troubleshooting

**All configs fail:**
- Check SLO thresholds are realistic
- Verify model and hardware are correct
- Try with fewer `num_requests` for faster debugging

**Out of memory:**
- Reduce `num_workers`
- Use smaller `num_requests` per simulation

**Slow performance:**
- Increase `qps_granularity` (e.g., 0.1 instead of 0.01)
- Reduce `num_requests` per simulation
- Use fewer configs in search space
