# Step 4: Vidur Integration - User Guide

## Overview

The config search tool supports **two simulators** with a unified interface:

1. **BLIS** (recommended): Fast coefficient-based simulator (~10-30s per config), more accurate
2. **Vidur**: ML-based simulator (~30-60s per config), less accurate but feature-rich

**Key Feature**: Both simulators use the **same config files**. Switch between them with a command-line flag.

## Quick Start

### Using BLIS (Recommended)

```bash
# Multi-config search with BLIS (fast and accurate, recommended)
python parallel_search.py --configs examples/configs_grid_search.yaml --simulator blis

# Single-config search with BLIS
python qps_search.py --config test_config.json --simulator blis
```

### Using Vidur

```bash
# Multi-config search with Vidur (ML-based)
python parallel_search.py --configs examples/configs_grid_search.yaml --simulator vidur

# Single-config search with Vidur
python qps_search.py --config test_config.json --simulator vidur
```

**That's it!** No separate config files needed. Both tools support both simulators with the required `--simulator` flag.

## Installation

### BLIS Setup (Steps 1-3)

```bash
# Already installed if you completed Steps 1-3
pip install -r requirements.txt
cd inference-sim && git checkout openevolve && go build -o ../simulation_worker main.go
```

### Vidur Setup (Step 4)

```bash
# Install Vidur environment
cd vidur
mamba env create -p ./env -f ./environment.yml
mamba activate ./env

# Disable WandB (optional)
export WANDB_MODE=disabled

# Verify profiling data exists for your model
ls vidur/data/profiling/compute/a100/meta-llama/
```

**Pre-profiled models**:
- `meta-llama/Llama-2-7b-hf` (A100)
- `meta-llama/Llama-2-70b-hf` (A100)

For other models, see `vidur/docs/profiling.md`.

## Configuration

### Unified Config Format

Both simulators use the **same YAML format**. Simulator-specific fields are ignored by the other:

```yaml
# Works for both BLIS and Vidur
model: codellama/CodeLlama-34b-Instruct-hf
hardware: H100
vllm_version: vllm/vllm-openai:v0.8.4
num_requests: 100

# BLIS-specific (ignored by Vidur)
model_config_folder_base: model_configs
hardware_config: hardware_config.json

# Common parameters (used by both)
workload parameters...
prompt_tokens: 2871
output_tokens: 1

# Grid search (used by both)
tp: [1, 2]
batch_size: [128, 256, 512]
max_scheduled_tokens: [2048, 4096]
max_model_len: [4096, 8192]
gpu_memory_utilization: [0.90]
block_size: [16]

# SLOs (used by both)
slos:
  - metric: e2e_p95_ms
    threshold_ms: 1000
  - metric: ttft_p90_ms
    threshold_ms: 500
```

**Available Configs**:
- `examples/configs_grid_search.yaml` - Works with both simulators
- `examples/configs_explicit.yaml` - Works with both simulators

## Simulator Comparison

| Aspect | BLIS | Vidur |
|--------|------|-------|
| **Speed** | Fast (~5-10s/config) | Slower (~30-60s/config) |
| **Accuracy** | High (more accurate) | Lower (less accurate) |
| **Latency Model** | Linear coefficients | Random Forest ML |
| **Setup** | Simple (pre-trained) | Complex (GPU profiling) |
| **Recommendation** | Production use | Research/ML experiments |
| **KV Cache** | Via config_explorer | Internal |
| **Multi-GPU (TP)** | ✅ Limited | ✅ Full support |
| **Multi-Replica** | ❌ | ✅ |
| **Pipeline Parallel** | ❌ | ✅ |
| **Chrome Traces** | ❌ | ✅ |
| **Dashboard** | ❌ | ✅ Streamlit |

## Usage Examples

### Example 1: Fast Exploration with BLIS

```bash
# BLIS simulator (fast and accurate, recommended)
python parallel_search.py \
  --configs examples/configs_grid_search.yaml \
  --simulator blis \
  --num-workers 4
```

**Output**:
```
PARALLEL CONFIG SEARCH
================================================================================

Simulator: BLIS

...

🏆 Config 5:
   Max QPS: 12.45
   tp: 1
   batch_size: 256
   max_scheduled_tokens: 4096
   max_model_len: 8192
   gpu_memory_utilization: 0.90
   block_size: 16
   total_kv_blocks: 140,532

Best Config BLIS Metrics:
  End-to-End Latency:
    Mean: 654.32 ms
    P90: 876.54 ms
    P95: 987.23 ms
    P99: 1234.56 ms
  ...
```

### Example 2: ML-Based Simulation with Vidur

```bash
# Vidur simulator (ML-based)
python parallel_search.py \
  --configs examples/configs_grid_search.yaml \
  --simulator vidur \
  --num-workers 2
```

**Output**:
```
PARALLEL CONFIG SEARCH
================================================================================

Simulator: Vidur

...

🏆 Config 5:
   Max QPS: 11.89
   tp: 1
   batch_size: 256
   ...

Best Config Vidur Metrics:
  End-to-End Latency:
    Mean: 678.45 ms
    P90: 912.34 ms
    P95: 1023.12 ms
    P99: 1345.67 ms
  ...
```

**Note**: Vidur output excludes `total_kv_blocks` (calculated internally).

### Example 3: Custom Trace File

```bash
# Works with both simulators (trace file only used for Vidur)
python parallel_search.py \
  --configs examples/configs_grid_search.yaml \
  --simulator vidur \
  --trace traces/chat.csv \
  --num-workers 4
```

Trace format (CSV):
```csv
prompt_tokens,output_tokens
512,128
1024,256
256,64
```

### Example 4: Save Results to JSON

```bash
python parallel_search.py \
  --configs examples/configs_grid_search.yaml \
  --simulator blis \
  --output results.json \
  --num-workers 2
```

## Metrics Format

Both simulators output **identical metric keys**:

| Category | Metrics |
|----------|---------|
| **End-to-End** | `e2e_mean_ms`, `e2e_p90_ms`, `e2e_p95_ms`, `e2e_p99_ms` |
| **TTFT** | `ttft_mean_ms`, `ttft_p90_ms`, `ttft_p95_ms`, `ttft_p99_ms` |
| **ITL** | `itl_mean_ms`, `itl_p90_ms`, `itl_p95_ms`, `itl_p99_ms` |
| **Throughput** | `responses_per_sec`, `tokens_per_sec` |
| **Requests** | `completed_requests`, `failed_requests`, `total_requests` |

**Units**: All latency metrics in milliseconds (ms)

**Difference**: BLIS includes `total_kv_blocks`, Vidur does not.

## Troubleshooting

### All SLOs Violated

**Issue**: No configs meet SLO thresholds

**Solutions**:
1. Relax SLO thresholds:
   ```yaml
   slos:
     - metric: e2e_p95_ms
       threshold_ms: 2000  # was 1000
   ```
2. Expand search space:
   ```yaml
   batch_size: [64, 128, 256, 512, 1024]
   ```
3. Reduce workload intensity:
   ```yaml
   prompt_tokens: 1000  # was 3000
   ```

### Vidur Simulation Fails

**Issue**: `[Config X] ❌ Failed: ...`

**Solutions**:
1. Verify Vidur environment:
   ```bash
   python -m vidur.main --help
   ```
2. Check model profiling data exists:
   ```bash
   ls vidur/data/profiling/compute/a100/<model_name>/
   ```
3. Use pre-profiled models (Llama-2-7b, Llama-2-70b on A100)
4. Reduce `num_requests` for faster debugging:
   ```yaml
   num_requests: 50  # instead of 500
   ```

### Vidur Too Slow

**Issue**: Simulations take very long

**Solutions**:
1. Use fewer workers (reduce memory pressure):
   ```bash
   python parallel_search.py ... --num-workers 2
   ```
2. Reduce `num_requests`:
   ```yaml
   num_requests: 100  # Vidur: 100-200, BLIS: 500+
   ```
3. BLIS is recommended for most use cases (faster and more accurate)

### Missing Profiling Data

**Issue**: `FileNotFoundError: data/profiling/compute/...`

**Solutions**:
- Use pre-profiled models: `meta-llama/Llama-2-7b-hf` or `Llama-2-70b-hf`
- Profile your model: See `vidur/docs/profiling.md`

## Recommendations

### When to Use BLIS (Recommended)
- Most production use cases
- Fast and accurate capacity planning
- Large config search spaces (100+ configs)
- When you need reliable results quickly

### When to Use Vidur
- Research on ML-based simulation approaches
- When you need additional features (multi-replica, chrome traces, dashboard)
- Comparative studies of simulation techniques

### Recommended Approach

**Use BLIS for production capacity planning** (fast and accurate):
```bash
python parallel_search.py --configs configs_large.yaml --simulator blis --num-workers 8
# Result: Accurate max QPS in ~10 minutes
```

BLIS provides fast and more accurate results than Vidur, making it the recommended choice for production capacity planning.

## Python API

### Basic Usage

```python
from vidur_runner import find_max_qps
# or
from qps_search import find_max_qps

config = {
    "model": "meta-llama/Llama-2-7b-hf",
    "hardware": "A100",
    "tp": 1,
    "batch_size": 256,
    ...
}

slos = [
    {"metric": "e2e_p95_ms", "threshold_ms": 5000},
    {"metric": "ttft_p90_ms", "threshold_ms": 1000},
]

max_qps, metrics = find_max_qps(
    config=config,
    slos=slos,
    trace_file="traces/chat.csv",
    verbose=True
)

print(f"Max QPS: {max_qps:.2f}")
print(f"TTFT P90: {metrics['ttft_p90_ms']:.2f} ms")
```

## Implementation Details

### How Simulator Selection Works

```python
# Both tools use --simulator flag (required argument)
# qps_search.py and parallel_search.py

# Set simulator choice from command line
USE_VIDUR = (args.simulator == 'vidur')

# Import both simulators
from vidur_runner import run_vidur
from blis_runner import run_blis

# Select simulator at runtime
if USE_VIDUR:
    metrics = run_vidur(config, test_qps, ...)
else:
    metrics = run_blis(config, test_qps, ...)
```

### vLLM to Vidur Parameter Mappings

Vidur runner (`vidur_runner.py`) maps vLLM parameters to Vidur CLI arguments:

| vLLM Parameter | Vidur CLI Argument | Notes |
|----------------|-------------------|-------|
| `model` | `--replica_config_model_name` | Model identifier |
| `hardware` | `--replica_config_device` | H100→h100, A100→a100 |
| `tp` | `--replica_config_tensor_parallel_size` | Tensor parallelism |
| `batch_size` | `--random_forrest_execution_time_predictor_config_prediction_max_batch_size` | Max batch size |
| `total_kv_blocks` | `--vllm_scheduler_config_num_blocks` | Calculated via `capacity_planner` |
| `watermark_blocks` | `--vllm_scheduler_config_watermark_blocks_fraction` | Set to 0.0 (no watermark) |

**KV Block Calculation**: Vidur uses the same `calculate_total_kv_blocks()` function as BLIS:

```python
from capacity_planner import calculate_total_kv_blocks

total_kv_blocks = calculate_total_kv_blocks(
    model=config['model'],
    hardware=config['hardware'],
    tp=config['tp'],
    max_model_len=config['max_model_len'],
    gpu_memory_utilization=config['gpu_memory_utilization'],
    block_size=config.get('block_size', 16)
)
total_kv_blocks = int(total_kv_blocks * 0.8)  # Conservative factor
```

This ensures consistent memory modeling across both simulators.

### API Compatibility

Both simulators expose **identical functions**:

```python
# Same signature for both BLIS and Vidur
def find_max_qps(
    config: Dict,
    slos: List[Dict],
    trace_file: Optional[str] = None,
    qps_min: float = 0.1,
    qps_max: float = 100.0,
    qps_granularity: float = 0.1,
    verbose: bool = True
) -> Tuple[float, Dict]:
    """Binary search for max QPS meeting SLOs"""
```

## Files Structure

```
config-explorer-evaluation/
├── parallel_search.py          # Multi-config search (works with both)
├── qps_search.py              # Single-config search (works with both)
├── vidur_runner.py            # Vidur interface (Step 4, 383 lines)
├── blis_runner.py             # BLIS simulator runner (Step 1)
├── capacity_planner.py        # KV cache calculation (Step 1)
├── examples/
│   ├── configs_grid_search.yaml    # Works with both
│   └── configs_explicit.yaml       # Works with both
├── traces/
│   └── chat.csv               # Workload traces
└── README_STEP4.md            # This file
```

**Note**: Both `parallel_search.py` (multi-config) and `qps_search.py` (single-config) support both simulators with the required `--simulator` flag.

## Next Steps

See also:
- [STEP3_SUMMARY.md](STEP3_SUMMARY.md) - Parallel search implementation
- [STEP4_SUMMARY.md](STEP4_SUMMARY.md) - Comprehensive Vidur implementation and integration details

## References

- **Vidur Paper**: "Vidur: A High-Fidelity Simulator for LLM Inference Serving" (Microsoft Research)
- **vLLM**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- **BLIS**: Blackbox LLM Inference Simulator with coefficient-based latency modeling

---

**Summary**: Use the required `--simulator` flag to switch between BLIS (fast and accurate, recommended) and Vidur (ML-based). Same configs work for both. BLIS is recommended for production capacity planning.
