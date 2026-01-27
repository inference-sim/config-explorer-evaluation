# Step 2: Binary Search for Max QPS

**Status**: ✅ Completed

## Overview

Binary search algorithm to find the maximum QPS where **multiple SLO constraints** are met, with configurable QPS granularity (default: 0.01 QPS precision).

**Supported Simulators**: Both **BLIS** (fast and accurate, recommended) and **Vidur** (ML-based) with command-line flag.

## Files

- `qps_search.py` - Binary search implementation supporting both BLIS and Vidur simulators

## Simulator Selection

Switch between simulators using the required `--simulator` argument:

```bash
# BLIS (fast and accurate, recommended)
python qps_search.py --config test_config.json --simulator blis

# Vidur (ML-based)
python qps_search.py --config test_config.json --simulator vidur
```

**Same config files work for both simulators** - no changes needed.

## Key Features

✅ **Multiple SLO constraints**: Support any combination of latency metrics (e.g., P95 E2E + P90 TTFT)
✅ **Flexible granularity**: Configurable QPS step size (default 0.01)
✅ **Config-based SLOs**: SLO thresholds defined in config file, not CLI args
✅ **Dual simulator support**: Works with both BLIS (fast and accurate, recommended) and Vidur (ML-based)
✅ **Runtime tracking**: Reports total search time for performance analysis
✅ **YAML support**: Works with both JSON and YAML config files
✅ **Simulation failure handling**: Gracefully handles invalid metrics and failed simulations
✅ **Detailed logging**: Shows status of each SLO metric during search
✅ **Robust**: Handles edge cases (all SLOs met, none met)

## Usage

### Using BLIS (Recommended)

```bash
# BLIS simulator (fast and accurate, recommended)
python qps_search.py --config test_config.json --simulator blis

# With YAML config
python qps_search.py -c examples/configs_grid_search.yaml --simulator blis

# With trace file (Vidur only - BLIS uses distribution mode)
python qps_search.py -c test_config.json --simulator blis --trace traces/chat.csv

# Custom search parameters
python qps_search.py -c test_config.json --simulator blis --qps-max 50.0 --qps-granularity 0.1
```

### Using Vidur

```bash
# Vidur simulator (ML-based)
python qps_search.py --config test_config.json --simulator vidur

# With trace file
python qps_search.py -c test_config.json --simulator vidur --trace traces/chat.csv
```

### Command-Line Options

**Required:**
- `--config` or `-c`: Path to configuration file (JSON or YAML)
- `--simulator` or `-s`: Simulator to use (`blis` or `vidur`)

**Optional:**
- `--trace` or `-t`: Path to trace file (CSV with prompt_tokens, output_tokens; only used for Vidur)
- `--qps-min`: Minimum QPS to search (default: 0.1)
- `--qps-max`: Maximum QPS to search (default: 100.0)
- `--qps-granularity`: QPS step size (default: 0.01)

## Configuration File Format

The config JSON file must include `"num_requests"` and `"slos"` fields:

```json
{
  "model": "meta-llama/llama-3.1-8b-instruct",
  "hardware": "H100",
  "tp": 1,
  "batch_size": 256,
  "max_scheduled_tokens": 4096,
  "max_model_len": 8192,
  "gpu_memory_utilization": 0.90,
  "block_size": 16,
  "vllm_version": "vllm/vllm-openai:v0.8.4",
  "num_requests": 500,
  "slos": [
    {"metric": "e2e_p95_ms", "threshold_ms": 1000},
    {"metric": "ttft_p90_ms", "threshold_ms": 500}
  ]
}
```

**Key fields:**
- `vllm_version`: vLLM Docker image version (optional, default: `"vllm/vllm-openai:v0.8.4"`)
- `num_requests`: Number of requests per simulation (optional, default: 500)
- `slos`: List of SLO constraints with metric and threshold_ms

**Available SLO metrics:**
- **End-to-End Latency:** `e2e_mean_ms`, `e2e_p90_ms`, `e2e_p95_ms`, `e2e_p99_ms`, `e2e_max_ms`
- **Time to First Token:** `ttft_mean_ms`, `ttft_p90_ms`, `ttft_p95_ms`, `ttft_p99_ms`, `ttft_max_ms`
- **Inter-Token Latency:** `itl_mean_ms`, `itl_p90_ms`, `itl_p95_ms`, `itl_p99_ms`, `itl_max_ms`

You can use **any combination** of these metrics in your SLO constraints.

## Algorithm

```python
def find_max_qps(config, slos, qps_min=0.1, qps_max=100.0, qps_granularity=0.01):
    # Create discrete QPS values array
    qps_values = np.arange(qps_min, qps_max, qps_granularity)

    # Binary search on indices
    low_idx = 0
    high_idx = len(qps_values) - 1
    max_qps = -1

    while low_idx <= high_idx:
        mid_idx = (low_idx + high_idx) // 2
        test_qps = qps_values[mid_idx]

        # Run simulation (BLIS or Vidur)
        metrics = run_simulator(config, test_qps)

        # Check if ANY SLO violated
        slo_violated = any(
            metrics[slo['metric']] > slo['threshold_ms']
            for slo in slos
        )

        if slo_violated:
            high_idx = mid_idx - 1  # Search lower
        else:
            max_qps = test_qps       # All SLOs met
            low_idx = mid_idx + 1    # Try higher

    return max_qps, best_metrics
```

## Python API

```python
from qps_search import find_max_qps
import json

# Load config with SLO definitions
with open('test_config.json', 'r') as f:
    config = json.load(f)

# Find max QPS meeting all SLOs
max_qps, metrics = find_max_qps(
    config=config,
    slos=config['slos'],
    qps_granularity=0.01
)

print(f"Max QPS: {max_qps:.2f}")
for slo in config['slos']:
    metric_value = metrics.get(slo['metric'], 0)
    print(f"{slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")
```

## Output

### BLIS Output Example

```bash
$ python qps_search.py --config test_config.json --simulator blis

============================================================
QPS Search - BLIS Simulator
============================================================

Configuration:
  Model: codellama/CodeLlama-34b-Instruct-hf
  Hardware: H100
  TP: 1
  Batch Size: 256
  Max Scheduled Tokens: 4096
  Max Model Length: 8192
  GPU Memory Utilization: 0.9
  Hardware Config: hardware_config.json
  Model Config Base: model_configs
  Roofline Model: ENABLED
  Num Requests: 500

Search Parameters:
  QPS Range: [0.1, 100.0]
  QPS Granularity: 0.01

SLO Constraints:
  e2e_p95_ms < 1000 ms

============================================================
Starting Binary Search for Max QPS
============================================================

Iteration 1: Testing QPS = 50.00 (index 4990/9999)
  ✅ e2e_p95_ms: 856.32 ms (SLO: 1000.0 ms)
  ✅ All SLOs met, searching higher

Iteration 2: Testing QPS = 75.00 (index 7490/9999)
  ❌ e2e_p95_ms: 1247.91 ms (SLO: 1000.0 ms)
  ❌ SLO violated: e2e_p95_ms (1247.91ms > 1000ms)
  Searching lower

...

============================================================
Binary Search Complete
============================================================
Max QPS meeting all SLOs: 67.34 QPS

SLO Metrics at Max QPS:
  ✅ e2e_p95_ms: 998.12 ms (SLO: 1000.0 ms)

Throughput: 67.34 QPS
Total KV Blocks: 140,230
============================================================

✅ Search completed successfully!

Results:
  Max QPS: 67.34
  Total Runtime: 182.47 seconds

SLO Metrics at Max QPS:
  ✅ e2e_p95_ms: 998.12 ms (SLO: 1000.0 ms)

All BLIS Metrics:

  End-to-End Latency:
    Mean: 845.23 ms
    P90: 892.45 ms
    P95: 998.12 ms
    P99: 1156.78 ms

  Time to First Token (TTFT):
    Mean: 124.56 ms
    P90: 143.21 ms
    P95: 156.89 ms
    P99: 178.45 ms

  Inter-Token Latency (ITL):
    Mean: 15.34 ms
    P90: 18.92 ms
    P95: 21.45 ms
    P99: 26.78 ms

  Throughput:
    Responses/sec: 67.34
    Tokens/sec: 1247.83

  Requests:
    Completed: 500
    Failed: 0
    Total: 500

  Configuration:
    Total KV Blocks: 140,230
    QPS: 67.34
```

### Vidur Output Example

```bash
$ python qps_search.py --config test_config.json

============================================================
QPS Search - Vidur Simulator
============================================================

Configuration:
  Model: codellama/CodeLlama-34b-Instruct-hf
  ...

All Vidur Metrics:

  End-to-End Latency:
    Mean: 878.45 ms
    P90: 923.12 ms
    P95: 1012.34 ms
    P99: 1189.56 ms
  ...

  Configuration:
    QPS: 64.12
```

**Note**: Vidur output excludes `Total KV Blocks` (calculated internally).

## Simulator Comparison

| Aspect | BLIS | Vidur |
|--------|------|-------|
| **Speed** | Fast (~5-10s/config) | Slower (~30-60s/config) |
| **Accuracy** | High (more accurate) | Lower (less accurate) |
| **Latency Model** | Linear coefficients | Random Forest ML |
| **Setup** | Simple (pre-trained) | Complex (GPU profiling) |
| **Recommendation** | Production use | Research/ML experiments |

**Use BLIS for production capacity planning** - it's faster and more accurate.

## Advanced Usage

### Multiple SLO Constraints

Define multiple SLOs to ensure both latency and responsiveness:

```json
{
  "model": "meta-llama/llama-3.1-8b-instruct",
  "num_requests": 500,
  "slos": [
    {"metric": "e2e_p95_ms", "threshold_ms": 1000},
    {"metric": "ttft_mean_ms", "threshold_ms": 150},
    {"metric": "itl_p99_ms", "threshold_ms": 50}
  ],
  ...
}
```

The search will find the max QPS where **all three** SLOs are met.

### Adjusting Number of Requests

Change `num_requests` in config file for different accuracy/speed trade-offs:

```json
{
  "num_requests": 100,   // Faster, less accurate
  "num_requests": 500,   // Balanced (recommended)
  "num_requests": 1000,  // Slower, more accurate
  ...
}
```

### Adjusting Search Precision

For faster searches with coarser precision:
```bash
python qps_search.py -c config.json --qps-granularity 0.1
```

For maximum precision (slower):
```bash
python qps_search.py -c config.json --qps-granularity 0.001
```

### Custom Search Range

If you know the approximate QPS range:
```bash
python qps_search.py -c config.json --qps-min 10 --qps-max 50
```

## Dependencies

Requires Step 1 components:
- `blis_runner.py`: For running BLIS simulations
- `capacity_planner.py`: For KV block calculations
- BLIS binary: `inference-sim/simulation_worker`

For Vidur:
- `vidur_runner.py`: For running Vidur simulations (Step 4)
- Vidur environment: See [README_STEP4.md](README_STEP4.md)

## Integration with Parallel Search

`qps_search.py` is used by `parallel_search.py` for binary search of each config:

```python
# parallel_search.py
# Both simulators supported via --simulator flag
from qps_search import find_max_qps, display_metrics_category
from vidur_runner import run_vidur
from blis_runner import run_blis
```

Both tools use the same `--simulator` flag for consistency.

## Next Steps

- **Step 3**: ✅ Parallel config search (completed)
- **Step 4**: ✅ Vidur wrapper (completed)
- **Step 5**: ⏳ Polish + docs

## Related Documentation

- [README_STEP1.md](README_STEP1.md) - BLIS runner and capacity planner
- [README_STEP3.md](README_STEP3.md) - Parallel config search
- [README_STEP4.md](README_STEP4.md) - Vidur integration guide
- [STEP2_SUMMARY.md](STEP2_SUMMARY.md) - Implementation details

---

**Summary**: `qps_search.py` finds max QPS meeting multiple SLO constraints using binary search. Supports both BLIS (fast) and Vidur (accurate) with simple flag switch. Same config files work for both simulators.
