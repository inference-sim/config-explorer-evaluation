# Step 2: Binary Search for Max QPS

**Status**: ✅ Completed

## What's Implemented

Binary search algorithm to find the maximum QPS where **multiple SLO constraints** are met, with configurable QPS granularity (default: 0.01 QPS precision).

## Files

- `qps_search.py` - Binary search implementation for finding max QPS with multi-SLO support

## How It Works

1. **Discrete Binary Search**: Uses numpy array of discrete QPS values (default: 0.01 step size)
2. **Multi-SLO Checking**: Tests all SLO constraints defined in config file
3. **Index-based Search**: Binary search on indices for precise convergence
4. **Result**: Returns max QPS that meets **all** SLO constraints

## Key Features

✅ **Multiple SLO constraints**: Support any combination of latency metrics (e.g., P95 E2E + P90 TTFT)
✅ **Flexible granularity**: Configurable QPS step size (default 0.01)
✅ **Config-based SLOs**: SLO thresholds defined in config file, not CLI args
✅ **Detailed logging**: Shows status of each SLO metric during search
✅ **Robust**: Handles edge cases (all SLOs met, none met)

## Usage

### Command Line

```bash
python qps_search.py --config <config.json> [OPTIONS]
```

**Required:**
- `--config` or `-c`: Path to configuration JSON file

**Optional:**
- `--trace` or `-t`: Path to trace file (CSV with prompt_tokens, output_tokens)
- `--qps-min`: Minimum QPS to search (default: 0.1)
- `--qps-max`: Maximum QPS to search (default: 100.0)
- `--qps-granularity`: QPS step size (default: 0.01)

**Examples:**

```bash
# Basic search (SLOs defined in config file)
python qps_search.py --config test_config.json

# With trace file
python qps_search.py -c test_config.json --trace traces/chat.csv

# Custom search parameters
python qps_search.py -c test_config.json --qps-max 50.0

# Quick test with coarser granularity
python qps_search.py -c test_config.json --qps-granularity 0.1
```

### Python API

```python
from qps_search import find_max_qps
import json

# Load config with SLO definitions
with open('test_config.json', 'r') as f:
    config = json.load(f)

# Config must include "slos" field (num_requests and vllm_version are optional):
# {
#   "vllm_version": "vllm/vllm-openai:v0.8.4",  # optional, default: v0.8.4
#   "num_requests": 500,  # optional, default: 500
#   "slos": [
#     {"metric": "e2e_p95_ms", "threshold_ms": 1000},
#     {"metric": "ttft_p90_ms", "threshold_ms": 500}
#   ],
#   ...
# }

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

**Available SLO metrics (any BLIS metric can be used):**
- **End-to-End Latency:** `e2e_mean_ms`, `e2e_p90_ms`, `e2e_p95_ms`, `e2e_p99_ms`, `e2e_max_ms`
- **Time to First Token:** `ttft_mean_ms`, `ttft_p90_ms`, `ttft_p95_ms`, `ttft_p99_ms`, `ttft_max_ms`
- **Inter-Token Latency:** `itl_mean_ms`, `itl_p90_ms`, `itl_p95_ms`, `itl_p99_ms`, `itl_max_ms`

You can use **any combination** of these metrics in your SLO constraints, including mean, and percentile values.

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

        # Run simulation
        metrics = run_blis(config, test_qps)

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

## Output

The search returns:
- **max_qps**: Highest QPS meeting all SLOs
- **metrics**: Comprehensive BLIS simulation metrics at max_qps

**All metrics displayed at the end include:**

- **End-to-End Latency:** Mean, P90, P95, P99, Max (ms)
- **Time to First Token (TTFT):** Mean, P90, P95, P99, Max (ms)
- **Inter-Token Latency (ITL):** Mean, P90, P95, P99, Max (ms)
- **Throughput:** Responses/sec, Tokens/sec
- **Requests:** Completed
- **Configuration:** Total KV Blocks, QPS

## Example Output

```bash
$ python qps_search.py --config test_config.json -n 100

Configuration:
  Model: codellama/CodeLlama-34b-Instruct-hf
  Hardware: H100
  TP: 1
  Batch Size: 256
  Max Scheduled Tokens: 4096
  Max Model Length: 8192
  GPU Memory Utilization: 0.9
  Num Requests: 500

Search Parameters:
  QPS Range: [0.1, 100.0]
  QPS Granularity: 0.01

SLO Constraints:
  e2e_p95_ms < 1000 ms

============================================================
Starting Binary Search for Max QPS
============================================================
SLO Constraints:
  e2e_p95_ms < 1000.0 ms
Search Range: [0.1, 100.0] QPS
Granularity: 0.01 QPS
Requests per simulation: 100
============================================================

Iteration 1: Testing QPS = 50.00 (index 4990/9999)
  ✅ e2e_p95_ms: 856.32 ms (SLO: 1000.0 ms)
  ✅ All SLOs met, searching higher

Iteration 2: Testing QPS = 75.00 (index 7490/9999)
  ❌ e2e_p95_ms: 1247.91 ms (SLO: 1000.0 ms)
  ❌ SLO violated: e2e_p95_ms (1247.91ms > 1000ms)
  Searching lower

Iteration 3: Testing QPS = 62.50 (index 6240/9999)
  ✅ e2e_p95_ms: 943.27 ms (SLO: 1000.0 ms)
  ✅ All SLOs met, searching higher

...

============================================================
Binary Search Complete
============================================================
Max QPS meeting all SLOs: 67.34 QPS

SLO Metrics at Max QPS:
  ✅ e2e_p95_ms: 998.12 ms (SLO: 1000.0 ms)

Throughput: 8.92 QPS
Total KV Blocks: 140230
============================================================

✅ Search completed successfully!

Results:
  Max QPS: 67.34

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
    Responses/sec: 8.92
    Tokens/sec: 1247.83

  Requests:
    Completed: 500

  Configuration:
    Total KV Blocks: 140230
    QPS: 67.34
```

## Parameters

**Config file parameters:**
- `num_requests`: Number of requests per simulation (default: 500 if not specified, more = slower but more accurate)
- `slos`: List of SLO constraints with `metric` and `threshold_ms` fields

**CLI parameters:**
- `--config`: Path to configuration JSON file (required)
- `--trace`: Path to trace file (optional, CSV format)
- `--qps-min`: Minimum QPS to search (default: 0.1)
- `--qps-max`: Maximum QPS to search (default: 100.0)
- `--qps-granularity`: QPS step size (default: 0.01, smaller = more precise but slower)

## Dependencies

Requires Step 1 components:
- `blis_runner.py`: For running BLIS simulations
- `capacity_planner.py`: For KV block calculations
- BLIS binary: `inference-sim/simulation_worker`

## Next Steps

With Step 2 complete, we can proceed to:

**Step 3** (1 day): Parallel config search
- Load config space from YAML
- Parallel execution with multiprocessing
- Find best configuration across multiple options

## Advanced Usage

### Multiple SLO Constraints

Define multiple SLOs to ensure both latency and responsiveness. You can use **any BLIS metric**, including mean, median, percentiles, or max:

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

**Example with mean values:**
```json
{
  "slos": [
    {"metric": "e2e_mean_ms", "threshold_ms": 800},
    {"metric": "ttft_mean_ms", "threshold_ms": 100},
    {"metric": "e2e_p99_ms", "threshold_ms": 1500}
  ]
}
```

This ensures mean latencies stay low while also capping tail latencies.

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

## Timeline

- Step 1: ✅ BLIS runner + capacity planner (1 day)
- Step 2: ✅ Binary search for max QPS with multi-SLO support (1 day)
- Step 3: ⏳ Parallel config search (1 day)
- Step 4: ⏳ Vidur wrapper (0.5 day)
- Step 5: ⏳ Polish + docs (0.5 day)
