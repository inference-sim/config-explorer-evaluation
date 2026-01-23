# Step 1: BLIS Runner with Capacity Planner

This implements Step 1 of the development roadmap: BLIS single run with capacity planner integration.

## What's Implemented

1. **Capacity Planner** (`capacity_planner.py`): Calculates total KV cache blocks from model, hardware, and memory configuration
2. **BLIS Runner** (`blis_runner.py`): Runs BLIS simulations with automatic KV block calculation
3. **Test Script** (`test_blis.py`): Simple testing interface

## Files

- `capacity_planner.py` - Calculates KV blocks based on max_model_len and gpu_memory_utilization
- `blis_runner.py` - Wrapper for running BLIS simulations
- `test_blis.py` - Test script for running single simulations
- `test_config.json` - Example configuration file

## Prerequisites

1. **Build BLIS** (if not already built):
   ```bash
   cd inference-sim
   git checkout openevolve
   go build -o simulation_worker main.go
   cd ..
   ```

2. **Verify defaults.yaml exists**:
   ```bash
   ls inference-sim/defaults.yaml
   ```
   If missing, it should have been created automatically during the build from coefficients.yaml and workloads.yaml.

## Usage

### Test Capacity Planner

Calculate KV blocks for a given configuration:

```bash
python capacity_planner.py
```

This will show the calculation for the default example (Llama-3.1-8B on H100).

### Run Single BLIS Simulation

Test a configuration at a specific QPS:

```bash
python test_blis.py <config.json> [qps] [num_requests]
```

Examples:

```bash
# Run with default QPS (5.0) and 100 requests
python test_blis.py test_config.json

# Run at 10 QPS with 200 requests
python test_blis.py test_config.json 10.0 200

# Quick test at 2 QPS with 50 requests
python test_blis.py test_config.json 2.0 50
```

### Configuration Format

Create a JSON config file with these parameters:

```json
{
  "model": "meta-llama/llama-3.1-8b-instruct",
  "hardware": "H100",
  "tp": 1,
  "batch_size": 64,
  "max_scheduled_tokens": 2048,
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.90,
  "block_size": 16,
  "prompt_tokens": 800,
  "prompt_tokens_stdev": 300,
  "output_tokens": 400,
  "output_tokens_stdev": 200
}
```

**Key parameters**:
- `max_model_len`: Maximum sequence length (swept in config search)
- `gpu_memory_utilization`: GPU memory target (swept in config search)
- `total_kv_blocks`: **Automatically calculated** by capacity planner
- `batch_size`: Max concurrent requests
- `max_scheduled_tokens`: Max tokens per iteration

### Using Python API

```python
from blis_runner import run_blis
import json

# Load config
with open('test_config.json', 'r') as f:
    config = json.load(f)

# Run simulation at 5 QPS
metrics = run_blis(config, qps=5.0, num_requests=100)

if metrics:
    print(f"P90 Latency: {metrics['e2e_p90_ms']:.2f} ms")
    print(f"P90 TTFT: {metrics['ttft_p90_ms']:.2f} ms")
```

## Output Metrics

BLIS returns the following metrics:

- `e2e_p90_ms`: 90th percentile end-to-end latency (ms)
- `e2e_p95_ms`: 95th percentile end-to-end latency (ms)
- `e2e_p99_ms`: 99th percentile end-to-end latency (ms)
- `ttft_p90_ms`: 90th percentile time to first token (ms)
- `ttft_p95_ms`: 95th percentile time to first token (ms)
- `ttft_p99_ms`: 99th percentile time to first token (ms)
- `itl_p90_ms`: 90th percentile inter-token latency (ms)
- `responses_per_sec`: Throughput (QPS)
- `tokens_per_sec`: Token throughput
- `completed_requests`: Number of completed requests
- `total_kv_blocks`: Calculated KV cache blocks (from capacity planner)
- `qps`: Requested arrival rate

## Example Output

```bash
$ python test_blis.py test_config.json 2.0 50

Capacity Planning Results:
  Model: meta-llama/llama-3.1-8b-instruct
  Hardware: H100 (80.0GB)
  Tensor Parallelism: 1
  GPU Memory Utilization: 90.00%
  Model Memory: 16384.00MB
  Available for KV Cache: 57344.00MB
  KV Cache per Block (16 tokens): 2.0000MB
  Total KV Blocks: 28,672

Running BLIS simulation:
  QPS: 2.0
  Batch Size: 64
  Max Scheduled Tokens: 2048
  Max Model Length: 4096
  GPU Memory Utilization: 0.9
  Total KV Blocks: 28672

============================================================
✅ Simulation completed successfully!
============================================================

Key Metrics:
  P90 End-to-End Latency: 6065.86 ms
  P90 TTFT: 38.36 ms
```

## Next Steps

Step 1 is complete. The next steps in the roadmap are:

- **Step 2**: Binary search for max QPS (find QPS where p90 latency meets SLO)
- **Step 3**: Parallel config search (evaluate multiple configs)
- **Step 4**: Vidur wrapper
- **Step 5**: Polish and documentation

## Troubleshooting

### "BLIS binary not found"

Build the binary:
```bash
cd inference-sim
go build -o simulation_worker main.go
cd ..
```

### "defaults.yaml not found"

The defaults.yaml file should exist in `inference-sim/` directory. If it's missing, it was created during the openevolve branch setup.

### Simulation times out

Increase the timeout in `blis_runner.py` or reduce the number of requests for testing.
