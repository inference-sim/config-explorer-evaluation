# vLLM Configuration Search Tool

A unified tool for finding optimal vLLM configurations that maximize QPS while meeting tail latency SLOs. Supports both **BLIS** (fast and accurate coefficient-based simulator) and **Vidur** (ML-based Random Forest simulator).

## Features

- **Binary Search for Max QPS**: Find the maximum queries-per-second that meets all SLO constraints with configurable precision (default: 0.01 QPS)
- **Multi-SLO Support**: Enforce multiple simultaneous latency constraints (e.g., P95 E2E + P90 TTFT)
- **Parallel Config Search**: Evaluate multiple configurations concurrently using multiprocessing
- **Dual Simulator Support**: Switch between BLIS and Vidur with a single `--simulator` flag
- **Automatic KV Cache Calculation**: Uses config_explorer library to compute optimal `total_kv_blocks`
- **Runtime Tracking**: Reports total runtime and per-config runtime for performance analysis
- **Flexible Config Format**: Supports both JSON and YAML config files
- **Grid Search**: Automatically generates Cartesian product from parameter lists
- **TP Search**: Tensor Parallelism as a searchable parameter

## Quick Start

### Prerequisites

```bash
# Python 3.11+ required
python --version

# Install config_explorer library
git clone https://github.com/llm-d/llm-d-benchmark.git
pip install -e ./llm-d-benchmark/config_explorer

# Build BLIS (openevolve branch for trace file support)
cd inference-sim
git checkout openevolve
go build -o ../simulation_worker main.go
cd ..

# Set BLIS_ROOT environment variable
export BLIS_ROOT=$(pwd)
```

### Single-Config Search (qps_search.py)

Find max QPS for a single configuration:

```bash
# BLIS (fast and accurate, recommended)
python qps_search.py --config test_config.json --simulator blis

# Vidur (ML-based)
python qps_search.py --config test_config.json --simulator vidur

# With YAML config
python qps_search.py --config examples/configs_grid_search.yaml --simulator blis

# Custom search parameters
python qps_search.py -c test_config.json --simulator blis \
  --qps-min 1.0 --qps-max 50.0 --qps-granularity 0.1
```

### Multi-Config Search (parallel_search.py)

Evaluate multiple configurations in parallel:

```bash
# Grid search (sweeps TP, batch size, etc.)
python parallel_search.py \
  --configs examples/configs_grid_search.yaml \
  --simulator blis \
  --num-workers 4

# With output file
python parallel_search.py \
  -c examples/configs_explicit.yaml \
  --simulator blis \
  --output results.json

# Vidur simulator
python parallel_search.py \
  -c examples/configs_grid_search.yaml \
  --simulator vidur \
  --num-workers 2
```

## Configuration Files

### JSON Format (for qps_search.py)

```json
{
  "model": "codellama/CodeLlama-34b-Instruct-hf",
  "hardware": "H100",
  "tp": 1,
  "batch_size": 256,
  "max_scheduled_tokens": 8192,
  "max_model_len": 8192,
  "gpu_memory_utilization": 0.90,
  "block_size": 16,
  "vllm_version": "vllm/vllm-openai:v0.8.4",
  "num_requests": 500,

  "model_config_folder_base": "model_configs",
  "hardware_config": "hardware_config.json",

  "slos": [
    {"metric": "e2e_p95_ms", "threshold_ms": 1000},
    {"metric": "ttft_p90_ms", "threshold_ms": 500}
  ]
}
```

### YAML Format (for parallel_search.py)

**Grid Search** (automatic Cartesian product):

```yaml
model: codellama/CodeLlama-34b-Instruct-hf
hardware: H100
vllm_version: vllm/vllm-openai:v0.8.4
num_requests: 100

# Roofline model parameters (optional)
model_config_folder_base: model_configs
hardware_config: hardware_config.json

# SLO constraints
slos:
  - metric: e2e_p95_ms
    threshold_ms: 1000

# Grid search parameters (lists = sweep)
tp: [1, 2]
batch_size: [128, 256, 512]
max_scheduled_tokens: [2048, 4096]
max_model_len: [4096, 8192]
gpu_memory_utilization: [0.90]
block_size: [16]
```

**Explicit Configs**:

```yaml
model: codellama/CodeLlama-34b-Instruct-hf
hardware: H100
num_requests: 500

slos:
  - metric: e2e_p95_ms
    threshold_ms: 1000

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

## Available SLO Metrics

- **End-to-End**: `e2e_mean_ms`, `e2e_p90_ms`, `e2e_p95_ms`, `e2e_p99_ms`
- **Time to First Token (TTFT)**: `ttft_mean_ms`, `ttft_p90_ms`, `ttft_p95_ms`, `ttft_p99_ms`
- **Inter-Token Latency (ITL)**: `itl_mean_ms`, `itl_p90_ms`, `itl_p95_ms`, `itl_p99_ms`

## Output

### Console Output

```
QPS Search - BLIS Simulator
============================================================

Configuration:
  Model: codellama/CodeLlama-34b-Instruct-hf
  Hardware: H100
  TP: 1
  Batch Size: 256
  Hardware Config: hardware_config.json
  Model Config Base: model_configs
  Roofline Model: ENABLED

Starting Binary Search for Max QPS
...

============================================================
✅ Search completed successfully!
============================================================

Results:
  Max QPS: 15.50
  Total Runtime: 182.47 seconds

SLO Metrics at Max QPS:
  ✅ e2e_p95_ms: 987.32 ms (SLO: 1000 ms)
  ✅ ttft_p90_ms: 456.78 ms (SLO: 500 ms)

Best Config achieves 15.50 QPS
Total Search Runtime: 182.47 seconds
```

### JSON Output (--output results.json)

```json
{
  "metadata": {
    "simulator": "BLIS",
    "timestamp": "2026-01-27T10:30:00",
    "model": "codellama/CodeLlama-34b-Instruct-hf",
    "hardware": "H100"
  },
  "summary": {
    "total_configs_evaluated": 8,
    "successful_configs": 7,
    "failed_configs": 1,
    "best_max_qps": 15.50,
    "total_search_runtime_seconds": 182.47
  },
  "successful_configs": [
    {
      "rank": 1,
      "max_qps": 15.50,
      "runtime_seconds": 45.32,
      "configuration": {
        "tp": 1,
        "batch_size": 256,
        "max_scheduled_tokens": 8192,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90
      },
      "slo_metrics": {
        "e2e_p95_ms": {
          "value_ms": 987.32,
          "threshold_ms": 1000,
          "passes": true
        }
      }
    }
  ]
}
```

## BLIS vs Vidur

| Aspect | BLIS | Vidur |
|--------|------|-------|
| **Speed** | Fast (~5-10s/config) | Slower (~30-60s/config) |
| **Accuracy** | High (more accurate) | Lower (less accurate) |
| **Latency Model** | Linear coefficients | Random Forest ML |
| **Setup** | Simple (pre-trained) | Complex (GPU profiling) |
| **Recommendation** | Production use | Research/ML experiments |

**Use BLIS for production capacity planning** - it's faster and more accurate.

## Project Structure

```
config-explorer-evaluation/
├── qps_search.py              # Single-config binary search
├── parallel_search.py         # Multi-config parallel search
├── blis_runner.py            # BLIS simulator interface
├── vidur_runner.py           # Vidur simulator interface
├── capacity_planner.py       # KV cache capacity calculation
├── test_config.json          # Example JSON config
├── examples/
│   ├── configs_grid_search.yaml   # Grid search example
│   └── configs_explicit.yaml      # Explicit configs example
├── SETUP.md                  # Detailed setup guide
├── README_STEP*.md           # Step-by-step guides
└── STEP*_SUMMARY.md          # Implementation summaries
```

## Documentation

- **[SETUP.md](SETUP.md)** - Detailed setup instructions
- **[README_STEP1.md](README_STEP1.md)** - BLIS runner and capacity planner
- **[README_STEP2.md](README_STEP2.md)** - Binary search for max QPS
- **[README_STEP3.md](README_STEP3.md)** - Parallel config search
- **[README_STEP4.md](README_STEP4.md)** - Vidur integration

## Requirements

- Python 3.11+
- Go 1.21+ (for BLIS)
- Dependencies:
  - `config_explorer` library from llm-d-benchmark
  - `numpy`, `pandas`, `pyyaml`
  - Standard library modules

## Key Features Explained

### Automatic KV Cache Calculation

The tool automatically calculates `total_kv_blocks` from:
- Model architecture (fetched from HuggingFace)
- Hardware memory (H100 = 80GB)
- `max_model_len` (user-specified)
- `gpu_memory_utilization` (user-specified)

No need to manually tune `total_kv_blocks`!

### Roofline Model Support

For hardware without pre-trained coefficients, specify:
```json
{
  "model_config_folder_base": "model_configs",
  "hardware_config": "hardware_config.json"
}
```

BLIS will use the roofline model for accurate predictions.

### Runtime Tracking

All tools report:
- Total search runtime
- Per-config runtime (in parallel_search.py)
- Simulation time per QPS test

Helps identify performance bottlenecks.

## Contributing

This is a research tool for vLLM configuration optimization. Contributions welcome!

## License

See individual component licenses (BLIS, Vidur, llm-d-benchmark).
