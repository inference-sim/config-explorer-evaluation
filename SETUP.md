# Quick Setup Guide

Complete setup instructions for running Steps 1, 2 & 3: BLIS Runner with Capacity Planner, Binary Search for Max QPS, and Parallel Config Search.

## Prerequisites

- Python ≥ 3.11
- Go ≥ 1.21
- Git

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/inference-sim/config-explorer-evaluation.git
cd config-explorer-evaluation
git submodule update --init --recursive
```

### 2. Install config_explorer Library (Required!)

```bash
# Clone llm-d-benchmark
git clone https://github.com/llm-d/llm-d-benchmark.git

# Install config_explorer
pip install -e ./llm-d-benchmark/config_explorer

# Verify installation
python -c "from config_explorer.capacity_planner import get_model_info_from_hf; print('✅ Installed')"
```

**Troubleshooting numpy errors:**
```bash
pip install --upgrade --force-reinstall scikit-learn pyarrow
```

### 3. Build BLIS Simulator

```bash
cd inference-sim
git checkout openevolve
go build -o ../simulation_worker main.go

# Create defaults.yaml
cat coefficients.yaml workloads.yaml > defaults.yaml

cd ..
```

### 4. Set Environment Variables (Recommended)

```bash
# Set BLIS_ROOT to your config-explorer-evaluation directory
export BLIS_ROOT=$(pwd)

# Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo "export BLIS_ROOT=$(pwd)" >> ~/.bashrc
```

**What is BLIS_ROOT?**
- Environment variable pointing to the config-explorer-evaluation root directory
- Defaults to current directory if not set
- All relative paths in configs are resolved relative to BLIS_ROOT

### 5. Test Installation

```bash
# Test capacity planner
python capacity_planner.py

# Test BLIS integration
python test_blis.py test_config.json 2.0 30
```

## Expected Output

**Capacity Planner:**
```
Fetching model info from HuggingFace: meta-llama/llama-3.1-8b-instruct

Capacity Planning Results:
  Model: meta-llama/llama-3.1-8b-instruct
  Hardware: H100 (80GB)
  Model Memory: 14.96 GiB
  Total KV Blocks: 29,205

Result: 29205 KV cache blocks
```

**BLIS Test:**
```
✅ Simulation completed successfully!

Key Metrics:
  P90 End-to-End Latency: 6065.86 ms
  P90 TTFT: 38.36 ms
```

## Common Issues

### ModuleNotFoundError: No module named 'config_explorer'

**Solution:** Install the config_explorer library:
```bash
git clone https://github.com/llm-d/llm-d-benchmark.git
pip install -e ./llm-d-benchmark/config_explorer
```

### BLIS: "panic: open defaults.yaml: no such file or directory"

**Solution:** Create defaults.yaml in inference-sim directory:
```bash
cd inference-sim
cat coefficients.yaml workloads.yaml > defaults.yaml
cd ..
```

### numpy.dtype size changed error

**Solution:** Reinstall scikit-learn and pyarrow:
```bash
pip install --upgrade --force-reinstall scikit-learn pyarrow
```

## Usage

Once installed, you can run both Step 1 and Step 2 tools:

### Step 1: Single Simulation Run

```bash
# Run capacity planner
python capacity_planner.py

# Test with your config
python test_blis.py <config.json> <qps> <num_requests>

# Example
python test_blis.py test_config.json 5.0 50
```

### Step 2: Binary Search for Max QPS

```bash
# BLIS simulator (fast and accurate, recommended)
python qps_search.py --config test_config.json --simulator blis

# Vidur simulator (ML-based)
python qps_search.py --config test_config.json --simulator vidur

# With YAML config
python qps_search.py -c examples/configs_grid_search.yaml --simulator blis

# With trace file (Vidur only)
python qps_search.py -c test_config.json --simulator vidur --trace traces/chat.csv

# Custom search parameters
python qps_search.py -c test_config.json --simulator blis \
  --qps-max 50 --qps-granularity 0.1
```

**Note**: Your config file must include `"num_requests"` and `"slos"` fields:
```json
{
  "model": "...",
  "num_requests": 500,
  "slos": [
    {"metric": "e2e_p95_ms", "threshold_ms": 1000}
  ],
  ...
}
```

### Step 3: Parallel Config Search

```bash
# BLIS grid search (evaluates all TP, batch size, etc. combinations)
python parallel_search.py --configs examples/configs_grid_search.yaml --simulator blis

# Vidur simulator
python parallel_search.py --configs examples/configs_grid_search.yaml --simulator vidur

# Explicit configs
python parallel_search.py --configs examples/configs_explicit.yaml --simulator blis

# With custom workers
python parallel_search.py -c examples/configs_grid_search.yaml --simulator blis --num-workers 4

# Save results to JSON
python parallel_search.py -c examples/configs_grid_search.yaml --simulator blis --output results.json
```

**Grid search example** sweeps over:
- TP: [1, 2]
- Batch sizes: [128, 256, 512]
- Max scheduled tokens: [2048, 4096]
- Max model length: [4096, 8192]
- GPU memory utilization: [0.90]

This automatically generates all combinations (24 configs) and finds the best one.

## New Features

### Runtime Tracking
Both tools now report:
- Total search runtime
- Per-config runtime (in parallel_search.py)
- Displayed in console and saved to JSON output

### Simulator Selection
All tools require `--simulator` flag:
- `--simulator blis` - Fast and accurate (recommended)
- `--simulator vidur` - ML-based Random Forest model

### YAML Support
qps_search.py now supports YAML files (uses first value from grid search lists).

## Next Steps

- See [README_STEP1.md](README_STEP1.md) for Step 1 detailed usage and API examples
- See [README_STEP2.md](README_STEP2.md) for Step 2 binary search documentation
- See [README_STEP3.md](README_STEP3.md) for Step 3 parallel config search documentation
