# Quick Setup Guide

Complete setup instructions for running Step 1: BLIS Runner with Capacity Planner.

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
go build -o simulation_worker main.go

# Create defaults.yaml
cat coefficients.yaml workloads.yaml > defaults.yaml

cd ..
```

### 4. Test Installation

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

Once installed, you can:

```bash
# Run capacity planner
python capacity_planner.py

# Test with your config
python test_blis.py <config.json> <qps> <num_requests>

# Example
python test_blis.py test_config.json 5.0 50
```

## Next Steps

See [README_STEP1.md](README_STEP1.md) for detailed usage instructions and API examples.
