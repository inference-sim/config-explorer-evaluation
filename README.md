# vLLM Configuration Optimizer

Find optimal vLLM configurations that maximize QPS while meeting tail latency SLOs.

## Prerequisites

```bash
# Python 3.11+, Go 1.21+
python --version && go version

# Install config_explorer library
git clone https://github.com/llm-d/llm-d-benchmark.git
pip install -e ./llm-d-benchmark/config_explorer

# Build BLIS (openevolve branch)
cd inference-sim && git checkout openevolve
go build -o ../simulation_worker main.go && cd ..

# Set environment variable
export BLIS_ROOT=$(pwd)
```

## Task 1: Saturation Detection

**Goal:** Find maximum QPS for a **single configuration** that meets SLO constraints.

### Step 1: Local Simulation to get Saturation Thresholds

```bash
# Find max QPS using BLIS simulator
python qps_search.py --config test_config_lowprefix.json --simulator blis --output results.json

# Output: Max QPS, SLO metrics, runtime
```

### Step 2: Real vLLM SLO Validation

```bash
# Validate on Kubernetes with real vLLM + GuideLLM
python saturation_orchestrator.py \
  --results results.json \
  --config-name test_config_lowprefix.json \
  --simulator blis \
  --use-k8s \
  --namespace diya \
  --output-dir saturation_results
```

## Task 2: Config Exploration

**Goal:** Find the **best configuration** from multiple candidates.

### Step 1: Simulate Multiple Configs

```bash
# Parallel search across config space
python parallel_search.py \
  --configs examples/configs_grid_search.yaml \
  --simulator blis \
  --output results/config_exp/blis_config_exploration.json
```

### Step 2: Validate Top Configs on Real vLLM

```bash
# Validate top 3 configs in parallel on Kubernetes (creates 3 pods concurrently)
python config_validator.py \
  --simulator blis \
  --results results/config_exp/blis_config_exploration.json \
  --top-n 3 \
  --namespace diya \
  --output-file blis_validation_report.json

# Do the same for Vidur (if testing both simulators)
python config_validator.py \
  --simulator vidur \
  --results results/config_exp/vidur_config_exploration.json \
  --top-n 3 \
  --namespace diya \
  --output-file vidur_validation_report.json

# Creates:
# - blis_validation_report.json (summary + metrics)
# - vidur_validation_report.json (summary + metrics)
# Note: Runs in true parallel (3 configs = 3× faster)
# Temp logs cleaned up automatically (use --keep-logs to preserve)
```

### Step 3: Compare Simulator Predictions vs Real vLLM

```bash
# Generate 6 comparison plots showing prediction accuracy
python compare_simulators.py \
  blis_validation_report.json \
  vidur_validation_report.json \
  -o validation_comparison_plots/
```

## Configuration Files

### JSON (for qps_search.py)

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
  "num_requests": 500,
  "slos": [
    {"metric": "e2e_p95_ms", "threshold_ms": 1000}
  ]
}
```

### YAML (for parallel_search.py)

```yaml
model: codellama/CodeLlama-34b-Instruct-hf
hardware: H100
num_requests: 100

slos:
  - metric: e2e_p95_ms
    threshold_ms: 1000

# Grid search (Cartesian product)
tp: [1, 2]
batch_size: [128, 256, 512]
max_scheduled_tokens: [2048, 4096]
max_model_len: [4096, 8192]
gpu_memory_utilization: [0.90]
block_size: [16]
```

## Available SLO Metrics

- **End-to-End**: `e2e_p90_ms`, `e2e_p95_ms`, `e2e_p99_ms`
- **Time to First Token**: `ttft_p90_ms`, `ttft_p95_ms`, `ttft_p99_ms`
- **Inter-Token Latency**: `itl_p90_ms`, `itl_p95_ms`, `itl_p99_ms`

## Output Examples

### Saturation Detection (qps_search.py)

```
✅ Search completed successfully!

Results:
  Max QPS: 15.50

SLO Metrics at Max QPS:
  ✅ e2e_p95_ms: 987.32 ms (SLO: 1000 ms)
```

### Config Exploration (parallel_search.py)

```json
{
  "summary": {
    "total_configs_evaluated": 24,
    "best_config_id": 2,
    "best_max_qps": 15.50
  },
  "successful_configs": [
    {
      "rank": 1,
      "config_id": 2,
      "max_qps": 15.50,
      "configuration": {
        "tp": 1,
        "batch_size": 256,
        "max_scheduled_tokens": 8192
      }
    }
  ]
}
```

### Validation Report (config_validator.py)

```json
{
  "timestamp": "2026-01-28 10:30:00",
  "simulator": "BLIS",
  "summary": {
    "total_configs_tested": 3,
    "configs_meeting_slos": 2,
    "total_guidellm_runtime_seconds": 1245.67
  },
  "results": [{
    "config_id": 2,
    "rank": 1,
    "meets_slos": true,
    "output_dir": "validation_report_configs/config_2_rank_1",
    "slo_metrics": {
      "e2e_p95_ms": {
        "simulator_ms": 987.32,
        "real_ms": 995.10,
        "error_percent": 0.79,
        "passes": true
      }
    }
  }]
}
```

## Documentation

- **[all_docs/README_CONFIG_VALIDATOR.md](all_docs/README_CONFIG_VALIDATOR.md)** - Config validation guide
- **[all_docs/README_SATURATION_VALIDATION.md](all_docs/README_SATURATION_VALIDATION.md)** - Saturation detection guide
- **[SETUP.md](SETUP.md)** - Detailed setup instructions

- Python 3.11+, Go 1.21+
- Kubernetes cluster with H100 GPUs (for validation only)
- Dependencies: `config_explorer`, `numpy`, `pyyaml`, `kubernetes`
