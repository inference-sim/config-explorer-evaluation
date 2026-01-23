# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains three main projects for LLM inference simulation and capacity planning:

1. **inference-sim** (BLIS): A Go-based discrete-event simulator using trained performance coefficients
2. **vidur**: A Python-based high-fidelity LLM inference cluster simulator from Microsoft Research
3. **config_search**: A Python-based capacity planning tool that finds optimal vLLM configurations for single GPU deployments

The simulators predict LLM serving performance (TTFT, TPOT, throughput) without requiring GPUs. The config search tool uses these simulators to automatically find the best vLLM configuration that maximizes QPS while meeting tail latency SLOs.

## Repository Structure

```
.
├── inference-sim/          # BLIS: Blackbox Inference Simulator (Go)
│   ├── cmd/               # CLI command definitions (Cobra)
│   ├── sim/               # Core simulation engine
│   ├── model_configs/     # LLM config files (config.json)
│   ├── coefficients.yaml  # Pre-trained α/β coefficients
│   ├── workloads.yaml     # Preset workload definitions
│   └── main.go           # Entry point
├── vidur/                 # Vidur simulator (Python)
│   ├── vidur/
│   │   ├── config/                    # Configuration system
│   │   ├── config_optimizer/          # Capacity planning & config search
│   │   │   ├── config_explorer/       # Automated config search
│   │   │   └── analyzer/              # Results analysis & dashboard
│   │   ├── entities/                  # Core data structures
│   │   ├── events/                    # Event system for DES
│   │   ├── execution_time_predictor/  # ML models for latency prediction
│   │   ├── metrics/                   # Performance metrics collection
│   │   ├── profiling/                 # Hardware profiling tools
│   │   └── request_generator/         # Workload generators
│   └── environment.yml                # Conda/Mamba dependencies
└── config_search/         # Unified capacity planning tool
    ├── blis_runner.py         # BLIS simulator interface
    ├── capacity_planner.py    # Wrapper for config_explorer library
    ├── qps_search.py          # Binary search for max QPS
    ├── parallel_search.py     # Parallel config evaluation
    ├── vidur_wrapper.py       # Vidur config generator + runner
    ├── config_search.py       # Main CLI
    ├── requirements.txt       # Python dependencies
    └── examples/
        ├── configs_blis.yaml      # BLIS config space examples
        ├── configs_vidur.yaml     # Vidur config space examples
        └── traces/                # Workload trace files (prompt_tokens, output_tokens)

External Dependencies:
├── config_explorer (from llm-d-benchmark)
    └── Used for KV cache capacity calculations via Python library
```

## Building and Running
- Create a venv if .venv does not exist and then continue

### BLIS (inference-sim)

**Note:** For use with the Config Search Tool, checkout the `openevolve` branch which includes trace file support.

**Build:**
```bash
cd inference-sim
# For config search tool usage:
# git checkout openevolve
go build -o simulation_worker main.go
```

**Run basic simulation:**
```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct
```

**Run with custom workload:**
```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload chatbot \
  --hardware H100 \
  --tp 1 \
  --vllm-version vllm/vllm-openai:v0.8.4
```

**Run with custom parameters:**
```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --rate 10 \
  --max-prompts 300 \
  --prompt-tokens 800 \
  --prompt-tokens-stdev 300 \
  --output-tokens 400 \
  --output-tokens-stdev 200
```

**Run with trace file (openevolve branch):**
```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --hardware H100 \
  --tp 1 \
  --rate 10 \
  --trace-file traces/chat.csv \
  --max-num-running-reqs 256 \
  --max-num-scheduled-tokens 8192 \
  --total-kv-blocks 140000 \
  --block-size-in-tokens 16
```

**Run with roofline approach (no pre-trained coefficients):**
```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --hardware H100 \
  --tp 1 \
  --vllm-version vllm/vllm-openai:v0.8.4 \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json
```

### Vidur

**Setup environment:**
```bash
cd vidur

# Using mamba (recommended)
mamba env create -p ./env -f ./environment.yml
mamba env update -f environment-dev.yml

# Or using venv with Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Run simulator:**
```bash
python -m vidur.main
```

**Run with full configuration:**
```bash
python -m vidur.main \
  --replica_config_device a100 \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --cluster_config_num_replicas 1 \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 512 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 16384 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 6.45 \
  --replica_scheduler_config_type sarathi \
  --sarathi_scheduler_config_batch_size_cap 512 \
  --sarathi_scheduler_config_chunk_size 512
```

**Run config explorer (capacity planning):**
```bash
python -m vidur.config_optimizer.config_explorer.main \
  --output-dir ./output \
  --config-path ./config.yaml \
  --scheduling-delay-slo-value 5.0 \
  --scheduling-delay-slo-quantile 0.99
```

**Run dashboard (visualize results):**
```bash
python -m streamlit run vidur/config_optimizer/analyzer/dashboard/main.py -- \
  --sim-results-dir ./output \
  --scheduling-delay-slo-percentile 95 \
  --scheduling-delay-slo-value 2.0
```

**Format code:**
```bash
make format  # Runs black and isort
```

**Lint code:**
```bash
make lint  # Runs black and isort checks
```

### Config Search Tool

**Purpose:** Find optimal vLLM configuration for single H100 GPU that maximizes QPS while meeting tail latency SLOs.

**Setup:**
```bash
cd config_search

# Install config_explorer library for capacity planning
git clone https://github.com/llm-d/llm-d-benchmark.git
pip install -e ./llm-d-benchmark/config_explorer

# Install other dependencies
pip install -r requirements.txt
```

**Note:** The config_explorer library requires Python 3.11+.

**Run BLIS-based search:**
```bash
# Note: Use openevolve branch of inference-sim
python config_search.py \
  --simulator blis \
  --configs configs_blis.yaml \
  --trace traces/chat.csv \
  --slo-p90-ms 1000 \
  --num-workers 8
```

**Run Vidur-based search:**
```bash
python config_search.py \
  --simulator vidur \
  --configs configs_vidur.yaml \
  --trace traces/chat.csv \
  --slo-p90-ms 1000
```

**Config space example (BLIS):**
```yaml
model: meta-llama/llama-3.1-8b-instruct
hardware: H100
tp: 1

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

  - batch_size: 512
    max_scheduled_tokens: 8192
    max_model_len: 16384
    gpu_memory_utilization: 0.85
    block_size: 16
```

**Key features:**
- Binary search to find max QPS with 0.1 precision
- Parallel evaluation of multiple configs
- Automatic calculation of `total_kv_blocks` using config_explorer library
- JSON output with best config and all results
- Support for custom workload trace files (CSV format: prompt_tokens, output_tokens)

**Using config_explorer library for capacity planning:**
```python
from config_explorer.capacity_planner import (
    get_model_info_from_hf,
    get_model_config_from_hf,
    total_kv_cache_blocks,
    gib_to_bytes
)

# Get model information from HuggingFace
model_name = "meta-llama/llama-3.1-8b-instruct"
model_info = get_model_info_from_hf(model_name)
model_config = get_model_config_from_hf(model_name)

# Calculate total KV cache blocks
total_blocks = total_kv_cache_blocks(
    model_info=model_info,
    model_config=model_config,
    context_len=8192,              # max_model_len
    gpu_memory=80,                 # H100 = 80 GiB
    gpu_mem_util=0.90,             # gpu_memory_utilization
    batch_size=1,
    block_size=16,
    tp=1,                          # tensor parallel size
    pp=1,                          # pipeline parallel size
    dp=1                           # data parallel size
)
# Returns: ~140000 blocks (for llama-3.1-8b on H100 with these settings)
```

**Key functions from config_explorer.capacity_planner:**
- `get_model_info_from_hf(model_name, hf_token=None)`: Fetches model metadata from HuggingFace
- `get_model_config_from_hf(model_name, hf_token=None)`: Fetches model architecture config
- `total_kv_cache_blocks(...)`: Calculates available KV cache blocks after model loading
- `model_memory_req(model_info, model_config)`: Returns GPU memory (GiB) needed for model weights
- `max_context_len(model_config)`: Extracts max_position_embeddings from model config
- `gib_to_bytes(gib)`: Converts GiB to bytes

### config_explorer Library (External Dependency)

**Source:** https://github.com/llm-d/llm-d-benchmark/tree/main/config_explorer

**Purpose:** Provides capacity planning functions for LLM serving, including GPU memory calculations and KV cache sizing.

**Installation:**
```bash
git clone https://github.com/llm-d/llm-d-benchmark.git
pip install -e ./llm-d-benchmark/config_explorer
```

**Requirements:**
- Python 3.11+
- Dependencies: huggingface_hub, transformers, numpy, pandas, pydantic, PyYAML

**Key Module:** `config_explorer.capacity_planner`

**Main Functions:**
1. **`total_kv_cache_blocks(model_info, model_config, context_len, gpu_memory, gpu_mem_util=0.9, batch_size=1, block_size=16, tp=1, pp=1, dp=1)`**
   - Calculates total number of KV cache blocks that fit in GPU memory
   - Accounts for model weights, KV cache per token, and parallelism strategy
   - Returns: `int` (number of blocks)

2. **`get_model_info_from_hf(model_name, hf_token=None)`**
   - Fetches model metadata from HuggingFace Hub
   - Returns: `ModelInfo` object with parameter counts, safetensors info

3. **`get_model_config_from_hf(model_name, hf_token=None)`**
   - Retrieves model architecture configuration
   - Returns: `AutoConfig` object with layer counts, hidden sizes, attention config

4. **`model_memory_req(model_info, model_config)`**
   - Calculates GPU memory required to load model weights
   - Handles quantization (INT8, INT4) and different precisions
   - Returns: `float` (memory in GiB)

5. **`max_context_len(model_config)`**
   - Extracts maximum context length from model config
   - Returns: `int` (max_position_embeddings)

**Attention Mechanisms Supported:**
- MHA (Multi-Head Attention): Standard transformer attention
- GQA (Grouped-Query Attention): Used in Llama 3, Mistral
- MQA (Multi-Query Attention): Single KV head shared across queries
- MLA (Multi-Latent Attention): Used in DeepSeek models

**Usage Pattern in Config Search:**
```python
# 1. Get model information from HuggingFace
model_info = get_model_info_from_hf("meta-llama/llama-3.1-8b-instruct")
model_config = get_model_config_from_hf("meta-llama/llama-3.1-8b-instruct")

# 2. Calculate model memory requirements
model_mem_gib = model_memory_req(model_info, model_config)  # ~16 GiB for 8B model

# 3. Calculate total KV blocks for given config
blocks = total_kv_cache_blocks(
    model_info, model_config,
    context_len=8192,
    gpu_memory=80,        # H100
    gpu_mem_util=0.90,
    block_size=16,
    tp=1
)  # Returns ~140,000 blocks
```

**Note:** Set `HUGGING_FACE_HUB_TOKEN` environment variable if accessing gated models.

## Architecture

### BLIS (inference-sim)

**Discrete Event Simulator:** Uses a priority queue-based event loop advancing a simulation clock without real GPU execution.

**Key Components:**
- **Event Engine**: Priority queue managing simulation clock and callbacks
- **Virtual Scheduler**: Mirrors vLLM's scheduler with Waiting/Running/Preempted queues
- **Virtual KV-Cache Manager**: Simulates PagedAttention block allocation, prefix caching, and memory management
- **Latency Model**: Predicts GPU iteration time using learned coefficients:
  - GPU time: `L_gpu = β₀ + β₁·X + β₂·Y` where X=prefill tokens, Y=decode tokens
  - CPU overhead: `L_cpu = α₀ + α₁·M + α₂·N` where M=input length, N=output length

**Configuration:**
- Pre-trained coefficients stored in `coefficients.yaml` for specific (Model, GPU, TP, vLLM version) combinations
- Roofline approach available when coefficients not trained (requires model config.json and hardware_config.json)
- Preset workloads in `workloads.yaml`: chatbot, summarization, contentgen, multidoc

**Entry point:** `main.go` delegates to Cobra CLI in `cmd/root.go`

### Vidur

**Discrete Event Simulator:** High-fidelity simulation of vLLM-style inference systems with support for multiple replicas, scheduling policies, and tensor/pipeline parallelism.

**Key Components:**
- **Entities** (`vidur/entities/`): Core data structures including Request, Batch, Replica, Cluster
- **Events** (`vidur/events/`): Event types for DES (request arrival, batch execution, scheduling)
- **Execution Time Predictors** (`vidur/execution_time_predictor/`): ML models (Random Forest, Linear Regression) for predicting GPU kernel execution times
- **Schedulers**: Multiple scheduling policies including vLLM baseline, Sarathi, and custom variants
- **Config Optimizer** (`vidur/config_optimizer/`):
  - **config_explorer**: Automated capacity search finding optimal configurations
  - **analyzer**: Post-simulation analysis tools including Streamlit dashboard for visualizing Pareto curves, cost analysis, and config comparisons
- **Profiling** (`vidur/profiling/`): Tools for profiling real GPU hardware to train execution time predictors

**Configuration System:**
- Hierarchical config using `vidur/config/` with CLI overrides
- Config explorer uses YAML files defining search spaces
- Analyzer dashboard reads simulation output CSVs

**Output:**
- Metrics logged to WandB (optional) and `simulator_output/<TIMESTAMP>/`
- Chrome traces for visualization (`chrome://tracing/`)
- CSV files with detailed per-request and aggregate metrics

### Config Search Tool

**Capacity Planning System:** Orchestrates BLIS or Vidur simulators to find optimal vLLM configurations for single GPU deployments.

**Key Components:**
- **BLIS Runner** (`blis_runner.py`): Wraps BLIS simulation_worker CLI, parses JSON output metrics (TTFT, TPOT, E2E latency percentiles)
- **Capacity Planner** (`capacity_planner.py`): Uses config_explorer library to calculate `total_kv_blocks` from model architecture, GPU memory (80GB for H100), `max_model_len`, and `gpu_memory_utilization` target
- **QPS Search** (`qps_search.py`): Binary search algorithm to find max QPS where tail latency meets SLO (0.1 QPS precision)
- **Parallel Search** (`parallel_search.py`): Multiprocessing pool to evaluate N configs concurrently, returns best config by max QPS
- **Vidur Wrapper** (`vidur_wrapper.py`): Generates Vidur YAML configs and invokes built-in config_explorer

**Algorithm:**
1. For each config in search space:
   - Calculate `total_kv_blocks` using capacity planner
   - Binary search to find max QPS meeting SLO threshold
   - Record max QPS and corresponding metrics
2. Return config with highest max QPS

**Configuration:**
- **Fixed**: Hardware=H100, TP=1 (single GPU)
- **Search space**:
  - `batch_size`: [64, 128, 256, 512]
  - `max_scheduled_tokens`: [2048, 4096, 8192]
  - `max_model_len`: [4096, 8192, 16384]
  - `gpu_memory_utilization`: [0.85, 0.90, 0.95]
  - `block_size`: [16, 32]
- **Derived**: `total_kv_blocks` calculated automatically

**Integration:**
- Uses `openevolve` branch of inference-sim (BLIS) with trace file support
- Uses config_explorer library from llm-d-benchmark (https://github.com/llm-d/llm-d-benchmark) as dependency for capacity planning
- Unified interface for both BLIS and Vidur simulators

## Key Differences Between Tools

| Aspect | BLIS (inference-sim) | Vidur | Config Search Tool |
|--------|---------------------|-------|-------------------|
| Language | Go | Python | Python |
| Latency Model | Linear coefficients (α/β) | Random Forest ML models | Uses BLIS or Vidur |
| Calibration | Bayesian optimization | GPU profiling + supervised learning | N/A (orchestrator) |
| Features | Prefix caching, chunked prefill | Multi-replica, PP/TP, multiple schedulers | Binary search, parallel eval, capacity planning |
| Config Search | Manual parameter sweep | Automated capacity search with Ray | Binary search for max QPS with SLO constraints |
| Visualization | CLI metrics output | Streamlit dashboard, Chrome traces | JSON output with metrics |
| Use Case | Fast capacity planning | Research on scheduling algorithms | Production config optimization for single GPU |
| Target | Any hardware/TP config | Multi-GPU clusters | Single H100 GPU (TP=1) |

## Common Development Tasks

When modifying BLIS:
- Core simulation logic is in `sim/simulator.go`
- Scheduling behavior in `sim/queue.go` and `sim/batch.go`
- KV-cache management in `sim/kvcache.go`
- Latency computation in `sim/roofline_step.go` (roofline) or uses coefficients directly
- Add new CLI flags in `cmd/root.go`

When modifying Vidur:
- Simulator entry point: `vidur/main.py`
- Add new scheduler: Implement in `vidur/scheduler/` and register
- Add new execution time predictor: Implement in `vidur/execution_time_predictor/` and register
- Config explorer modifications: `vidur/config_optimizer/config_explorer/config_explorer.py`
- Dashboard pages: `vidur/config_optimizer/analyzer/dashboard/`

When modifying Config Search Tool:
- Main entry point: `config_search/config_search.py` (CLI interface)
- BLIS integration: `config_search/blis_runner.py` (subprocess management, JSON parsing)
- Capacity calculations: `config_search/capacity_planner.py` (wrapper around config_explorer library)
  - Uses `config_explorer.capacity_planner.total_kv_cache_blocks()` for KV block calculations
  - Requires HuggingFace model info and config via `get_model_info_from_hf()` and `get_model_config_from_hf()`
- Search algorithm: `config_search/qps_search.py` (binary search with SLO constraints)
- Parallelization: `config_search/parallel_search.py` (multiprocessing pool)
- Vidur integration: `config_search/vidur_wrapper.py` (YAML generation, result parsing)
- Config spaces defined in YAML: `config_search/examples/configs_blis.yaml`, `configs_vidur.yaml`
- Workload traces: CSV format with columns `prompt_tokens`, `output_tokens`

**Development workflow:**
1. Install config_explorer library: `pip install -e ./llm-d-benchmark/config_explorer`
2. Use `openevolve` branch of inference-sim submodule
3. Test single config run with `blis_runner.py` first
4. Verify binary search convergence with `qps_search.py`
5. Scale to parallel search with `parallel_search.py`
6. Output format: JSON with `best_config`, `max_qps`, `metrics`, and `all_results`

**Example capacity calculation:**
```python
from config_explorer.capacity_planner import (
    get_model_info_from_hf,
    get_model_config_from_hf,
    total_kv_cache_blocks
)

def calculate_total_kv_blocks(model, max_model_len, gpu_memory_utilization):
    model_info = get_model_info_from_hf(model)
    model_config = get_model_config_from_hf(model)

    return total_kv_cache_blocks(
        model_info=model_info,
        model_config=model_config,
        context_len=max_model_len,
        gpu_memory=80,  # H100 = 80 GiB
        gpu_mem_util=gpu_memory_utilization,
        batch_size=1,
        block_size=16,
        tp=1, pp=1, dp=1
    )
```

## Notes

- BLIS requires Go ≥ 1.21
- Vidur requires Python ≥ 3.10
- Config Search Tool requires:
  - Python ≥ 3.11 (required by config_explorer library)
  - config_explorer library from llm-d-benchmark: `pip install -e ./llm-d-benchmark/config_explorer`
  - Either BLIS or Vidur simulator installed
- **Important**: Config Search Tool requires the `openevolve` branch of inference-sim for trace file support
- All tools are CPU-only and do not require GPUs for inference (profiling phase for Vidur requires GPUs)
- WandB is optional for Vidur; disable with `export WANDB_MODE=disabled`
- Chrome traces are useful for understanding simulation execution flow in Vidur
- Config Search Tool uses config_explorer library (v0.3.0) from llm-d-benchmark project for KV cache capacity calculations
  - Library fetches model metadata from HuggingFace Hub
  - Supports various attention mechanisms (MHA, GQA, MQA, MLA)
  - Accounts for model quantization and precision in memory calculations
- Binary search precision: 0.1 QPS, configurable timeout: 300s per simulation run
