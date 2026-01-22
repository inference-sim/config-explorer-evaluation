# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains two main subprojects for evaluating and comparing LLM inference simulators:

1. **inference-sim** (BLIS): A Go-based discrete-event simulator using trained performance coefficients
2. **vidur**: A Python-based high-fidelity LLM inference cluster simulator from Microsoft Research

Both simulators predict LLM serving performance (TTFT, TPOT, throughput) without requiring GPUs.

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
└── vidur/                 # Vidur simulator (Python)
    ├── vidur/
    │   ├── config/                    # Configuration system
    │   ├── config_optimizer/          # Capacity planning & config search
    │   │   ├── config_explorer/       # Automated config search
    │   │   └── analyzer/              # Results analysis & dashboard
    │   ├── entities/                  # Core data structures
    │   ├── events/                    # Event system for DES
    │   ├── execution_time_predictor/  # ML models for latency prediction
    │   ├── metrics/                   # Performance metrics collection
    │   ├── profiling/                 # Hardware profiling tools
    │   └── request_generator/         # Workload generators
    └── environment.yml                # Conda/Mamba dependencies
```

## Building and Running

### BLIS (inference-sim)

**Build:**
```bash
cd inference-sim
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

## Key Differences Between Simulators

| Aspect | BLIS (inference-sim) | Vidur |
|--------|---------------------|-------|
| Language | Go | Python |
| Latency Model | Linear coefficients (α/β) | Random Forest ML models |
| Calibration | Bayesian optimization | GPU profiling + supervised learning |
| Features | Prefix caching, chunked prefill | Multi-replica, PP/TP, multiple schedulers |
| Config Search | Manual parameter sweep | Automated capacity search with Ray |
| Visualization | CLI metrics output | Streamlit dashboard, Chrome traces |
| Use Case | Fast capacity planning | Research on scheduling algorithms |

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

## Notes

- BLIS requires Go ≥ 1.21
- Vidur requires Python ≥ 3.10
- Both simulators are CPU-only and do not require GPUs for inference (profiling phase for Vidur requires GPUs)
- WandB is optional for Vidur; disable with `export WANDB_MODE=disabled`
- Chrome traces are useful for understanding simulation execution flow in Vidur
