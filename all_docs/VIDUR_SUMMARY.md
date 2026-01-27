# Vidur Simulator - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [Installation and Setup](#installation-and-setup)
3. [CLI Arguments Reference](#cli-arguments-reference)
4. [Configuration System](#configuration-system)
5. [Running Vidur - Examples](#running-vidur---examples)
6. [Output Format and Metrics](#output-format-and-metrics)
7. [Parsing Metrics (TTFT, ITL, E2E)](#parsing-metrics-ttft-itl-e2e)
8. [Request Generation](#request-generation)
9. [Scheduling Policies](#scheduling-policies)
10. [Execution Time Predictor](#execution-time-predictor)
11. [Config Explorer (Capacity Planning)](#config-explorer-capacity-planning)
12. [Analyzer Dashboard](#analyzer-dashboard)

---

## Overview

**Vidur** is a high-fidelity discrete-event simulator for LLM inference systems developed by Microsoft Research. It simulates vLLM-style serving clusters without requiring GPUs (after initial profiling).

### Key Features
- **Discrete Event Simulation (DES)**: Priority queue-based event loop for accurate temporal modeling
- **Multiple Schedulers**: vLLM, Sarathi, LightLLM, Orca, FasterTransformer
- **Multi-GPU Support**: Tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP)
- **ML-based Latency Prediction**: Random Forest models trained on GPU profiling data
- **Comprehensive Metrics**: TTFT, TPOT/ITL, E2E latency, throughput, scheduling delays
- **Capacity Planning**: Automated config search (config_explorer) with SLO constraints
- **Visualization**: Streamlit dashboard with Pareto curves, Chrome traces

### Architecture
```
Vidur Simulator
├── Entities: Request, Batch, Replica, Cluster
├── Events: RequestArrival, BatchSchedule, BatchExecution
├── Execution Time Predictor: Random Forest / Linear Regression
├── Schedulers: vLLM, Sarathi, LightLLM, Orca, FasterTransformer
├── Request Generators: Synthetic (Poisson, Gamma) or Trace Replay
└── Metrics Collector: CSV output, Chrome traces, WandB integration
```

---

## Installation and Setup

### Prerequisites
- Python ≥ 3.10
- Mamba or Conda (recommended) or venv

### Using Mamba (Recommended)
```bash
cd vidur

# Create environment
mamba env create -p ./env -f ./environment.yml
mamba env update -f environment-dev.yml

# Activate
mamba activate ./env
```

### Using venv
```bash
cd vidur
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Disable WandB (Optional)
```bash
export WANDB_MODE=disabled
```

---

## CLI Arguments Reference

Vidur uses a hierarchical dataclass-based configuration system that auto-generates CLI arguments. Arguments follow the pattern: `--{parent_class}_{field_name}`.

### Core Argument Groups

#### 1. Replica Configuration (`replica_config_*`)

**Model and Hardware:**
- `--replica_config_model_name <str>`: Model identifier
  - Examples: `meta-llama/Llama-2-7b-hf`, `meta-llama/Meta-Llama-3-8B`, `meta-llama/Llama-2-70b-hf`

- `--replica_config_device <str>`: GPU type
  - Options: `a100` (80GB), `h100`, `a40`

- `--replica_config_network_device <str>`: Network topology
  - Options: `a100_dgx` (8 GPUs NVLink), `a100_pairwise_nvlink`, `a40_pairwise_nvlink`

**Parallelism:**
- `--replica_config_tensor_parallel_size <int>`: Tensor parallelism dimension (TP)
  - Options: 1, 2, 4, 8
  - Default: 1

- `--replica_config_num_pipeline_stages <int>`: Pipeline parallelism dimension (PP)
  - Options: 1, 2, 4, 8
  - Default: 1

**Memory:**
- `--replica_config_memory_margin_fraction <float>`: Safety margin for memory allocation
  - Default: 0.1 (10% margin)
  - Range: 0.0 to 0.3

#### 2. Cluster Configuration (`cluster_config_*`)

- `--cluster_config_num_replicas <int>`: Number of replicas in cluster
  - Default: 1
  - Use >1 for multi-replica simulations with global scheduler

#### 3. Request Generator (`request_generator_config_type`)

**Type Selection:**
- `--request_generator_config_type <str>`: Request generator type
  - Options: `synthetic`, `trace_replay`

**Synthetic Request Generator:**
- `--synthetic_request_generator_config_num_requests <int>`: Total requests to generate
  - Required if `duration` not specified

- `--synthetic_request_generator_config_duration <float>`: Simulation duration in seconds
  - Alternative to `num_requests`

**Trace Replay Generator:**
- `--trace_request_generator_config_trace_file <str>`: Path to trace CSV
  - Format: `timestamp,num_input_tokens,num_output_tokens`

- `--trace_request_generator_config_prefill_scale_factor <float>`: Scale prefill tokens
  - Default: 1.0

- `--trace_request_generator_config_decode_scale_factor <float>`: Scale decode tokens
  - Default: 1.0

- `--trace_request_generator_config_time_scale_factor <float>`: Speed up/slow down arrivals
  - Default: 1.0

#### 4. Length Generator (`length_generator_config_type`)

**Type Selection:**
- `--length_generator_config_type <str>`: Request length distribution
  - Options: `trace`, `fixed`, `uniform`, `zipf`

**Trace Length Generator:**
- `--trace_request_length_generator_config_trace_file <str>`: Path to trace CSV
  - Format: `num_input_tokens,num_output_tokens` (header required)

- `--trace_request_length_generator_config_max_tokens <int>`: Max context length
  - Default: 4096
  - Common values: 4096, 8192, 16384, 32768

- `--trace_request_length_generator_config_prefill_scale_factor <float>`: Scale prefill
- `--trace_request_length_generator_config_decode_scale_factor <float>`: Scale decode

**Fixed Length Generator:**
- `--fixed_request_length_generator_config_prefill_tokens <int>`: Fixed prefill tokens
- `--fixed_request_length_generator_config_decode_tokens <int>`: Fixed decode tokens

**Uniform Length Generator:**
- `--uniform_request_length_generator_config_min_tokens <int>`: Minimum tokens
- `--uniform_request_length_generator_config_max_tokens <int>`: Maximum tokens
- `--uniform_request_length_generator_config_prefill_to_decode_ratio <float>`: P/D ratio

**Zipf Length Generator:**
- `--zipf_request_length_generator_config_theta <float>`: Zipf skewness parameter
- `--zipf_request_length_generator_config_min_tokens <int>`
- `--zipf_request_length_generator_config_max_tokens <int>`
- `--zipf_request_length_generator_config_prefill_to_decode_ratio <float>`

#### 5. Interval Generator (`interval_generator_config_type`)

**Type Selection:**
- `--interval_generator_config_type <str>`: Request arrival distribution
  - Options: `poisson`, `gamma`, `trace`, `static`

**Poisson Interval Generator:**
- `--poisson_request_interval_generator_config_qps <float>`: Target QPS
  - Required
  - Generates exponentially distributed inter-arrival times

**Gamma Interval Generator:**
- `--gamma_request_interval_generator_config_qps <float>`: Target QPS
- `--gamma_request_interval_generator_config_cv <float>`: Coefficient of variation
  - cv=1: Equivalent to Poisson
  - cv>1: More bursty arrivals
  - cv<1: More regular arrivals

**Trace Interval Generator:**
- `--trace_request_interval_generator_config_trace_file <str>`: Trace with timestamps
- `--trace_request_interval_generator_config_start_time <float>`: Start timestamp
- `--trace_request_interval_generator_config_end_time <float>`: End timestamp
- `--trace_request_interval_generator_config_time_scale_factor <float>`: Time scaling

**Static Interval Generator:**
- No parameters (all requests arrive at t=0)

#### 6. Scheduler Configuration (`replica_scheduler_config_type`)

**Type Selection:**
- `--replica_scheduler_config_type <str>`: Scheduler type
  - Options: `vllm`, `sarathi`, `lightllm`, `orca`, `faster_transformer`

**Common Scheduler Parameters (all types):**
- `--{scheduler}_scheduler_config_batch_size_cap <int>`: Max requests per batch
  - Default: 256
  - Common values: 32, 64, 128, 256, 512

- `--{scheduler}_scheduler_config_block_size <int>`: KV cache block size
  - Default: 16 tokens
  - Common values: 8, 16, 32

- `--{scheduler}_scheduler_config_num_blocks <int>`: Total KV cache blocks
  - Default: Auto-calculated based on memory
  - Override for manual control

- `--{scheduler}_scheduler_config_watermark_blocks_fraction <float>`: Memory watermark
  - Default: 0.01 (1% reserved)
  - Range: 0.0 to 0.1

**vLLM-specific:**
- `--vllm_scheduler_config_max_tokens_in_batch <int>`: Max tokens per iteration
  - Default: 16384
  - Limits total tokens (prefill + decode) processed simultaneously

**Sarathi-specific:**
- `--sarathi_scheduler_config_chunk_size <int>`: Chunked prefill size
  - Options: 256, 512, 1024, 2048, 4096
  - Larger chunks: Better throughput, worse TPOT
  - Smaller chunks: Better TPOT, worse TTFT

**LightLLM-specific:**
- `--lightllm_scheduler_config_max_tokens_in_batch <int>`: Max tokens per batch
- `--lightllm_scheduler_config_max_waiting_iters <int>`: Max wait before preemption

#### 7. Execution Time Predictor (`execution_time_predictor_config`)

**Type Selection:**
- `--execution_time_predictor_config_type <str>`: Predictor model type
  - Options: `random_forrest`, `linear_regression`
  - Default: `random_forrest` (recommended)

**Random Forest Predictor:**
- `--random_forrest_execution_time_predictor_config_compute_input_file <str>`: MLP profiling data
  - Default: `data/profiling/compute/{device}/{model}/mlp.csv`

- `--random_forrest_execution_time_predictor_config_attention_input_file <str>`: Attention profiling data
  - Default: `data/profiling/compute/{device}/{model}/attention.csv`

- `--random_forrest_execution_time_predictor_config_all_reduce_input_file <str>`: AllReduce profiling
  - Default: `data/profiling/network/{network_device}/all_reduce.csv`

- `--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size <int>`: Max prefill chunk for prediction
  - Default: 4096
  - **Important:** Use 16384 for Llama-3 models with long contexts

- `--random_forrest_execution_time_predictor_config_prediction_max_batch_size <int>`: Max batch size
  - Default: 128
  - **Important:** Use 512 for large batch sizes

- `--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request <int>`: Max tokens per request
  - Default: 4096

- `--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling <bool>`: Skip CPU overhead
  - Default: True (faster simulation, slight accuracy loss)

**Linear Regression Predictor:**
- `--linear_regression_execution_time_predictor_config_polynomial_degree <int>`: Polynomial degree
  - Options: 1, 2, 3, 4, 5
- `--linear_regression_execution_time_predictor_config_polynomial_include_bias <bool>`
- `--linear_regression_execution_time_predictor_config_polynomial_interaction_only <bool>`
- `--linear_regression_execution_time_predictor_config_fit_intercept <bool>`

#### 8. Metrics Configuration (`metrics_config_*`)

**Output Settings:**
- `--metrics_config_output_dir <str>`: Output directory
  - Default: `simulator_output`

- `--metrics_config_write_metrics <bool>`: Enable metrics writing
  - Default: True

- `--metrics_config_enable_chrome_trace <bool>`: Generate Chrome trace
  - Default: True

- `--metrics_config_write_json_trace <bool>`: Write full event trace JSON
  - Default: False (large file size)

**Per-request/batch Metrics:**
- `--metrics_config_store_request_metrics <bool>`: Store per-request metrics
  - Default: True

- `--metrics_config_store_batch_metrics <bool>`: Store per-batch metrics
  - Default: True

**WandB Integration:**
- `--metrics_config_wandb_project <str>`: WandB project name
- `--metrics_config_wandb_group <str>`: WandB group name
- `--metrics_config_wandb_run_name <str>`: WandB run name

#### 9. Global Settings

- `--seed <int>`: Random seed
  - Default: 42

- `--log_level <str>`: Logging level
  - Options: `debug`, `info`, `warning`, `error`
  - Default: `info`

- `--time_limit <int>`: Simulation time limit in seconds
  - Default: 0 (no limit)

---

## Configuration System

### Hierarchy

```
SimulationConfig (root)
├── seed: int
├── log_level: str
├── time_limit: int
├── cluster_config: ClusterConfig
│   ├── num_replicas: int
│   ├── replica_config: ReplicaConfig
│   │   ├── model_name: str
│   │   ├── device: str
│   │   ├── network_device: str
│   │   ├── tensor_parallel_size: int
│   │   ├── num_pipeline_stages: int
│   │   └── memory_margin_fraction: float
│   ├── global_scheduler_config: BaseGlobalSchedulerConfig
│   └── replica_scheduler_config: BaseReplicaSchedulerConfig
├── request_generator_config: BaseRequestGeneratorConfig
├── execution_time_predictor_config: BaseExecutionTimePredictorConfig
└── metrics_config: MetricsConfig
```

### Supported Models

Pre-configured models with architecture details:

| Model | Identifier | Layers | Hidden Size | Attention |
|-------|------------|--------|-------------|-----------|
| Llama-2-7B | `meta-llama/Llama-2-7b-hf` | 32 | 4096 | MHA (32 heads) |
| Llama-2-70B | `meta-llama/Llama-2-70b-hf` | 80 | 8192 | GQA (64Q/8KV) |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B` | 32 | 4096 | GQA (32Q/8KV) |
| Llama-3-70B | `meta-llama/Meta-Llama-3-70B` | 80 | 8192 | GQA (64Q/8KV) |
| CodeLlama-34B | `codellama/CodeLlama-34b-Instruct-hf` | 48 | 8192 | GQA (64Q/8KV) |
| InternLM-20B | `internlm/internlm-20b` | 60 | 5120 | MHA (40 heads) |
| InternLM2-20B | `internlm/internlm2-20b` | 48 | 6144 | GQA (48Q/8KV) |
| Qwen-72B | `Qwen/Qwen-72B` | 80 | 8192 | MHA (64 heads) |
| Phi-2 | `microsoft/phi-2` | 32 | 2560 | MHA (32 heads) |

**Attention Mechanisms:**
- **MHA (Multi-Head Attention)**: Standard transformer attention
- **GQA (Grouped-Query Attention)**: Multiple query heads share KV heads (memory efficient)

### Supported Devices

| Device | Memory | FP16 TFLOPS | Memory BW | Notes |
|--------|--------|-------------|-----------|-------|
| A100 | 80 GB | 1935 | 312 GB/s | Well-profiled |
| H100 | 80 GB | ~3958 | 3350 GB/s | Requires profiling |
| A40 | 48 GB | ~149 | 696 GB/s | Requires profiling |

**Network Topologies:**
- `a100_dgx`: 8 GPUs fully connected via NVLink (600 GB/s)
- `a100_pairwise_nvlink`: 4 GPUs with pairwise NVLink
- `a40_pairwise_nvlink`: 8 GPUs with pairwise NVLink

---

## Running Vidur - Examples

### Example 1: Basic Single-GPU Simulation

```bash
python -m vidur.main \
  --replica_config_device a100 \
  --replica_config_model_name meta-llama/Llama-2-7b-hf \
  --replica_config_tensor_parallel_size 1 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 1000 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 512 \
  --fixed_request_length_generator_config_decode_tokens 128 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 5.0 \
  --replica_scheduler_config_type vllm \
  --vllm_scheduler_config_batch_size_cap 256
```

**What this does:**
- Simulates Llama-2-7B on single A100 GPU
- 1000 synthetic requests with Poisson arrivals (5 QPS)
- Fixed request size: 512 input tokens, 128 output tokens
- vLLM scheduler with max batch size 256

### Example 2: Trace-based Simulation with Sarathi

```bash
python -m vidur.main \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_device a100 \
  --replica_config_tensor_parallel_size 1 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 512 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --trace_request_length_generator_config_max_tokens 16384 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 6.45 \
  --replica_scheduler_config_type sarathi \
  --sarathi_scheduler_config_batch_size_cap 512 \
  --sarathi_scheduler_config_chunk_size 512 \
  --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384
```

**What this does:**
- Simulates Llama-3-8B on A100 with Sarathi scheduler
- Request lengths sampled from splitwise conversation trace
- Max context length: 16384 tokens
- Chunked prefill with 512-token chunks
- Important: Increased `prediction_max_prefill_chunk_size` to 16384 for long contexts

### Example 3: Multi-GPU with Tensor Parallelism

```bash
python -m vidur.main \
  --replica_config_model_name meta-llama/Llama-2-70b-hf \
  --replica_config_device a100 \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_network_device a100_pairwise_nvlink \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 500 \
  --length_generator_config_type uniform \
  --uniform_request_length_generator_config_min_tokens 512 \
  --uniform_request_length_generator_config_max_tokens 2048 \
  --uniform_request_length_generator_config_prefill_to_decode_ratio 0.5 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 2.0 \
  --replica_scheduler_config_type vllm \
  --vllm_scheduler_config_batch_size_cap 128
```

**What this does:**
- Simulates Llama-2-70B with TP=4 across 4 A100 GPUs
- Network communication via pairwise NVLink
- Uniform request size distribution (512-2048 tokens, 50% prefill)
- Lower QPS (2.0) suitable for large model

### Example 4: High QPS with Large Batch Sizes

```bash
python -m vidur.main \
  --replica_config_model_name meta-llama/Llama-2-7b-hf \
  --replica_config_device a100 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 2000 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 256 \
  --fixed_request_length_generator_config_decode_tokens 64 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 50.0 \
  --replica_scheduler_config_type vllm \
  --vllm_scheduler_config_batch_size_cap 512 \
  --vllm_scheduler_config_max_tokens_in_batch 32768 \
  --random_forrest_execution_time_predictor_config_prediction_max_batch_size 512
```

**What this does:**
- High QPS (50.0) stress test
- Large batch size cap (512 requests)
- Increased `max_tokens_in_batch` and `prediction_max_batch_size` for accuracy

### Example 5: Trace Replay with Full Configuration

```bash
python -m vidur.main \
  --replica_config_model_name codellama/CodeLlama-34b-Instruct-hf \
  --replica_config_device a100 \
  --replica_config_tensor_parallel_size 2 \
  --replica_config_network_device a100_pairwise_nvlink \
  --request_generator_config_type trace_replay \
  --trace_request_generator_config_trace_file ./data/processed_traces/code_generation_trace.csv \
  --trace_request_generator_config_prefill_scale_factor 1.0 \
  --trace_request_generator_config_decode_scale_factor 1.0 \
  --trace_request_generator_config_time_scale_factor 0.5 \
  --replica_scheduler_config_type sarathi \
  --sarathi_scheduler_config_chunk_size 1024 \
  --metrics_config_output_dir ./output_code \
  --metrics_config_enable_chrome_trace true
```

**What this does:**
- Full trace replay (timestamps + request sizes from CSV)
- Time scaled by 0.5x (2x arrival rate)
- CodeLlama-34B with TP=2
- Sarathi scheduler with 1024-token chunks
- Custom output directory with Chrome trace

---

## Output Format and Metrics

### Output Directory Structure

Default location: `simulator_output/YYYY-MM-DD_HH-MM-SS-ffffff/`

```
simulator_output/2026-01-26_10-30-45-123456/
├── config.json                    # Complete simulation configuration
├── plots/                         # CDF and histogram plots (PNG + CSV)
│   ├── request_e2e_time.png       # E2E latency CDF
│   ├── request_e2e_time.csv       # E2E latency data
│   ├── request_scheduling_delay.png
│   ├── prefill_e2e_time.png       # TTFT CDF
│   ├── decode_time_execution_plus_preemption_normalized.png  # TPOT/ITL CDF
│   ├── batch_size.png             # Batch size distribution
│   ├── batch_num_tokens.png       # Tokens per batch
│   └── ...
├── request_metrics.csv            # ⭐ Primary output: per-request metrics
├── batch_metrics.csv              # Per-batch metrics (if enabled)
├── operation_metrics.csv          # Per-operation metrics (if enabled)
├── cpu_operation_metrics.csv      # CPU overhead metrics (if enabled)
├── chrome_trace.json              # ⭐ Chrome trace for visualization
└── event_trace.json               # Full event log (if enabled, large file)
```

### Primary Output: request_metrics.csv

This is the **most important file** for SLO evaluation and metrics extraction.

#### Column Reference

**Request Identifiers:**
- `Request Id`: Unique integer identifier for each request

**E2E Latency Metrics:**
- `request_e2e_time`: **Total end-to-end latency** (arrival → completion) in seconds
  - This is what users experience
- `request_e2e_time_normalized`: E2E time per output token = `request_e2e_time / num_decode_tokens`
- `request_scheduling_delay`: **Queueing delay** (time waiting before first schedule) in seconds
- `request_execution_time`: Total GPU execution time (excludes queueing and preemption)
- `request_preemption_time`: Time spent preempted or waiting due to memory constraints

**TTFT (Time to First Token):**
- `prefill_e2e_time`: **TTFT** (time from arrival to first token) in seconds
  - Formula: `prefill_completed_at - arrived_at`
  - Includes queueing delay + prefill execution
- `prefill_time_execution_plus_preemption`: TTFT excluding queueing delay
  - Pure execution time for prefill phase

**TPOT/ITL (Time Per Output Token / Inter-Token Latency):**
- `decode_time_execution_plus_preemption_normalized`: **TPOT/ITL** in seconds per token
  - Formula: `(completed_at - prefill_completed_at) / num_decode_tokens`
  - **This is the key metric for streaming latency**
  - Lower is better (faster token generation)

**Request Characteristics:**
- `request_num_tokens`: Total tokens (prefill + decode)
- `request_num_prefill_tokens`: Input tokens
- `request_num_decode_tokens`: Output tokens (generated)
- `request_pd_ratio`: Prefill-to-decode ratio = `prefill_tokens / decode_tokens`
- `request_num_restarts`: Number of restarts/preemptions (vLLM specific)

**Timestamps:**
- `arrived_at`: Arrival timestamp (seconds since simulation start)
- `scheduled_at`: First scheduled timestamp
- `prefill_completed_at`: Timestamp when prefill completed
- `completed_at`: Final completion timestamp
- `request_inter_arrival_delay`: Time since previous request arrived

**Additional Metrics:**
- `replica_id`: Which replica served this request (multi-replica only)
- `prefill_chunks_execution_time`: List of prefill chunk execution times (Sarathi)
- `decode_iterations_execution_time`: List of decode iteration times

### Secondary Outputs

#### batch_metrics.csv

**When generated:** If `metrics_config_store_batch_metrics=True`

**Columns:**
- `Batch Id`: Unique batch identifier
- `batch_size`: Number of requests in batch
- `batch_num_tokens`: Total tokens processed in batch
- `batch_num_prefill_tokens`: Prefill tokens in batch
- `batch_num_decode_tokens`: Decode tokens in batch
- `batch_execution_time`: Time to execute batch (seconds)
- `batch_scheduled_time`: Timestamp when batch was scheduled
- `replica_id`: Replica that executed this batch

#### chrome_trace.json

**Purpose:** Visualize simulation execution timeline

**How to use:**
1. Open Chrome or Edge browser
2. Navigate to `chrome://tracing/` or `edge://tracing/`
3. Click "Load" and select `chrome_trace.json`
4. Interactive timeline visualization

**Shows:**
- Request lifecycle (arrival → prefill → decode → completion)
- Batch scheduling and execution
- GPU utilization timeline
- Preemption events
- Pipeline bubbles (PP)

**Useful for:**
- Debugging scheduling issues
- Understanding head-of-line blocking
- Analyzing preemption patterns
- Visualizing GPU idle time

### Plots Directory

All plots saved as **PNG images + CSV data** for easy analysis.

**CDF (Cumulative Distribution Function) Plots:**
- `request_e2e_time`: E2E latency distribution
- `prefill_e2e_time`: TTFT distribution
- `decode_time_execution_plus_preemption_normalized`: TPOT/ITL distribution
- `request_scheduling_delay`: Queueing delay distribution

**Histogram Plots:**
- `batch_size`: Batch size distribution
- `batch_num_tokens`: Tokens per batch
- `request_num_tokens`: Request size distribution

**Usage example:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CDF data
ttft_data = pd.read_csv("plots/prefill_e2e_time.csv")
plt.plot(ttft_data['x'], ttft_data['y'])
plt.xlabel("TTFT (seconds)")
plt.ylabel("CDF")
plt.title("TTFT Distribution")
plt.show()
```

---

## Parsing Metrics (TTFT, ITL, E2E)

### Python Script for Metric Extraction

```python
import pandas as pd
import glob
import os

def load_latest_metrics(base_dir="simulator_output"):
    """Load metrics from latest simulation output."""
    output_dirs = glob.glob(f"{base_dir}/*")
    latest_dir = max(output_dirs, key=os.path.getctime)

    df = pd.read_csv(f"{latest_dir}/request_metrics.csv")
    return df, latest_dir

def calculate_slo_metrics(df):
    """Calculate all key SLO metrics."""
    metrics = {
        # TTFT (Time to First Token)
        "TTFT_P50_sec": df['prefill_e2e_time'].quantile(0.50),
        "TTFT_P90_sec": df['prefill_e2e_time'].quantile(0.90),
        "TTFT_P95_sec": df['prefill_e2e_time'].quantile(0.95),
        "TTFT_P99_sec": df['prefill_e2e_time'].quantile(0.99),
        "TTFT_mean_sec": df['prefill_e2e_time'].mean(),

        # TPOT/ITL (Time Per Output Token / Inter-Token Latency)
        "TPOT_P50_sec": df['decode_time_execution_plus_preemption_normalized'].quantile(0.50),
        "TPOT_P90_sec": df['decode_time_execution_plus_preemption_normalized'].quantile(0.90),
        "TPOT_P95_sec": df['decode_time_execution_plus_preemption_normalized'].quantile(0.95),
        "TPOT_P99_sec": df['decode_time_execution_plus_preemption_normalized'].quantile(0.99),
        "TPOT_mean_sec": df['decode_time_execution_plus_preemption_normalized'].mean(),

        # E2E Latency (End-to-End)
        "E2E_P50_sec": df['request_e2e_time'].quantile(0.50),
        "E2E_P90_sec": df['request_e2e_time'].quantile(0.90),
        "E2E_P95_sec": df['request_e2e_time'].quantile(0.95),
        "E2E_P99_sec": df['request_e2e_time'].quantile(0.99),
        "E2E_mean_sec": df['request_e2e_time'].mean(),

        # Scheduling Delay (Queueing Delay)
        "SchedDelay_P50_sec": df['request_scheduling_delay'].quantile(0.50),
        "SchedDelay_P90_sec": df['request_scheduling_delay'].quantile(0.90),
        "SchedDelay_P95_sec": df['request_scheduling_delay'].quantile(0.95),
        "SchedDelay_P99_sec": df['request_scheduling_delay'].quantile(0.99),
        "SchedDelay_mean_sec": df['request_scheduling_delay'].mean(),

        # Throughput
        "total_requests": len(df),
        "total_duration_sec": df['completed_at'].max() - df['arrived_at'].min(),
    }

    # Calculate throughput (requests per second)
    metrics["throughput_rps"] = metrics["total_requests"] / metrics["total_duration_sec"]

    # Calculate token throughput
    total_tokens = df['request_num_tokens'].sum()
    metrics["token_throughput_tps"] = total_tokens / metrics["total_duration_sec"]

    return metrics

def check_slo_compliance(metrics, slo_config):
    """Check if SLOs are met.

    Example slo_config:
    {
        "TTFT_P90_sec": 1.0,
        "TPOT_P99_sec": 0.2,
        "E2E_P95_sec": 5.0,
    }
    """
    results = {}
    for metric_name, threshold in slo_config.items():
        if metric_name in metrics:
            actual = metrics[metric_name]
            met = actual <= threshold
            results[metric_name] = {
                "threshold": threshold,
                "actual": actual,
                "met": met,
                "margin": threshold - actual,
            }
    return results

# Usage example
df, output_dir = load_latest_metrics()
metrics = calculate_slo_metrics(df)

# Print all metrics
print("=== Simulation Metrics ===")
print(f"Output Directory: {output_dir}")
print()
print("TTFT (Time to First Token):")
print(f"  P50: {metrics['TTFT_P50_sec']*1000:.2f} ms")
print(f"  P90: {metrics['TTFT_P90_sec']*1000:.2f} ms")
print(f"  P95: {metrics['TTFT_P95_sec']*1000:.2f} ms")
print(f"  P99: {metrics['TTFT_P99_sec']*1000:.2f} ms")
print(f"  Mean: {metrics['TTFT_mean_sec']*1000:.2f} ms")
print()
print("TPOT/ITL (Time Per Output Token):")
print(f"  P50: {metrics['TPOT_P50_sec']*1000:.2f} ms/token")
print(f"  P90: {metrics['TPOT_P90_sec']*1000:.2f} ms/token")
print(f"  P95: {metrics['TPOT_P95_sec']*1000:.2f} ms/token")
print(f"  P99: {metrics['TPOT_P99_sec']*1000:.2f} ms/token")
print(f"  Mean: {metrics['TPOT_mean_sec']*1000:.2f} ms/token")
print()
print("E2E Latency:")
print(f"  P50: {metrics['E2E_P50_sec']:.3f} sec")
print(f"  P90: {metrics['E2E_P90_sec']:.3f} sec")
print(f"  P95: {metrics['E2E_P95_sec']:.3f} sec")
print(f"  P99: {metrics['E2E_P99_sec']:.3f} sec")
print(f"  Mean: {metrics['E2E_mean_sec']:.3f} sec")
print()
print("Throughput:")
print(f"  Requests/sec: {metrics['throughput_rps']:.2f}")
print(f"  Tokens/sec: {metrics['token_throughput_tps']:.2f}")
print()

# Check SLO compliance
slo_config = {
    "TTFT_P90_sec": 1.0,      # 1 second P90 TTFT
    "TPOT_P99_sec": 0.2,      # 200ms P99 TPOT
    "E2E_P95_sec": 5.0,       # 5 second P95 E2E
}

slo_results = check_slo_compliance(metrics, slo_config)
print("=== SLO Compliance ===")
for metric_name, result in slo_results.items():
    status = "✓ PASS" if result["met"] else "✗ FAIL"
    print(f"{metric_name}: {status}")
    print(f"  Threshold: {result['threshold']}")
    print(f"  Actual: {result['actual']:.6f}")
    print(f"  Margin: {result['margin']:.6f}")
```

### Converting to Milliseconds

Most SLOs are expressed in milliseconds. Convert seconds to ms:

```python
# TTFT in milliseconds
ttft_p90_ms = df['prefill_e2e_time'].quantile(0.90) * 1000

# TPOT in milliseconds per token
tpot_p99_ms = df['decode_time_execution_plus_preemption_normalized'].quantile(0.99) * 1000

# E2E in milliseconds
e2e_p95_ms = df['request_e2e_time'].quantile(0.95) * 1000
```

### Key Metrics Summary Table

| Metric | Column Name | Unit | Interpretation |
|--------|-------------|------|----------------|
| **TTFT** | `prefill_e2e_time` | seconds | Time from request arrival to first token. Lower is better. Target: <1s P90 |
| **TPOT/ITL** | `decode_time_execution_plus_preemption_normalized` | sec/token | Time per output token during streaming. Lower is better. Target: <200ms P99 |
| **E2E Latency** | `request_e2e_time` | seconds | Total latency from arrival to completion. Lower is better. |
| **Scheduling Delay** | `request_scheduling_delay` | seconds | Queueing delay before first schedule. Indicates load. Lower is better. |
| **Throughput** | Calculated from `completed_at` | requests/sec | System throughput. Higher is better. |

---

## Request Generation

### Overview

Vidur supports two main request generation strategies:
1. **Synthetic**: Generate requests with configurable distributions
2. **Trace Replay**: Replay real production traces

### Synthetic Request Generator

**Configuration:**
```bash
--request_generator_config_type synthetic
--synthetic_request_generator_config_num_requests <int>
```

**Components:**
- **Length Generator**: Controls request size distribution
- **Interval Generator**: Controls request arrival pattern

#### Length Generators

**1. Trace Length Generator (Recommended for Realism)**

Samples request lengths from real trace distributions.

```bash
--length_generator_config_type trace
--trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv
--trace_request_length_generator_config_max_tokens 16384
--trace_request_length_generator_config_prefill_scale_factor 1.0
--trace_request_length_generator_config_decode_scale_factor 1.0
```

**Trace CSV format:**
```csv
num_input_tokens,num_output_tokens
512,128
1024,256
...
```

**Available traces:**
- `splitwise_conv.csv`: Chat conversations
- `splitwise_code.csv`: Code generation
- `arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv`: Document summarization

**2. Fixed Length Generator (Simple, Deterministic)**

All requests have same size.

```bash
--length_generator_config_type fixed
--fixed_request_length_generator_config_prefill_tokens 512
--fixed_request_length_generator_config_decode_tokens 128
```

**3. Uniform Length Generator**

Uniform random distribution.

```bash
--length_generator_config_type uniform
--uniform_request_length_generator_config_min_tokens 128
--uniform_request_length_generator_config_max_tokens 2048
--uniform_request_length_generator_config_prefill_to_decode_ratio 0.5
```

**4. Zipf Length Generator (Heavy-tailed)**

Zipf distribution for skewed workloads.

```bash
--length_generator_config_type zipf
--zipf_request_length_generator_config_theta 1.0
--zipf_request_length_generator_config_min_tokens 128
--zipf_request_length_generator_config_max_tokens 4096
--zipf_request_length_generator_config_prefill_to_decode_ratio 0.3
```

#### Interval Generators

**1. Poisson Interval Generator (Recommended for Realistic Load)**

Exponentially distributed inter-arrival times (memoryless).

```bash
--interval_generator_config_type poisson
--poisson_request_interval_generator_config_qps 10.0
```

**Characteristics:**
- Independent arrivals
- No correlation between requests
- Models random user arrivals

**2. Gamma Interval Generator (Bursty Traffic)**

Gamma distribution with configurable coefficient of variation.

```bash
--interval_generator_config_type gamma
--gamma_request_interval_generator_config_qps 10.0
--gamma_request_interval_generator_config_cv 2.0
```

**CV (Coefficient of Variation):**
- `cv=1.0`: Equivalent to Poisson
- `cv>1.0`: More bursty (higher variance)
- `cv<1.0`: More regular (lower variance)

**3. Static Interval Generator (Batch Processing)**

All requests arrive at t=0.

```bash
--interval_generator_config_type static
```

**Use case:** Testing maximum batch capacity.

### Trace Replay Generator

**Configuration:**
```bash
--request_generator_config_type trace_replay
--trace_request_generator_config_trace_file ./data/processed_traces/full_trace.csv
--trace_request_generator_config_prefill_scale_factor 1.0
--trace_request_generator_config_decode_scale_factor 1.0
--trace_request_generator_config_time_scale_factor 1.0
```

**Trace CSV format:**
```csv
timestamp,num_input_tokens,num_output_tokens
0.0,512,128
0.25,1024,256
0.50,256,64
...
```

**Scale factors:**
- `prefill_scale_factor`: Multiply all input tokens (test larger contexts)
- `decode_scale_factor`: Multiply all output tokens (test longer generations)
- `time_scale_factor`: Multiply all timestamps (speed up/slow down arrivals)
  - `time_scale_factor=0.5`: 2x arrival rate
  - `time_scale_factor=2.0`: 0.5x arrival rate

---

## Scheduling Policies

### Overview

Vidur supports 5 scheduling policies, each with different trade-offs.

| Scheduler | Batching | Preemption | Chunked Prefill | Best For |
|-----------|----------|------------|-----------------|----------|
| vLLM | Dynamic | Yes (swap) | No | General purpose, high throughput |
| Sarathi | Dynamic | Yes | Yes | Low TPOT, streaming |
| LightLLM | Dynamic | Yes | No | General purpose |
| Orca | Selective | Limited | No | Research baseline |
| FasterTransformer | Static | No | No | Fixed batch sizes |

### 1. vLLM Scheduler

**Type:** Dynamic batching with iteration-level scheduling

**Key Features:**
- KV cache swapping when memory full
- Preemption with restart
- Greedy FCFS batching

**Configuration:**
```bash
--replica_scheduler_config_type vllm
--vllm_scheduler_config_batch_size_cap 256
--vllm_scheduler_config_max_tokens_in_batch 16384
--vllm_scheduler_config_block_size 16
--vllm_scheduler_config_num_blocks 140000  # Optional, auto-calculated
--vllm_scheduler_config_watermark_blocks_fraction 0.01
```

**Scheduling Algorithm:**
1. Each iteration: try to add new requests from waiting queue
2. Add requests until `batch_size_cap` or `max_tokens_in_batch` reached
3. If memory insufficient, preempt lowest-priority running requests
4. Preempted requests swapped to CPU (if space) or restarted

**When to use:**
- General-purpose serving
- High throughput workloads
- Variable request sizes

**Trade-offs:**
- Pro: High throughput, good memory utilization
- Con: Prefill can block decode (head-of-line blocking), variable TPOT

### 2. Sarathi Scheduler

**Type:** Chunked prefill with dynamic batching

**Key Features:**
- Splits large prefills into chunks
- Batches prefill chunks with decode requests
- Reduces prefill interference with decode

**Configuration:**
```bash
--replica_scheduler_config_type sarathi
--sarathi_scheduler_config_batch_size_cap 512
--sarathi_scheduler_config_chunk_size 512
--sarathi_scheduler_config_block_size 16
```

**Chunk Size Trade-offs:**

| Chunk Size | TTFT | TPOT | Throughput | Use Case |
|------------|------|------|------------|----------|
| 256 | Higher | Best | Lower | Streaming focus (e.g., chat) |
| 512 | Medium | Good | Medium | Balanced |
| 1024 | Medium | Medium | Good | Balanced |
| 2048 | Lower | Medium | Better | Throughput focus |
| 4096 | Lowest | Worse | Best | Offline batch |

**Scheduling Algorithm:**
1. Split prefills into chunks of `chunk_size` tokens
2. Treat each prefill chunk as a decode request
3. Batch prefill chunks with decode requests uniformly

**When to use:**
- Streaming applications (e.g., chat)
- SLOs focused on TPOT/ITL
- Mix of short and long prompts

**Trade-offs:**
- Pro: Lower TPOT variance, better streaming experience
- Con: Slightly higher TTFT, more complex scheduling

### 3. LightLLM Scheduler

**Type:** Dynamic batching with enhanced preemption

**Configuration:**
```bash
--replica_scheduler_config_type lightllm
--lightllm_scheduler_config_batch_size_cap 256
--lightllm_scheduler_config_max_tokens_in_batch 16384
--lightllm_scheduler_config_max_waiting_iters 10
```

**Difference from vLLM:**
- More aggressive preemption
- `max_waiting_iters`: Preempt if request waits too long

**When to use:**
- Similar to vLLM
- When strict latency SLOs required

### 4. Orca Scheduler

**Type:** Selective batching

**Configuration:**
```bash
--replica_scheduler_config_type orca
--orca_scheduler_config_batch_size_cap 128
```

**When to use:**
- Research baseline for comparison
- Understanding impact of selective batching

### 5. FasterTransformer Scheduler

**Type:** Static batching (no preemption)

**Configuration:**
```bash
--replica_scheduler_config_type faster_transformer
--faster_transformer_scheduler_config_batch_size_cap 64
```

**When to use:**
- Offline batch processing
- Fixed batch sizes
- Baseline comparisons

---

## Execution Time Predictor

### Overview

Vidur predicts GPU execution time using **machine learning models trained on profiling data**. This enables simulation without running on real GPUs.

### Two-Phase Approach

**Phase 1: Profiling (One-time, Requires GPUs)**
- Profile MLP operations (linear layers)
- Profile attention operations (self-attention kernels)
- Profile network collectives (AllReduce for TP)
- Profile CPU overheads (Python + CUDA launch)
- Save profiling data to CSVs

**Phase 2: Prediction (Simulation, No GPUs)**
- Load profiling CSVs
- Train ML model (Random Forest or Linear Regression)
- Predict batch execution time during simulation

### Random Forest Predictor (Recommended)

**Configuration:**
```bash
--execution_time_predictor_config_type random_forrest
--random_forrest_execution_time_predictor_config_compute_input_file data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/mlp.csv
--random_forrest_execution_time_predictor_config_attention_input_file data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 4096
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 128
--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling true
```

**Important Parameters:**

**1. `prediction_max_prefill_chunk_size`**
- Default: 4096
- **For Llama-3 or long contexts:** Use 16384
- Determines max chunk size for prefill prediction
- Must cover largest prefill chunks in simulation

**2. `prediction_max_batch_size`**
- Default: 128
- **For large batch sizes:** Use 512
- Determines max batch size for prediction
- Must cover largest batches in simulation

**3. `skip_cpu_overhead_modeling`**
- Default: True
- Set False for highest accuracy (slower simulation)
- CPU overhead typically <5% of GPU time

### ML Model Details

**Features used for prediction:**
- Batch size
- Number of prefill tokens
- Number of decode tokens
- KV cache size (tokens)
- Tensor parallel size
- Pipeline parallel size

**Training:**
- Scikit-learn RandomForestRegressor
- Grid search over hyperparameters:
  - `num_estimators`: [250, 500, 750]
  - `max_depth`: [8, 16, 32]
  - `min_samples_split`: [2, 5, 10]

**Accuracy:**
- Typically 5-10% error on unseen workloads
- Higher accuracy for interpolation (within profiled range)
- Lower accuracy for extrapolation (outside profiled range)

### Profiling Process

**Prerequisites:**
- Access to target GPU (A100, H100, etc.)
- Model weights loaded

**Steps:**

**1. Profile MLP operations:**
```bash
cd vidur/profiling/mlp
python main.py \
  --model-name meta-llama/Llama-2-7b-hf \
  --device a100 \
  --output-dir ../../data/profiling/compute/a100/meta-llama/Llama-2-7b-hf
```

**2. Profile attention operations:**
```bash
cd vidur/profiling/attention
python main.py \
  --model-name meta-llama/Llama-2-7b-hf \
  --device a100 \
  --output-dir ../../data/profiling/compute/a100/meta-llama/Llama-2-7b-hf
```

**3. Profile network collectives (if TP>1):**
```bash
cd vidur/profiling/collectives
python main.py \
  --network-device a100_pairwise_nvlink \
  --output-dir ../../data/profiling/network/a100_pairwise_nvlink
```

**4. Profile CPU overheads (optional):**
```bash
cd vidur/profiling/cpu_overhead
python main.py \
  --output-dir ../../data/profiling/cpu_overhead
```

**Output:** CSV files with execution time measurements for different batch sizes, token counts, etc.

### Adding New Models/Devices

See `docs/profiling.md` for detailed instructions on:
- Setting up profiling environment
- Running profiling scripts with custom parameters
- Validating profiling data quality

---

## Config Explorer (Capacity Planning)

### Overview

**Config Explorer** is Vidur's automated capacity planning tool that finds optimal configurations for given SLO constraints.

**Goal:** Find the configuration that maximizes throughput (QPS) while meeting SLO thresholds.

### How It Works

**Algorithm:** Binary search over QPS for each configuration

```
For each config in search space:
  1. Binary search to find max QPS:
     - Start with QPS range [low, high]
     - Run simulation at mid QPS
     - If SLO met: increase QPS (search higher)
     - If SLO violated: decrease QPS (search lower)
  2. Record max QPS and metrics

Select config with highest max QPS
```

### Running Config Explorer

**Command:**
```bash
python -m vidur.config_optimizer.config_explorer.main \
  --config-path ./config.yaml \
  --output-dir ./output \
  --scheduling-delay-slo-value 5.0 \
  --scheduling-delay-slo-quantile 0.99 \
  --max-iterations 20 \
  --min-search-granularity 2.5 \
  --time-limit 30 \
  --num-threads 8
```

### CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config-path` | str | Required | YAML config file defining search space |
| `--output-dir` | str | Required | Output directory for results |
| `--scheduling-delay-slo-value` | float | 5.0 | SLO threshold in seconds (P99 scheduling delay) |
| `--scheduling-delay-slo-quantile` | float | 0.99 | SLO percentile (0.90, 0.95, 0.99) |
| `--max-iterations` | int | 20 | Max binary search iterations |
| `--min-search-granularity` | float | 2.5 | Stop when within X% of optimal |
| `--time-limit` | int | 30 | Simulation time limit (minutes) |
| `--num-threads` | int | cpu_count-2 | Parallel workers |
| `--cache-dir` | str | ./cache_tmpfs | Cache directory |
| `--skip-cache-warmup` | bool | False | Skip initial cache warmup |

### Config YAML Format

**Example:** `vidur/config_optimizer/config_explorer/config/config.yml`

```yaml
# Hardware clusters to explore
clusters:
  - device: h100
    num_gpus: 16
    gpus_per_node: 4
  - device: a100
    num_gpus: 16
    gpus_per_node: 4

# Scheduling policies
schedulers:
  - scheduler: vllm
  - scheduler: sarathi
    chunk_size: 512
  - scheduler: sarathi
    chunk_size: 2048

# Workload traces
traces:
  - name: chat
    trace_file: "./data/processed_traces/lmsys_chat_1m_conversation_stats_llama2_tokenizer.csv"
    max_seq_len: 4096
    num_requests: 16000
    start_qps: 32
  - name: arxiv
    trace_file: "./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv"
    max_seq_len: 4096
    num_requests: 16000
    start_qps: 16

# Search space parameters
batch_sizes: [32, 64, 128]
tp_dimensions: [1, 2, 4, 8]
pp_dimensions: [1, 2, 4]

# Models to evaluate
models:
  - name: llama-2-7b-hf
    identifier: meta-llama/Llama-2-7b-hf
  - name: codellama-34b-instruct-hf
    identifier: codellama/CodeLlama-34b-Instruct-hf
  - name: qwen-72b
    identifier: Qwen/Qwen-72B
    exclude_tp_dims: [1]  # Skip TP=1 for large models (OOM)
```

### Search Space

Config Explorer generates **Cartesian product** of all parameter combinations:

**Total configs = |clusters| × |schedulers| × |traces| × |models| × |batch_sizes| × |tp_dimensions| × |pp_dimensions|**

**Example:**
- 2 clusters × 3 schedulers × 2 traces × 3 models × 3 batch sizes × 4 TP dims × 3 PP dims
- = **1,296 configurations**

**Runtime:** With 8 parallel workers, ~10-20 minutes per config → ~27-54 hours total

**Optimization:** Use caching and start with smaller search space.

### Output Format

**Directory structure:**
```
output/
├── args.json                      # CLI arguments
├── config.json                    # Config YAML as JSON
├── llama-2-7b_chat_vllm/          # Per-config subdirectories
│   ├── qps_1.0/
│   │   └── simulator_output/      # Full Vidur output
│   │       ├── request_metrics.csv
│   │       ├── chrome_trace.json
│   │       └── ...
│   ├── qps_2.0/
│   ├── qps_4.0/
│   └── ...
├── llama-2-7b_chat_sarathi/
└── ...
```

### Analyzing Results

Config Explorer output is analyzed using the **Analyzer Dashboard** (see next section).

---

## Analyzer Dashboard

### Overview

**Analyzer Dashboard** is a Streamlit web application for visualizing and comparing Config Explorer results.

### Running Dashboard

```bash
python -m streamlit run vidur/config_optimizer/analyzer/dashboard/main.py -- \
  --sim-results-dir ./output \
  --scheduling-delay-slo-percentile 95 \
  --scheduling-delay-slo-value 2.0
```

**Parameters:**
- `--sim-results-dir`: Path to Config Explorer output directory
- `--scheduling-delay-slo-percentile`: SLO percentile (90, 95, 99)
- `--scheduling-delay-slo-value`: SLO threshold in seconds

**Access:** Open browser to `http://localhost:8501`

### Dashboard Pages

#### 1. Best Config

**Purpose:** Find best configuration for each model-trace pair

**Features:**
- Table with best config per model-trace
- Columns: Model, Trace, Scheduler, TP, PP, Batch Size, Max QPS, TTFT P90, TBT P99, E2E P95

**Use case:** Quick answer to "What's the best config for Llama-2-7B on chat workload?"

#### 2. Pareto Curves

**Purpose:** Visualize trade-offs between metrics

**Features:**
- Interactive scatter plots:
  - TTFT vs QPS
  - TBT vs QPS
  - E2E vs QPS
- Pareto frontier highlighting
- Hover for config details

**Use case:** Understand trade-offs between latency and throughput

#### 3. Cost Analysis

**Purpose:** Calculate QPS per dollar for different hardware

**Features:**
- QPS per dollar metric
- Cost breakdown by hardware (A100, H100)
- Best value configurations

**Use case:** Budget-constrained capacity planning

#### 4. Config Compare

**Purpose:** Side-by-side comparison of specific configs

**Features:**
- Select multiple configs
- Compare all metrics (TTFT, TBT, E2E, throughput)
- CDF plots overlaid

**Use case:** Deep dive into why one config outperforms another

#### 5. Search Analysis

**Purpose:** Understand binary search convergence

**Features:**
- QPS convergence plots
- Iteration count per config
- SLO violation visualization

**Use case:** Debugging Config Explorer behavior

### Key Metrics in Dashboard

| Metric | Column Name | Interpretation |
|--------|-------------|----------------|
| **TTFT** | P50/P90/P95/P99 | Time to first token (seconds) |
| **TBT** | P50/P90/P95/P99 | Time between tokens = TPOT/ITL (seconds/token) |
| **E2E** | P50/P90/P95/P99 | End-to-end latency (seconds) |
| **QPS** | Max QPS | Maximum throughput meeting SLO |
| **SchedulingDelay** | P50/P90/P95/P99 | Queueing delay (seconds) |

---

## Summary

### When to Use Vidur

**Best for:**
- Research on scheduling algorithms
- Multi-GPU cluster capacity planning
- Comparing scheduler policies (vLLM vs Sarathi vs others)
- Long-term infrastructure planning
- Understanding tensor/pipeline parallelism trade-offs

**Not ideal for:**
- Quick single-GPU capacity planning (use BLIS instead - faster)
- Real-time inference (simulation only)
- Fine-grained kernel optimization (high-level simulation)

### Key Takeaways

1. **Input:** CLI arguments configure model, hardware, workload, scheduler
2. **Output:** `request_metrics.csv` with TTFT, TPOT/ITL, E2E latency per request
3. **Metrics:** Extract percentiles (P90, P95, P99) for SLO evaluation
4. **Config Explorer:** Automated capacity planning via binary search
5. **Dashboard:** Visualize results with Pareto curves and comparisons

### Integration with Config Search Tool

**Vidur Wrapper (Planned):**
- Generate Vidur YAML configs from config search tool
- Invoke Config Explorer programmatically
- Parse `request_metrics.csv` for metrics
- Return best config to unified interface

**See:** `config_search/vidur_wrapper.py` (Step 4)

---

## References

- **Vidur Paper:** "Vidur: A High-Fidelity Simulator for LLM Inference Serving" (Microsoft Research)
- **Sarathi Scheduler:** "Efficient Serving of Large Language Models via Chunked Prefill"
- **vLLM:** "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- **Repository:** https://github.com/microsoft/vidur (assumed, check for actual URL)

---

## Appendix: Common Issues

### 1. Profiling Data Not Found

**Error:**
```
FileNotFoundError: data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/mlp.csv
```

**Solution:**
- Run profiling scripts for your model/device
- Or use pre-profiled models (Llama-2-7B, Llama-2-70B on A100)

### 2. Prediction Out of Bounds

**Error:**
```
WARNING: Batch size 512 exceeds prediction_max_batch_size=128
```

**Solution:**
```bash
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 512
```

### 3. Long Context Issues

**Error:**
```
WARNING: Prefill chunk 16384 exceeds prediction_max_prefill_chunk_size=4096
```

**Solution:**
```bash
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384
```

### 4. WandB Login Required

**Error:**
```
wandb.errors.UsageError: api_key not configured
```

**Solution:**
```bash
export WANDB_MODE=disabled
```

### 5. Memory Overflow

**Error:**
```
ERROR: KV cache memory exceeds GPU capacity
```

**Solution:**
- Reduce `batch_size_cap`
- Reduce `max_tokens_in_batch`
- Increase `memory_margin_fraction`
- Use larger `tensor_parallel_size`

---

**End of Vidur Summary**
