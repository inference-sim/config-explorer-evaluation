# Saturation Validation: Implementation Complete

## Overview

Validates simulation results (BLIS/Vidur) against real vLLM performance using GuideLLM for load generation.

**Input:** Simulation results file (e.g., `lowprefix_results.json`) containing:
- Complete workload configuration
- Max QPS from BLIS and/or Vidur
- SLO constraints

**Output:** vLLM logs, GuideLLM logs, and benchmark results (JSON/HTML/CSV)

---

## Usage

### Mode 1: Use Existing Deployment

```bash
python saturation_orchestrator.py \
  --results lowprefix_results.json \
  --config-name test_config_lowprefix.json \
  --simulator blis \
  --output-dir validation_results \
  --deployment-name vllm-server \
  --namespace diya
```

### Mode 2: Create New Deployment

```bash
python saturation_orchestrator.py \
  --results lowprefix_results.json \
  --config-name test_config_lowprefix.json \
  --simulator blis \
  --output-dir validation_results \
  --use-k8s \
  --namespace diya \
  --image vllm/vllm-openai:v0.14.0
```

---

## Implementation

### Two Execution Modes

**Existing Deployment Mode (`--deployment-name`):**
- Finds pod from deployment name
- Installs dependencies (pip, guidellm)
- Starts vLLM server in pod
- Runs GuideLLM benchmark
- Downloads results via base64 encoding
- Deployment/pod remains running

**New Deployment Mode (`--use-k8s`):**
- Creates K8s deployment with H100 GPU affinity
- Waits for pod to be running
- Installs dependencies
- Starts vLLM server
- Runs GuideLLM benchmark
- Downloads results via base64 encoding
- Optionally cleans up deployment

---

## Key Implementation Details

### KubernetesManager Class

Core methods:
- `create_deployment()` - Creates K8s deployment with H100 GPU node affinity
- `find_pod_from_deployment()` - Finds pod from deployment label selector
- `wait_for_pod_ready()` - Polls until pod phase is Running
- `exec_command(cmd, background=False)` - Executes commands in pod (supports nohup for background)
- `read_file_from_pod(pod_path)` - Reads file using base64 encoding to avoid truncation
- `list_files_from_pod(pod_dir)` - Lists files in directory
- `delete_deployment()` - Deletes deployment and pod

### Helper Functions

Key helper functions:
- `build_vllm_command_string()` - Builds vLLM CLI command from config
- `build_guidellm_command_string()` - Builds GuideLLM CLI command from config
- `wait_for_vllm_in_pod()` - Polls /health and /v1/models endpoints via curl
- `wait_for_guidellm_completion()` - Polls for JSON/CSV files in output directory

### File Transfer

Uses base64 encoding to avoid truncation:
- `read_file_from_pod()` - Reads file via `base64 -w 0` then decodes locally
- `list_files_from_pod()` - Lists files via `find` command

---

## Parameter Mapping

| Config Field | vLLM Argument | Example |
|--------------|---------------|---------|
| `model` | model name | `codellama/CodeLlama-34b-Instruct-hf` |
| `batch_size` | `--max-num-seqs` | 256 |
| `max_scheduled_tokens` | `--max-num-batched-tokens` | 4096 |
| `max_model_len` | `--max-model-len` | 8192 |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | 0.9 |
| `block_size` | `--block-size` | 16 |
| `tp` | `--tensor-parallel-size` | 1 |
