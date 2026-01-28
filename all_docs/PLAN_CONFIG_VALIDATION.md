# Config Validator Implementation Plan

## Overview

**Purpose:** Answer two key questions for top-N simulator configs:
1. Do they meet SLOs on real vLLM?
2. How far off are simulator predictions vs real metrics?

**Key Requirement:** Fresh vLLM restart between configs (startup parameters cannot be changed dynamically).

**Cleanup:** Per-config result folders are deleted after extracting metrics (only final report kept).

| Component | Value |
|-----------|-------|
| **Script** | `config_validator.py` (~500 lines) |
| **Input** | BLIS/Vidur results JSON from `parallel_search.py` |
| **Output** | Single `validation_report.json` (temp folders deleted) |
| **Report Content** | SLO compliance + simulator accuracy only |
| **GPU Requirements** | max(TP) GPUs from all configs |
| **Reuse** | `KubernetesManager` + helpers from `saturation_orchestrator.py` |

---

## Input File Structure

Results from `parallel_search.py`:

```json
{
  "metadata": {
    "simulator": "BLIS",
    "model": "codellama/CodeLlama-34b-Instruct-hf",
    "hardware": "H100",
    "slo_constraints": [{"metric": "e2e_p95_ms", "threshold_ms": 1000}],
    "workload": {
      "num_requests": 100,
      "prefix_tokens": 129,
      "prompt_tokens": 2871,
      "prompt_tokens_stdev": 945,
      ...
    }
  },
  "successful_configs": [
    {
      "rank": 1,
      "config_id": 15,
      "max_qps": 7.27,
      "configuration": {
        "tp": 2,
        "batch_size": 128,
        "max_scheduled_tokens": 4096,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
        "block_size": 16
      },
      "slo_metrics": {
        "e2e_p95_ms": {"value_ms": 962.44, "threshold_ms": 1000, "passes": true}
      }
    }
  ]
}
```

---

## Implementation

### Core Functions

**Load and select configs:**
```python
def load_simulator_results(results_file):
    """Load JSON file"""

def get_top_n_configs(results_data, top_n):
    """Return first N from successful_configs"""

def build_validation_config(sim_config, metadata):
    """Merge sim_config.configuration + metadata.workload"""
```

**Validate single config:**
```python
def validate_single_config(sim_config, config, max_qps, slo_constraints, args, k8s, temp_dir):
    """
    10-step validation:
    1. Kill existing vLLM: pkill -f 'vllm serve'
    2. Wait 10s
    3. Start NEW vLLM with config's parameters
    4. Wait for health check
    5. Run GuideLLM at simulator's max_qps
    6. Wait for completion
    7. Download logs + benchmarks.json to temp_dir
    8. Parse metrics, compare simulator vs real
    9. Delete temp_dir (cleanup)
    10. Return result with SLO compliance + simulator accuracy
    """
```

**Parse GuideLLM results:**
```python
def extract_metric_from_guidellm(guidellm_results, metric_name):
    """
    Extract metrics from benchmarks[0].metrics:
    - request_latency.total.percentiles.p95 (seconds → × 1000 for ms)
    - time_to_first_token_ms.total.percentiles.p90 (already ms)
    - inter_token_latency_ms.total.percentiles.p95 (already ms)
    """
```

**GPU allocation:**
```python
def determine_max_tp(configs):
    """Scan configs and return max TP value"""
    return max(cfg['configuration']['tp'] for cfg in configs)
```

**Generate report:**
```python
def generate_validation_report(all_results, output_dir):
    """
    Create validation_report.json with:
    - Summary: configs meeting/failing SLOs
    - Per-config: SLO compliance + simulator accuracy (error_percent)

    error_percent = ((real_ms - simulator_ms) / simulator_ms) * 100
    """

def cleanup_temp_dir(temp_dir):
    """Delete temporary per-config directory after metrics extracted"""
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
```

### Main Loop

```python
def main():
    # Parse args
    # Determine max TP across all configs
    # Setup K8s manager (create or connect to deployment)
    # For each simulator (BLIS, Vidur):
    #   - Load results file
    #   - Get top N configs
    #   - For each config:
    #       - Create temp_dir for this config
    #       - Validate against real vLLM at simulator's predicted QPS
    #       - Check if SLOs are met
    #       - Calculate simulator accuracy (error_percent)
    #       - Delete temp_dir (cleanup)
    #       - Collect result for report
    # Generate validation report (SLO compliance + accuracy only)
    # Cleanup K8s deployment (if --use-k8s and not --keep-deployment)
```

---

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--blis-results` | One of BLIS/Vidur | - | BLIS results JSON |
| `--vidur-results` | One of BLIS/Vidur | - | Vidur results JSON |
| `--top-n` | Yes | - | Number of top configs to validate |
| `--deployment-name` | One of deploy/use-k8s | - | Existing deployment name |
| `--use-k8s` | One of deploy/use-k8s | False | Create new deployment |
| `--namespace` | No | diya | K8s namespace |
| `--image` | No | vllm/vllm-openai:v0.14.0 | vLLM image |
| `--output-dir` | Yes | - | Output directory |
| `--startup-timeout` | No | 600 | vLLM startup timeout (s) |
| `--benchmark-timeout` | No | 3600 | GuideLLM timeout (s) |
| `--keep-deployment` | No | False | Keep deployment after |

---

## Output Structure

```
validation_results/
└── validation_report.json              # Only file kept: SLO compliance + accuracy
```

**Per-config folders are temporary** (created during validation, deleted after metrics extracted):
- `config_15_rank_1/` → Contains vllm.log, guidellm.log, benchmarks.json
- Deleted after extracting metrics to final report

**validation_report.json** (SLO compliance + simulator accuracy only):
```json
{
  "timestamp": "2026-01-28 10:30:00",
  "summary": {
    "total_configs_tested": 6,
    "configs_meeting_slos": 4,
    "configs_failing_slos": 2
  },
  "results": [
    {
      "simulator": "BLIS",
      "config_id": 15,
      "rank": 1,
      "meets_slos": true,
      "slo_metrics": {
        "e2e_p95_ms": {
          "threshold_ms": 1000,
          "simulator_ms": 962.44,
          "real_ms": 978.50,
          "difference_ms": 16.06,
          "error_percent": 1.67,
          "passes": true
        }
      }
    }
  ]
}
```

**Key fields:**
- `meets_slos`: Boolean - does config meet ALL SLO requirements?
- `simulator_ms`: What simulator predicted
- `real_ms`: What real vLLM measured
- `difference_ms`: real - simulator (positive = simulator underestimated)
- `error_percent`: (difference / simulator) × 100
- `passes`: Does real metric meet the threshold?

---

## Critical Implementation Notes

### vLLM Restart (Mandatory)
- vLLM parameters (TP, batch_size, max_model_len) are startup-only
- Kill and restart between configs: `pkill -f 'vllm serve'`
- Wait 10s after kill, 10s after start, then health check

### GPU Allocation
- Automatically detect max(TP) from all configs
- Create deployment with that many GPUs
- TP=1 needs 1 GPU, TP=2 needs 2 GPUs

### Metric Extraction
- `request_latency` is in **seconds** (multiply by 1000 for ms)
- `time_to_first_token_ms` and `inter_token_latency_ms` already in ms
- Use `benchmarks[0].metrics.<metric>.total.percentiles.<pXX>`

### Simulator Accuracy Calculation
- **difference_ms** = real_ms - simulator_ms
  - Positive: Real vLLM slower than predicted
  - Negative: Real vLLM faster than predicted
- **error_percent** = (difference_ms / simulator_ms) × 100
  - Example: (978.50 - 962.44) / 962.44 × 100 = 1.67%

### Cleanup Strategy
- Per-config folders created temporarily during validation
- After extracting metrics to report, delete the folder with `shutil.rmtree()`
- Only `validation_report.json` remains at the end
- Keeps output directory clean and saves disk space

### Error Handling
- Continue validation if a config fails (OOM, timeout)
- Mark as ERROR with details in report
- Don't halt entire validation process
- Clean up temp folders even on error

---

## Code Reuse from saturation_orchestrator.py

**Direct reuse (100%):**
- `KubernetesManager` class
- `build_vllm_command_string()`
- `build_guidellm_command_string()`
- `wait_for_vllm_in_pod()`
- `wait_for_guidellm_completion()`

**New components:**
- Results file parsing
- Config merging
- Multi-config orchestration
- GuideLLM metric extraction
- Comparison logic (simulator vs real)
- Report generation (simplified)
- Temp folder cleanup

---

## Verification Steps

**1. Single config test:**
```bash
python config_validator.py \
  --blis-results results/config_exp/blis_results_lowprefix.json \
  --top-n 1 \
  --deployment-name vllm-server \
  --namespace diya \
  --output-dir validation_test

# Verify:
# - validation_test/validation_report.json exists
# - No config_* subdirectories (cleaned up)
# - Report shows meets_slos and error_percent
```

**2. Multi-config test:**
```bash
python config_validator.py \
  --blis-results results/config_exp/blis_results_lowprefix.json \
  --top-n 3 \
  --deployment-name vllm-server \
  --namespace diya \
  --output-dir validation_blis

# Verify:
# - Only validation_report.json in output dir
# - Summary shows configs_meeting_slos count
# - All temp folders cleaned up
```

**3. Dual-simulator test:**
```bash
python config_validator.py \
  --blis-results results/config_exp/blis_results_lowprefix.json \
  --vidur-results results/config_exp/vidur_results_lowprefix.json \
  --top-n 2 \
  --deployment-name vllm-server \
  --namespace diya \
  --output-dir validation_both

# Verify:
# - 4 results in report (2 BLIS + 2 Vidur)
# - Each marked with "simulator" field
# - No leftover directories
```

---

## Troubleshooting

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| **vLLM startup timeout** | Check `/tmp/vllm.log` | OOM → reduce gpu_memory_utilization<br>Model not found → check HUGGING_FACE_HUB_TOKEN |
| **GuideLLM timeout** | Check `/tmp/guidellm.log` | Increase `--benchmark-timeout` |
| **Pod not ready** | `kubectl describe pod` | Pending → insufficient GPUs<br>CrashLoopBackOff → check logs |
| **Metrics not extracted** | Check benchmarks.json structure | Verify GuideLLM version matches v0.5.3 |

---

## Success Criteria

- [ ] Report shows which configs meet SLOs (meets_slos: true/false)
- [ ] Report shows simulator accuracy for each SLO metric:
  - simulator_ms vs real_ms
  - difference_ms (real - simulator)
  - error_percent ((real - sim) / sim × 100)
- [ ] Summary counts configs meeting/failing SLOs
- [ ] Per-config temp folders deleted after validation
- [ ] Only validation_report.json remains in output directory
- [ ] Error handling prevents script crashes

---

## Final Output

After running the validator, the output directory contains **only**:

```
validation_results/
└── validation_report.json
```

**No per-config folders** - they are created temporarily, used to extract metrics, then deleted.

**Report answers two questions:**
1. ✓ Does the config meet SLOs? → `meets_slos: true/false`
2. ✓ How accurate was the simulator? → `error_percent` for each metric
