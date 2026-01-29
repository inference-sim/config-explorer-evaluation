# Config Validator

Validates top-N simulator configs against real vLLM. Each config runs in its own K8s pod in parallel for fast validation and clean GPU memory state.

## Quick Start

```bash
# Validate top 3 configs in parallel
python config_validator.py \
  --simulator blis \
  --results results/config_exp/blis_config_exploration.json \
  --top-n 3 \
  --namespace diya \
  --output-file validation_report.json

# Keep pods/logs for debugging
python config_validator.py \
  --simulator blis \
  --results results/config_exp/blis_config_exploration.json \
  --top-n 3 \
  --namespace diya \
  --output-file validation_report.json \
  --keep-deployment \
  --keep-logs
```

## Workflow

1. Spawn N K8s pods (one per config) in parallel
2. Each pod:
   - Installs dependencies (guidellm)
   - Starts vLLM with config parameters
   - Runs GuideLLM benchmark at predicted QPS
   - Extracts metrics (real SLO values)
3. Collect results → generate `validation_report.json`
4. Cleanup: Delete temp pods/logs (unless `--keep-*` flags set)

## CLI Arguments

```
--simulator {blis,vidur}    Simulator type (required)
--results FILE              Results from parallel_search.py (required)
--top-n N                   Top N configs to validate (required)
--output-file FILE          Output report path (required)
--namespace NAMESPACE       K8s namespace (default: diya)
--image IMAGE              vLLM image (default: vllm/vllm-openai:v0.14.0)
--startup-timeout SECS     vLLM startup timeout (default: 600)
--benchmark-timeout SECS   GuideLLM timeout (default: 3600)
--keep-deployment          Keep pods after completion
--keep-logs               Keep per-config logs
```

## Output

```json
{
  "timestamp": "2026-01-28 10:30:00",
  "simulator": "BLIS",
  "summary": {
    "total_configs_tested": 3,
    "configs_meeting_slos": 2,
    "configs_failing_slos": 1,
    "total_guidellm_runtime_seconds": 110.45,
    "total_simulator_runtime_seconds": 309.96
  },
  "results": [{
    "config_id": 15,
    "rank": 1,
    "meets_slos": true,
    "benchmark_duration_seconds": 36.71,
    "slo_metrics": {
      "e2e_p95_ms": {
        "threshold_ms": 1000,
        "simulator_ms": 962.44,
        "real_ms": 978.50,
        "difference_ms": 16.06,
        "error_percent": 1.67,
        "passes": true
      }
    },
    "status": "SUCCESS"
  }]
}
```

**Key Fields:**
- `meets_slos`: True if config meets ALL SLO constraints
- `simulator_ms`: Predicted value from simulator
- `real_ms`: Measured value from real vLLM
- `error_percent`: Prediction accuracy: `(real - simulator) / simulator × 100`
- `total_guidellm_runtime_seconds`: Actual validation benchmark time
- `total_simulator_runtime_seconds`: Simulator config search time

## Troubleshooting

| Issue | Solution |
|-------|----------|
| vLLM startup timeout | Reduce `gpu_memory_utilization` or increase `--startup-timeout` |
| GuideLLM timeout | Increase `--benchmark-timeout` |
| Pod not ready | `kubectl get pods -n diya` - check for insufficient GPUs |
| Metrics not extracted | Check `guidellm.log` in config logs |

**Note:** By default, only `validation_report.json` is kept. Use `--keep-logs` to preserve per-config directories for debugging.
