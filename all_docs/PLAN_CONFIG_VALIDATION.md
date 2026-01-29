# Config Validator Implementation Plan

## Purpose
Validate top-N simulator configs against real vLLM to answer:
1. Do they meet SLOs in production?
2. How accurate are simulator predictions?

## Architecture
- **Input**: BLIS/Vidur results from `parallel_search.py`
- **Output**: `validation_report.json` with SLO compliance + prediction accuracy
- **Execution**: Parallel validation (one pod per config, ThreadPoolExecutor)
- **Cleanup**: Temp directories deleted after metrics extraction

## Input Schema
```json
{
  "metadata": {
    "slo_constraints": [{"metric": "e2e_p95_ms", "threshold_ms": 1000}],
    "workload": {...},
    "model": "...",
    "hardware": "H100"
  },
  "summary": {
    "total_search_runtime_seconds": 309.96
  },
  "successful_configs": [
    {
      "rank": 1,
      "config_id": 15,
      "max_qps": 7.27,
      "configuration": {...},
      "slo_metrics": {...}
    }
  ]
}
```

## Output Schema
```json
{
  "timestamp": "2026-01-28 10:30:00",
  "simulator": "BLIS",
  "summary": {
    "total_configs_tested": 3,
    "configs_meeting_slos": 2,
    "configs_failing_slos": 1,
    "total_guidellm_runtime_seconds": 36.71,
    "total_simulator_runtime_seconds": 309.96
  },
  "results": [
    {
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
    }
  ]
}
```

## Key Fields
- `meets_slos`: Boolean - config meets ALL SLO constraints
- `simulator_ms`: Predicted value
- `real_ms`: Measured value from real vLLM
- `error_percent`: `(real - simulator) / simulator × 100`
- `total_simulator_runtime_seconds`: Config exploration time from simulator
- `total_guidellm_runtime_seconds`: Actual validation benchmark time

## Core Implementation

**Workflow:**
1. Load simulator results + metadata
2. Get top-N configs
3. For each config (parallel):
   - Create K8s pod with fresh GPU
   - Start vLLM with config parameters
   - Run GuideLLM at simulator's predicted QPS
   - Extract metrics, compare vs simulator prediction
   - Calculate SLO compliance + prediction error
   - Delete temp directory
4. Aggregate results into report

**Key constraint:** vLLM restart required between configs (parameters are startup-only)

## CLI Arguments
```
--simulator {blis,vidur}          Required: which simulator
--results FILE                     Required: results JSON from parallel_search.py
--top-n N                          Required: number of top configs to validate
--output-file FILE                 Required: output report path
--namespace NAMESPACE              K8s namespace (default: diya)
--image IMAGE                      vLLM image (default: vllm/vllm-openai:v0.14.0)
--startup-timeout SECS             vLLM startup timeout (default: 600)
--benchmark-timeout SECS           GuideLLM timeout (default: 3600)
--keep-deployment                  Keep pods after completion (for debugging)
--keep-logs                        Keep per-config logs (for debugging)
```

## Error Handling
- Continue if a config fails (mark as ERROR)
- Clean up temp directories even on error
- Log details in result for debugging
- Don't halt entire validation process

## Success Criteria
- [ ] Report shows SLO compliance per config
- [ ] Prediction accuracy (error_percent) calculated
- [ ] Summary aggregates meeting/failing counts
- [ ] Temp directories cleaned up
- [ ] Only validation_report.json remains
