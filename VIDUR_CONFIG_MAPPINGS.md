# Vidur Config Mappings Update

## Overview

Updated `vidur_wrapper.py` to use correct Vidur CLI argument mappings and calculate `total_kv_blocks` using the same library function as BLIS.

## Changes Made

### 1. Added capacity_planner Import

```python
from capacity_planner import calculate_total_kv_blocks
```

This allows us to use the same KV block calculation logic as BLIS.

### 2. Calculate total_kv_blocks

Added calculation before building Vidur command (line ~297):

```python
# Calculate total_kv_blocks using same method as BLIS
total_kv_blocks = calculate_total_kv_blocks(
    model=config['model'],
    hardware=config['hardware'],
    tp=config.get('tp', 1),
    max_model_len=config['max_model_len'],
    gpu_memory_utilization=config['gpu_memory_utilization'],
    block_size=config.get('block_size', 16)
)
# Apply same 0.8 factor as BLIS
total_kv_blocks = int(total_kv_blocks * 0.8)
```

**Note**: The 0.8 factor matches BLIS behavior for conservative memory estimation.

### 3. Updated Vidur CLI Mappings

Added the following mappings to the Vidur command:

#### vLLM → Vidur Parameter Mappings

| vLLM Concept | Vidur CLI Argument | Value |
|--------------|-------------------|-------|
| `model` | `--replica_config_model_name` | `config['model']` |
| `hardware` | `--replica_config_device` | `device` (mapped: H100→h100, A100→a100) |
| `tp` | `--replica_config_tensor_parallel_size` | `config['tp']` |
| `batch_size` (max_num_seqs) | `--random_forrest_execution_time_predictor_config_prediction_max_batch_size` | `config['batch_size']` |
| `total_kv_blocks` | `--vllm_scheduler_config_num_blocks` | `total_kv_blocks` (calculated) |
| `watermark_blocks` | `--vllm_scheduler_config_watermark_blocks_fraction` | `0.0` (no watermark) |

#### Updated Command Structure

```python
cmd = base_cmd + [
    # Model and hardware (already correct)
    '--replica_config_model_name', config['model'],
    '--replica_config_device', device,
    '--replica_config_tensor_parallel_size', str(config.get('tp', 1)),
    '--replica_config_num_pipeline_stages', '1',
    '--cluster_config_num_replicas', '1',

    # Scheduler
    '--replica_scheduler_config_type', 'vllm',
    '--vllm_scheduler_config_batch_size_cap', str(config.get('batch_size', 256)),
    '--vllm_scheduler_config_max_tokens_in_batch', str(config.get('max_scheduled_tokens', 8192)),
    '--vllm_scheduler_config_num_blocks', str(total_kv_blocks),  # NEW
    '--vllm_scheduler_config_watermark_blocks_fraction', '0.0',  # NEW

    # Random Forest execution time predictor
    '--random_forrest_execution_time_predictor_config_prediction_max_batch_size', str(config.get('batch_size', 256)),  # NEW

    # Request generator
    '--request_generator_config_type', 'trace_replay',
    '--trace_request_generator_config_trace_file', actual_trace_file,
    '--trace_request_generator_config_max_tokens', str(config.get('max_model_len', 4096)),

    # Output
    '--metrics_config_output_dir', output_dir,
    ...
]
```

### 4. Updated Verbose Output

Added total_kv_blocks to verbose logging:

```python
if verbose:
    print(f"\nRunning Vidur simulation:")
    print(f"  QPS: {qps}")
    print(f"  Num Requests: {num_requests}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Max Scheduled Tokens: {config['max_scheduled_tokens']}")
    print(f"  Max Model Length: {config['max_model_len']}")
    print(f"  Total KV Blocks: {total_kv_blocks:,}")  # NEW
    print(f"  Command: {' '.join(cmd)}\n")
```

## Mapping Details

### 1. `--vllm_scheduler_config_num_blocks`

**Purpose**: Total number of KV cache blocks available

**Calculation**:
```python
total_kv_blocks = calculate_total_kv_blocks(
    model=config['model'],
    hardware=config['hardware'],
    tp=config['tp'],
    max_model_len=config['max_model_len'],
    gpu_memory_utilization=config['gpu_memory_utilization'],
    block_size=config.get('block_size', 16)
)
total_kv_blocks = int(total_kv_blocks * 0.8)  # Conservative factor
```

**Example**: For Llama-3.1-8B on H100 with:
- max_model_len: 8192
- gpu_memory_utilization: 0.90
- block_size: 16
- Result: ~140,000 blocks × 0.8 = ~112,000 blocks

### 2. `--random_forrest_execution_time_predictor_config_prediction_max_batch_size`

**Purpose**: Maximum batch size for Random Forest latency prediction

**Value**: Set to `config['batch_size']` (e.g., 256)

**Why needed**: Vidur's Random Forest predictor needs to know the maximum batch size to accurately predict execution times.

### 3. `--vllm_scheduler_config_watermark_blocks_fraction`

**Purpose**: Fraction of KV blocks to reserve as watermark for memory management

**Value**: Set to `0.0` (no watermark)

**Why 0.0**:
- Disables the watermark mechanism in vLLM scheduler
- Allows full utilization of calculated KV blocks
- Matches typical vLLM configuration without reserved blocks
- Simplifies memory accounting to match BLIS behavior

## Consistency with BLIS

Both BLIS and Vidur now use:

1. **Same KV block calculation**: Via `capacity_planner.calculate_total_kv_blocks()`
2. **Same 0.8 factor**: Conservative memory estimation
3. **Same input parameters**: model, hardware, tp, max_model_len, gpu_memory_utilization, block_size

### BLIS Code (blis_runner.py)

```python
from capacity_planner import calculate_total_kv_blocks

total_kv_blocks = calculate_total_kv_blocks(
    model=config['model'],
    hardware=config['hardware'],
    tp=config['tp'],
    max_model_len=config['max_model_len'],
    gpu_memory_utilization=config['gpu_memory_utilization'],
    block_size=config.get('block_size', 16)
)
total_kv_blocks = int(total_kv_blocks * 0.8)
```

### Vidur Code (vidur_wrapper.py)

```python
from capacity_planner import calculate_total_kv_blocks

total_kv_blocks = calculate_total_kv_blocks(
    model=config['model'],
    hardware=config['hardware'],
    tp=config.get('tp', 1),
    max_model_len=config['max_model_len'],
    gpu_memory_utilization=config['gpu_memory_utilization'],
    block_size=config.get('block_size', 16)
)
total_kv_blocks = int(total_kv_blocks * 0.8)
```

**Identical logic ensures consistent behavior across simulators.**

## Benefits

1. **Accurate Memory Modeling**: Vidur now uses calculated KV blocks instead of defaults
2. **Consistency**: Same calculation method as BLIS
3. **Proper Batch Size Handling**: Random Forest predictor gets correct max batch size
4. **Explicit Configuration**: All parameters explicitly set via CLI arguments

## Testing

### Verify KV Block Calculation

```bash
# Run with verbose mode to see calculated values
python qps_search.py --config test_config.json
# Or
python parallel_search.py --configs examples/configs_grid_search.yaml

# Example output:
#   Calculated total_kv_blocks: 112,184
```

### Compare BLIS vs Vidur

```bash
# BLIS (default)
python qps_search.py --config test_config.json

# Vidur (edit USE_VIDUR = True in qps_search.py)
python qps_search.py --config test_config.json
```

Both should show similar total_kv_blocks values (same calculation).

## Files Modified

- **vidur_wrapper.py**:
  - Added `calculate_total_kv_blocks` import
  - Added KV block calculation before Vidur command
  - Added `--vllm_scheduler_config_num_blocks` to command
  - Added `--random_forrest_execution_time_predictor_config_prediction_max_batch_size` to command
  - Updated verbose output to show total_kv_blocks

## Related Documentation

- [README_STEP4.md](README_STEP4.md) - Vidur integration guide
- [capacity_planner.py](capacity_planner.py) - KV block calculation logic
- [blis_runner.py](blis_runner.py) - BLIS implementation reference

---

**Summary**: Vidur now uses the same KV block calculation as BLIS and correctly maps vLLM parameters to Vidur CLI arguments, ensuring consistent and accurate memory modeling across both simulators.
