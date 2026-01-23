# Config Search Tool - Simplified Roadmap

## Goal
Given tail latency SLO (e.g., p90 < 1s) and workload trace, find optimal vLLM config on **single H100 GPU** that maximizes QPS.

**Fixed**: Hardware=H100, TP=1 (single GPU)
**Search over**: vLLM scheduler knobs (batch size, max_model_len, gpu_memory_utilization, scheduling limits)

---

## Step 1: BLIS Single Run (1 day)

**What**: Run BLIS with one config at one QPS, parse JSON output

**Implementation**:
Please use `openevolve` branch of blis (inference-sim) 

```python
# blis_runner.py
from capacity_planner import calculate_total_kv_blocks  # From llm-d-benchmark

def run_blis(config, qps, trace_file):
    # Calculate total_kv_blocks from max_model_len and gpu_memory_utilization
    total_kv_blocks = calculate_total_kv_blocks(
        model=config['model'],
        hardware='H100',
        tp=1,
        max_model_len=config['max_model_len'],
        gpu_memory_utilization=config['gpu_memory_utilization']
    )

    cmd = [
        './simulation_worker', 'run',
        '--model', config['model'],
        '--hardware', 'H100',
        '--tp', '1',
        '--rate', str(qps),
        '--trace-file', trace_file, # correct arg tracesWorkloadFilePath
        '--max-num-running-reqs', str(config['batch_size']),
        '--max-num-scheduled-tokens', str(config['max_scheduled_tokens']),  
        '--total-kv-blocks', str(total_kv_blocks),  # Calculated
        '--block-size-in-tokens', str(config['block_size']),
    ] # output is written to stdout as json format
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    return json.loads(result.stdout)  # {ttft_p90_ms, itl_p95_ms, e2e_p90_ms, ...}
```

**Config knobs** (BLIS):
- `batch_size`: 64, 128, 256, 512
- `max_scheduled_tokens`: 2048, 4096, 8192
- `max_model_len`: 4096, 8192, 16384 (2-3 values)
- `gpu_memory_utilization`: 0.85, 0.90, 0.95 (2-3 values)
- `block_size`: 16, 32
- `total_kv_blocks`: **Calculated** from max_model_len + gpu_memory_utilization

**Capacity Planner**:
```python
# capacity_planner.py (adapted from llm-d-benchmark)
def calculate_total_kv_blocks(model, hardware, tp, max_model_len, gpu_memory_utilization):
    """
    Calculate total KV cache blocks based on:
    - Model size and architecture
    - Hardware memory (H100 = 80GB)
    - Max sequence length
    - GPU memory utilization target

    Returns: Number of KV cache blocks
    """
    # Get model info
    model_memory_mb = get_model_memory_footprint(model, tp)
    device_memory_mb = get_device_memory(hardware)  # H100 = 80GB

    # Available memory for KV cache
    available_memory_mb = device_memory_mb * gpu_memory_utilization - model_memory_mb

    # Calculate KV cache size per block
    block_size = 16  # tokens per block
    kv_cache_per_block_mb = calculate_kv_cache_per_block(model, block_size)

    # Total blocks
    total_blocks = int(available_memory_mb / kv_cache_per_block_mb)

    return total_blocks
```

**Test**:
```bash
python blis_runner.py --rate 10.0 --trace traces/chat.csv --config config.json
```

---

## Step 2: Binary Search for Max QPS (1 day)

**What**: Find max QPS where SLO is met (0.1 QPS precision)

**Implementation**:
```python
def find_max_qps(config, slo_threshold_ms, trace_file):
    qps_min, qps_max = 0.1, 100.0

    # Probe at 10 QPS to narrow bounds
    probe = run_blis(config, 10.0, trace_file)
    if probe['e2e_p90_ms'] > slo_threshold_ms:
        qps_max = 10.0
    else:
        qps_min = 10.0

    # Binary search with 0.1 QPS threshold
    while (qps_max - qps_min) > 0.1:
        mid = (qps_min + qps_max) / 2
        metrics = run_blis(config, mid, trace_file)

        if metrics['e2e_p90_ms'] <= slo_threshold_ms:
            qps_min = mid
        else:
            qps_max = mid

    return qps_min, metrics
```

**Test**:
```bash
python search.py --config config.json --slo-p90-ms 1000 --trace traces/chat.csv
```

---

## Step 3: Parallel Config Search (1 day)

**What**: Evaluate N configs in parallel, report best

**Config space** (example):
```yaml
# configs.yaml
model: meta-llama/llama-3.1-8b-instruct
hardware: H100
tp: 1

configs:
  # Sweep over max_model_len and gpu_memory_utilization
  # total_kv_blocks will be calculated automatically

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

  - batch_size: 256
    max_scheduled_tokens: 8192
    max_model_len: 8192
    gpu_memory_utilization: 0.95
    block_size: 16

  - batch_size: 512
    max_scheduled_tokens: 8192
    max_model_len: 16384
    gpu_memory_utilization: 0.85
    block_size: 16
```

**Implementation**:
```python
from multiprocessing import Pool

def evaluate_config(config):
    max_qps, metrics = find_max_qps(config, slo_threshold, trace_file)

    # Include calculated total_kv_blocks in result
    total_kv_blocks = calculate_total_kv_blocks(
        config['model'], 'H100', 1,
        config['max_model_len'],
        config['gpu_memory_utilization']
    )

    return {
        'config': {**config, 'total_kv_blocks': total_kv_blocks},
        'max_qps': max_qps,
        'metrics': metrics
    }

# Parallel search
with Pool(8) as pool:
    results = pool.map(evaluate_config, configs)

best = max(results, key=lambda x: x['max_qps'])
```

**Test**:
```bash
python search.py --configs configs.yaml --slo-p90-ms 1000 --trace traces/chat.csv
# Output: Best config achieves 47.3 QPS
#         (max_model_len=8192, gpu_memory_utilization=0.90, total_kv_blocks=152340)
```

---

## Step 4: Vidur Wrapper (0.5 day)

**What**: Generate Vidur config YAML and run built-in explorer

**Vidur config example**:
```yaml
# vidur_config.yaml
clusters:
  - device: h100
    num_gpus: 1
    gpus_per_node: 1

schedulers:
  - scheduler: vllm

traces:
  - name: chat
    trace_file: "./traces/chat.csv"
    max_seq_len: 4096
    num_requests: 1000
    start_qps: 10

batch_sizes: [64, 128, 256, 512]
tp_dimensions: [1]
pp_dimensions: [1]

models:
  - name: llama-3.1-8b
    identifier: meta-llama/llama-3.1-8b-instruct
```

**Implementation**:
```python
def run_vidur_search(model, trace_file, slo_ms):
    # Generate config from template
    config = generate_vidur_config(model, trace_file)

    # Run Vidur config explorer
    cmd = [
        'python', '-m', 'vidur.config_optimizer.config_explorer.main',
        '--output-dir', './vidur_results',
        '--config-path', config,
        '--scheduling-delay-slo-quantile', '0.90',
        '--scheduling-delay-slo-value', str(slo_ms / 1000)
    ]
    subprocess.run(cmd, check=True)

    # Parse Vidur results
    return parse_vidur_results('./vidur_results')
```

**Test**:
```bash
python vidur_wrapper.py --model llama-3.1-8b --trace traces/chat.csv --slo-p90-ms 1000
```

---

## Step 5: Polish (0.5 day)

**Error handling**: Timeouts, OOM, invalid configs
**Output**: JSON with best config and all results
**Docs**: README with examples

---

## Complete Example

```bash
# 1. Prepare config space
cat > configs.yaml <<EOF
model: meta-llama/llama-3.1-8b-instruct
configs:
  - batch_size: 128
    max_scheduled_tokens: 4096
    max_model_len: 4096
    gpu_memory_utilization: 0.90
  - batch_size: 256
    max_scheduled_tokens: 8192
    max_model_len: 8192
    gpu_memory_utilization: 0.90
  - batch_size: 512
    max_scheduled_tokens: 8192
    max_model_len: 16384
    gpu_memory_utilization: 0.85
EOF

# 2. Run BLIS search
python config_search.py \
  --simulator blis \
  --configs configs.yaml \
  --trace traces/prefix_heavy_chat.csv \
  --slo-p90-ms 1000 \
  --num-workers 8

# 3. View results
# Best: batch_size=256, max_model_len=8192, gpu_mem_util=0.90
#       total_kv_blocks=152340 (calculated), max_qps=42.7
```

---

## File Structure

```
config_search/
├── blis_runner.py         # Run BLIS, parse JSON
├── capacity_planner.py    # Calculate total_kv_blocks from llm-d-benchmark
├── qps_search.py          # Binary search for max QPS
├── parallel_search.py     # Parallel config evaluation
├── vidur_wrapper.py       # Vidur config generator + runner
└── config_search.py       # Main CLI

examples/
├── configs_blis.yaml      # Example BLIS config space
├── configs_vidur.yaml     # Example Vidur config space
└── traces/
    └── chat.csv           # prompt_tokens,output_tokens
```

---

## vLLM Config Knobs

**BLIS** (sweep these):
- `--max-num-running-reqs`: Max requests in batch [64, 128, 256, 512]
- `--max-num-scheduled-tokens`: Max tokens per iteration [2048, 4096, 8192]
- `--max-model-len`: Max sequence length [4096, 8192, 16384]
- `--gpu-memory-utilization`: GPU memory target [0.85, 0.90, 0.95]
- `--block-size-in-tokens`: Block size [16, 32]
- `--total-kv-blocks`: **Calculated** via capacity planner (not swept directly)

**Vidur** (similar):
- `batch_size_cap`: Max batch size
- `max_tokens_in_batch`: Max tokens per batch
- `num_blocks`: KV cache blocks (auto-calculated by Vidur)
- `block_size`: Block size
- `watermark_blocks_fraction`: Safety margin [0.01, 0.05]

---

## Capacity Planner Integration

**Source**: https://github.com/llm-d/llm-d-benchmark/tree/main/config_explorer

**Key method**: `total_kv_cache_blocks(model, hardware, tp, max_model_len, gpu_memory_utilization)`

**Usage**:
1. User specifies: `max_model_len` and `gpu_memory_utilization`
2. Capacity planner calculates: `total_kv_blocks` based on:
   - Model memory footprint
   - Hardware memory (H100 = 80GB)
   - Available memory after model loading
   - KV cache size per block

**Example**:
```python
# Config: max_model_len=8192, gpu_memory_utilization=0.90
# Model: llama-3.1-8b (model_size ~16GB)
# Hardware: H100 (80GB)

# Calculation:
# - Available memory: 80GB * 0.90 = 72GB
# - Model memory: 16GB
# - KV cache memory: 72GB - 16GB = 56GB
# - KV per block (16 tokens): ~0.4MB
# - Total blocks: 56GB / 0.4MB ≈ 140,000 blocks

total_kv_blocks = calculate_total_kv_blocks(
    model='meta-llama/llama-3.1-8b-instruct',
    hardware='H100',
    tp=1,
    max_model_len=8192,
    gpu_memory_utilization=0.90
)
# Returns: 140000 (example)
```

---

## Timeline

| Step | Deliverable | Time |
|------|-------------|------|
| 1 | BLIS runner + capacity planner integration | 1 day |
| 2 | Binary search for max QPS | 1 day |
| 3 | Parallel config search | 1 day |
| 4 | Vidur wrapper | 0.5 day |
| 5 | Polish + docs | 0.5 day |
| **Total** | **Working tool** | **4 days** |

---

## Success Criteria

✅ Calculate total_kv_blocks from max_model_len + gpu_memory_utilization
✅ Find max QPS with 0.1 precision
✅ Search 20 configs in <1 hour
✅ Works with trace files
✅ Unified interface for BLIS/Vidur
✅ Clean JSON output
