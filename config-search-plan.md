# Configuration Search Tool for Inference-Sim (BLIS) and Vidur

## Overview

Build a unified configuration search tool that finds the optimal vLLM configuration to maximize QPS while meeting latency objectives for both inference-sim (BLIS) and Vidur simulators.

**Key Features:**
- Bayesian Optimization for intelligent configuration search
- Parallel simulation execution (4-8 concurrent simulations)
- Performance prediction to eliminate infeasible configs
- Adaptive search parameters based on model/hardware
- Robust error handling and timeout mechanisms
- Multi-level search strategy for efficiency

## Summary of Improvements

Based on comprehensive review, this plan incorporates:

1. **Reduced Runtime**: From 32+ hours to ~30 minutes through:
   - Bayesian optimization replacing exhaustive grid search
   - 4-8 parallel workers for concurrent simulations
   - Intelligent sampling with Latin Hypercube Design
   - Early stopping for failing configurations

2. **Smarter Search**:
   - Adaptive initial QPS based on model size and hardware
   - Dynamic convergence criteria (percentage-based)
   - Performance prediction to skip infeasible configs
   - Model-aware search space generation

3. **Robustness**:
   - Comprehensive error handling for timeouts, OOM, failures
   - Retry mechanisms with exponential backoff
   - Graceful degradation when configs fail
   - Result validation with multiple runs

4. **Better Coverage**:
   - Added missing critical parameters (memory margins, block sizes)
   - Adaptive parameter ranges based on model/workload
   - Support for all scheduler types
   - Proper handling of large context lengths

## User Requirements

**Inputs:**
- Latency objectives (TTFT, ITL, E2E with percentile targets)
- LLM type (model name)
- Workload type (chatbot, summarization, contentgen, multidoc, or custom parameters)

**Outputs:**
- Configuration that delivers maximum QPS
- The full configuration details
- Achieved QPS and latency metrics

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────┐
│          Configuration Search Orchestrator          │
│  (Unified interface for both BLIS and Vidur)       │
└──────────────┬────────────────────┬─────────────────┘
               │                    │
               ▼                    ▼
    ┌──────────────────┐  ┌──────────────────┐
    │  BLIS Adapter    │  │  Vidur Adapter   │
    │  - Build CLI     │  │  - Build CLI     │
    │  - Run sim       │  │  - Run sim       │
    │  - Parse output  │  │  - Parse output  │
    │  - Timeout mgmt   │  │  - Dir scanning  │
    └──────────────────┘  └──────────────────┘
               │                    │
               ▼                    ▼
         ┌──────────────────────────────┐
         │  Bayesian Search Optimizer   │
         │  - Gaussian Process model    │
         │  - Acquisition function      │
         │  - Binary search on QPS      │
         └──────────────────────────────┘
               │
               ▼
         ┌──────────────────────────────┐
         │    Parallel Executor         │
         │  - Process pool (4-8 sims)   │
         │  - Queue management          │
         │  - Failure recovery          │
         └──────────────────────────────┘
               │
               ▼
         ┌──────────────────────────────┐
         │  Performance Predictor       │
         │  - Roofline model           │
         │  - Memory estimation        │
         │  - Feasibility check        │
         └──────────────────────────────┘
```

## Configuration Search Space

### BLIS (Inference-Sim) Parameters

**Hardware Selection:**
- `hardware`: [H100, A100-SXM]
- `tp`: [1, 2, 4, 8]

**vLLM Server Configuration (tunable):**
- `total-kv-blocks`: [100000, 500000, 1000000, 2000000]
- `max-num-running-reqs`: [64, 96, 128, 192, 256, 384, 512, 768, 1024]
- `max-num-scheduled-tokens`: [1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288]
- `block-size-in-tokens`: [8, 16, 32] (affects memory efficiency)
- `long-prefill-token-threshold`: [0, 512, 1024, 2048]
- `max-model-len`: [2048, 4096, 8192, 16384, 32768] (based on model capacity)

**Fixed Parameters:**
- `model`: From user input
- `workload`: From user input
- `max-prompts`: 100 (reduced for faster screening, 500 for final validation)
- `seed`: 42 (reproducibility)
- `log`: error (minimize output)

### Vidur Parameters

**Hardware Selection:**
- `replica_config_device`: [a100, h100, a40]
- `replica_config_tensor_parallel_size`: [1, 2, 4, 8]
- `replica_config_num_pipeline_stages`: [1, 2, 4] (if applicable)
- `cluster_config_num_replicas`: [1, 2, 4, 8]

**Scheduler Configuration (tunable):**
- `replica_scheduler_config_type`: [vllm, sarathi, orca, faster_transformer, lightllm]
- `replica_scheduler_config_batch_size_cap`: [64, 96, 128, 192, 256, 384, 512, 768, 1024]
- `replica_scheduler_config_watermark_blocks_fraction`: [0.01, 0.05, 0.1] (OOM safety)
- `replica_scheduler_config_block_size`: [8, 16, 32]
- `vllm_scheduler_config_max_tokens_in_batch`: [2048, 3072, 4096, 6144, 8192, 12288, 16384]
- `sarathi_scheduler_config_chunk_size`: [256, 512, 1024, 2048]

**Memory Configuration:**
- `replica_config_memory_margin_fraction`: [0.05, 0.1, 0.15, 0.2] (stability margin)
- `replica_scheduler_config_num_blocks`: Auto-calculated based on memory

**Fixed Parameters:**
- `replica_config_model_name`: From user input
- Length/interval generators: Based on workload type
- `synthetic_request_generator_config_num_requests`: 100 (fast screening), 512 (validation)
- `seed`: 42
- `log_level`: warning
- `metrics_config_enable_chrome_trace`: False (faster)

## Search Algorithm

### Multi-Level Bayesian Optimization Strategy

#### Phase 1: Coarse-Grained Search with Performance Prediction
1. **Initial Sampling (20-30 configs)**:
   - Use Latin Hypercube Sampling for initial coverage
   - Select configs across hardware/TP/batch size dimensions
   - Run at adaptive initial QPS based on model size

2. **Performance Prediction Filtering**:
   - Apply roofline model to eliminate infeasible configs
   - Calculate memory requirements: `mem_needed = model_params + kv_cache_size`
   - Skip configs exceeding hardware limits
   - Estimate theoretical max QPS: `max_qps = min(compute_bound, memory_bound)`

3. **Gaussian Process Modeling**:
   - Build surrogate model of performance surface
   - Features: hardware specs, TP, batch size, scheduler params
   - Target: maximum achievable QPS under constraints

#### Phase 2: Fine-Grained Bayesian Optimization
1. **Acquisition Function** (Expected Improvement):
   ```python
   def acquisition_function(config, gp_model, best_qps):
       mu, sigma = gp_model.predict(config)
       z = (mu - best_qps) / sigma
       ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
       return ei
   ```

2. **Iterative Refinement** (50-70 iterations):
   - Select next config using acquisition function
   - Run simulation with binary search for max QPS
   - Update GP model with result
   - Continue until convergence or budget exhausted

#### Phase 3: Binary Search on QPS (Per Configuration)

**Adaptive Initial QPS**:
```python
def get_initial_qps(model_params, hardware_tflops, tp):
    # Scale by model size and hardware capacity
    base_qps = 10.0
    size_factor = 1e9 / model_params  # Smaller models = higher QPS
    hw_factor = hardware_tflops / 300  # Normalize to A100
    tp_factor = math.sqrt(tp)  # Sublinear scaling

    return base_qps * size_factor * hw_factor * tp_factor
```

**Adaptive Convergence**:
```python
def get_convergence_threshold(current_qps):
    # 1% of current QPS or 0.5 req/s, whichever is larger
    return max(0.5, 0.01 * current_qps)
```

**Binary Search with Early Stopping**:
```python
def binary_search_qps(config, adapter):
    qps_min = 0.1
    qps_max = get_theoretical_max_qps(config)  # From roofline model

    # Warm-up run at low QPS
    warmup_qps = max(qps_min, qps_max * 0.01)
    metrics = run_with_timeout(adapter, config, warmup_qps, timeout=300)

    if not meets_constraints(metrics):
        return 0  # Early stop if fails at very low QPS

    best_qps = warmup_qps
    iterations = 0

    while qps_max - qps_min > get_convergence_threshold(best_qps) and iterations < 15:
        mid_qps = (qps_min + qps_max) / 2
        metrics = run_with_timeout(adapter, config, mid_qps, timeout=300)

        if meets_constraints(metrics):
            best_qps = mid_qps
            qps_min = mid_qps
        else:
            qps_max = mid_qps

        iterations += 1

    # Final validation run
    validation_metrics = run_with_timeout(adapter, config, best_qps, timeout=300)
    if meets_constraints(validation_metrics):
        return best_qps
    else:
        return best_qps * 0.95  # Safety margin
```

### Parallel Execution Framework

```python
class ParallelSimulationExecutor:
    def __init__(self, num_workers=4):
        self.pool = multiprocessing.Pool(num_workers)
        self.active_jobs = {}
        self.results_cache = {}

    def submit(self, config, qps, adapter):
        job_id = hash((str(config), qps))
        if job_id in self.results_cache:
            return self.results_cache[job_id]

        future = self.pool.apply_async(
            run_simulation_with_retry,
            args=(adapter, config, qps),
            kwargs={'max_retries': 3, 'timeout': 300}
        )
        self.active_jobs[job_id] = future
        return job_id

    def collect_results(self):
        completed = []
        for job_id, future in list(self.active_jobs.items()):
            if future.ready():
                try:
                    result = future.get()
                    self.results_cache[job_id] = result
                    completed.append((job_id, result))
                    del self.active_jobs[job_id]
                except Exception as e:
                    logging.error(f"Job {job_id} failed: {e}")
                    del self.active_jobs[job_id]
        return completed
```

### Latency Constraint Validation

```python
def meets_constraints(metrics, constraints):
    if metrics is None:  # Simulation failed
        return False

    for metric_name, (percentile, threshold) in constraints.items():
        metric_key = f"{metric_name}_{percentile}_ms"
        if metric_key not in metrics:
            logging.warning(f"Metric {metric_key} not found")
            return False
        if metrics[metric_key] > threshold:
            return False

    return True
```

### Search Space Pruning with Performance Models

**Roofline-Based Pruning**:
```python
def is_config_feasible(config, model_info):
    # Memory check
    model_params = model_info['params']
    seq_len = config['max_model_len']
    batch_size = config['max_num_running_reqs']
    kv_cache_size = estimate_kv_cache_size(model_params, seq_len, batch_size)

    device_memory = get_device_memory(config['hardware'])
    if (model_params + kv_cache_size) > device_memory * 0.9:
        return False

    # Compute check
    tflops_required = estimate_tflops(model_params, batch_size)
    device_tflops = get_device_tflops(config['hardware'])
    if tflops_required > device_tflops * config['tp']:
        return False

    return True
```

**Intelligent Priority Ordering**:
```python
def prioritize_configs(configs, model_info):
    # Score configs based on expected performance
    scored_configs = []

    for config in configs:
        score = 0
        # Higher TP generally better for large models
        score += config['tp'] * (model_info['params'] / 1e9)

        # Larger batches better for throughput
        score += math.log(config['max_num_running_reqs'])

        # Prefer efficient schedulers
        scheduler_scores = {'vllm': 1.0, 'sarathi': 0.9, 'orca': 0.8}
        score *= scheduler_scores.get(config.get('scheduler_type', 'vllm'), 0.7)

        # Prefer newer hardware
        hw_scores = {'H100': 1.0, 'h100': 1.0, 'A100-SXM': 0.7, 'a100': 0.7, 'a40': 0.5}
        score *= hw_scores.get(config['hardware'], 0.5)

        scored_configs.append((score, config))

    return [cfg for _, cfg in sorted(scored_configs, reverse=True)]
```

## Implementation Plan

### File Structure

```
config-explorer-evaluation/
├── config_search/
│   ├── __init__.py
│   ├── main.py                    # CLI entry point
│   ├── orchestrator.py            # Main search orchestrator
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract base adapter
│   │   ├── blis_adapter.py        # BLIS simulator adapter
│   │   └── vidur_adapter.py       # Vidur simulator adapter
│   ├── search/
│   │   ├── __init__.py
│   │   ├── bayesian_optimizer.py  # Bayesian optimization with GP
│   │   ├── binary_search.py       # Binary search on QPS
│   │   ├── performance_predictor.py # Roofline model & feasibility
│   │   └── sampling.py             # Latin hypercube sampling
│   ├── parallel/
│   │   ├── __init__.py
│   │   ├── executor.py             # Parallel simulation executor
│   │   └── job_manager.py          # Job queue and retry logic
│   ├── config/
│   │   ├── __init__.py
│   │   ├── search_space.py        # Define search spaces
│   │   ├── constraints.py         # Latency constraint definitions
│   │   └── hardware_specs.py      # Hardware specifications
│   └── utils/
│       ├── __init__.py
│       ├── metrics_parser.py      # Parse simulator outputs
│       ├── result_formatter.py    # Format final results
│       ├── timeout_manager.py     # Timeout and process management
│       └── cache.py               # Results caching
├── requirements.txt               # Python dependencies
└── README.md                      # Usage documentation
```

### Core Components

#### 1. Base Adapter (adapters/base.py)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import subprocess
import time

class SimulatorAdapter(ABC):
    @abstractmethod
    def build_command(self, config: Dict[str, Any], qps: float) -> List[str]:
        """Build CLI command for simulator"""
        pass

    def run_simulation(self, command: List[str], timeout: int = 300) -> Optional[Dict[str, float]]:
        """Run simulation with timeout and return metrics"""
        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )

            if result.returncode != 0:
                logging.error(f"Simulation failed: {result.stderr}")
                return None

            metrics = self.parse_output(result.stdout)
            metrics['simulation_time'] = time.time() - start_time
            return metrics

        except subprocess.TimeoutExpired:
            logging.error(f"Simulation timed out after {timeout}s")
            return None
        except Exception as e:
            logging.error(f"Simulation error: {e}")
            return None

    @abstractmethod
    def parse_output(self, stdout: str) -> Dict[str, float]:
        """Parse simulator output to extract metrics"""
        pass

    @abstractmethod
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model parameters and requirements"""
        pass
```

#### 2. BLIS Adapter (adapters/blis_adapter.py)

```python
class BlisAdapter(SimulatorAdapter):
    def __init__(self, blis_binary_path: str):
        self.binary_path = blis_binary_path

    def build_command(self, config: Dict, qps: float) -> List[str]:
        cmd = [self.binary_path, "run"]
        cmd.extend([
            "--model", config["model"],
            "--hardware", config["hardware"],
            "--tp", str(config["tp"]),
            "--rate", str(qps),
            "--max-prompts", "500",
            "--total-kv-blocks", str(config["total_kv_blocks"]),
            "--max-num-running-reqs", str(config["max_num_running_reqs"]),
            "--max-num-scheduled-tokens", str(config["max_num_scheduled_tokens"]),
            "--log", "error"
        ])
        # Add workload params
        if config.get("workload"):
            cmd.extend(["--workload", config["workload"]])
        return cmd

    def parse_output(self, stdout: str) -> Dict[str, float]:
        # Parse BLIS output format
        metrics = {}
        for line in stdout.split('\n'):
            if 'ttft_p90_ms' in line:
                metrics['ttft_p90_ms'] = float(line.split(':')[1].strip())
            # ... parse other metrics
        return metrics
```

#### 3. Vidur Adapter (adapters/vidur_adapter.py)

```python
class VidurAdapter(SimulatorAdapter):
    def __init__(self, vidur_module_path: str):
        self.module_path = vidur_module_path

    def build_command(self, config: Dict, qps: float) -> List[str]:
        cmd = ["python", "-m", "vidur.main"]
        cmd.extend([
            "--replica_config_device", config["device"],
            "--replica_config_model_name", config["model"],
            "--replica_config_tensor_parallel_size", str(config["tp"]),
            "--cluster_config_num_replicas", str(config["num_replicas"]),
            "--poisson_request_interval_generator_config_qps", str(qps),
            "--replica_scheduler_config_type", config["scheduler_type"],
            "--replica_scheduler_config_batch_size_cap", str(config["batch_size_cap"]),
            "--log_level", "warning",
            "--metrics_config_enable_chrome_trace", "False"
        ])
        return cmd

    def run_simulation(self, command: List[str]) -> Dict[str, float]:
        # Run simulation
        result = subprocess.run(command, capture_output=True, text=True)
        # Parse output directory for metrics
        output_dir = self._find_latest_output_dir()
        return self._parse_metrics_from_json(output_dir)
```

#### 4. Search Orchestrator (orchestrator.py)

```python
from search.bayesian_optimizer import BayesianOptimizer
from search.performance_predictor import PerformancePredictor
from parallel.executor import ParallelSimulationExecutor
import logging
import time

class ConfigSearchOrchestrator:
    def __init__(self, adapter: SimulatorAdapter, search_space: Dict,
                 constraints: Dict, num_workers: int = 4):
        self.adapter = adapter
        self.search_space = search_space
        self.constraints = constraints
        self.num_workers = num_workers

        # Initialize components
        self.optimizer = BayesianOptimizer(search_space)
        self.predictor = PerformancePredictor()
        self.executor = ParallelSimulationExecutor(num_workers)

        self.best_config = None
        self.best_qps = 0
        self.results_history = []

    def search(self, max_iterations: int = 100, time_limit: int = 14400) -> Dict[str, Any]:
        start_time = time.time()
        iteration = 0

        # Phase 1: Initial sampling with Latin Hypercube
        logging.info("Phase 1: Initial sampling...")
        initial_configs = self.optimizer.get_initial_samples(n_samples=25)

        # Filter with performance predictor
        model_info = self.adapter.get_model_info(self.search_space['model'][0])
        feasible_configs = [
            cfg for cfg in initial_configs
            if self.predictor.is_feasible(cfg, model_info)
        ]
        logging.info(f"Filtered to {len(feasible_configs)} feasible configs")

        # Phase 2: Bayesian optimization
        logging.info("Phase 2: Bayesian optimization...")

        while iteration < max_iterations and (time.time() - start_time) < time_limit:
            # Get next configs to evaluate
            if iteration < len(feasible_configs):
                # Use initial samples first
                configs_to_eval = feasible_configs[iteration:iteration+self.num_workers]
            else:
                # Use acquisition function
                configs_to_eval = []
                for _ in range(min(self.num_workers, max_iterations - iteration)):
                    next_config = self.optimizer.suggest_next(self.results_history)
                    if self.predictor.is_feasible(next_config, model_info):
                        configs_to_eval.append(next_config)

            # Submit parallel jobs
            job_ids = []
            for config in configs_to_eval:
                # Binary search for max QPS
                job_id = self.executor.submit_binary_search(
                    config, self.adapter, self.constraints, model_info
                )
                job_ids.append((job_id, config))

            # Collect results
            for job_id, config in job_ids:
                result = self.executor.wait_for_result(job_id, timeout=600)
                if result:
                    max_qps, metrics = result
                    self.results_history.append({
                        'config': config,
                        'max_qps': max_qps,
                        'metrics': metrics
                    })

                    if max_qps > self.best_qps:
                        self.best_qps = max_qps
                        self.best_config = config
                        self.best_metrics = metrics
                        logging.info(f"New best QPS: {max_qps:.2f}")

            iteration += len(configs_to_eval)
            self._print_progress(iteration, start_time)

        # Final validation of best config
        logging.info("Phase 3: Final validation...")
        final_metrics = self._validate_best_config()

        return {
            "simulator": self.adapter.__class__.__name__,
            "model": self.search_space['model'][0],
            "workload": self.search_space.get('workload', ['custom'])[0],
            "search_time_seconds": time.time() - start_time,
            "configurations_tested": len(self.results_history),
            "best_result": {
                "max_qps": self.best_qps,
                "configuration": self.best_config,
                "metrics": final_metrics,
                "constraints_met": self._check_constraints(final_metrics)
            },
            "all_viable_configs": sorted(
                [r for r in self.results_history if r['max_qps'] > 0],
                key=lambda x: x['max_qps'],
                reverse=True
            )[:10]  # Top 10 configs
        }

    def _validate_best_config(self) -> Dict[str, float]:
        """Run best config 3 times to ensure stability"""
        validation_metrics = []
        for i in range(3):
            logging.info(f"Validation run {i+1}/3")
            cmd = self.adapter.build_command(self.best_config, self.best_qps * 0.98)
            metrics = self.adapter.run_simulation(cmd, timeout=600)
            if metrics and self._check_constraints(metrics):
                validation_metrics.append(metrics)

        if not validation_metrics:
            logging.warning("Validation failed, using original metrics")
            return self.best_metrics

        # Average the metrics
        avg_metrics = {}
        for key in validation_metrics[0]:
            if key.endswith('_ms') or key == 'responses_per_sec':
                values = [m[key] for m in validation_metrics]
                avg_metrics[key] = sum(values) / len(values)

        return avg_metrics
```

#### 5. CLI Interface (main.py)

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description='Config Search for LLM Simulators')
    parser.add_argument('--simulator', choices=['blis', 'vidur'], required=True)
    parser.add_argument('--model', required=True, help='LLM model name')
    parser.add_argument('--workload', required=True)

    # Latency constraints
    parser.add_argument('--ttft-p90-ms', type=float, help='TTFT P90 threshold (ms)')
    parser.add_argument('--ttft-p95-ms', type=float, help='TTFT P95 threshold (ms)')
    parser.add_argument('--itl-p95-ms', type=float, help='ITL P95 threshold (ms)')
    parser.add_argument('--e2e-p99-ms', type=float, help='E2E P99 threshold (ms)')

    # Paths
    parser.add_argument('--blis-binary', help='Path to BLIS binary')
    parser.add_argument('--vidur-path', help='Path to Vidur module')

    # Output
    parser.add_argument('--output', default='search_results.json')

    args = parser.parse_args()

    # Build constraints dict
    constraints = {}
    if args.ttft_p90_ms:
        constraints['ttft'] = ('p90', args.ttft_p90_ms)
    # ... add other constraints

    # Create adapter
    if args.simulator == 'blis':
        adapter = BlisAdapter(args.blis_binary)
        search_space = get_blis_search_space(args.model, args.workload)
    else:
        adapter = VidurAdapter(args.vidur_path)
        search_space = get_vidur_search_space(args.model, args.workload)

    # Run search
    orchestrator = ConfigSearchOrchestrator(adapter, search_space, constraints)
    results = orchestrator.search()

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Best QPS: {results['max_qps']:.2f}")
    print(f"Configuration: {json.dumps(results['config'], indent=2)}")
```

### Search Space Definitions (config/search_space.py)

```python
import numpy as np

def get_model_size(model_name: str) -> float:
    """Estimate model parameters in billions"""
    model_sizes = {
        'llama-3.1-8b': 8, 'llama-3.1-70b': 70, 'llama-3.3-70b': 70,
        'phi-4': 14, 'phi-2': 2.7, 'mixtral-8x7b': 47,
        'qwen2.5-7b': 7, 'qwen-72b': 72,
        'nemotron-70b': 70, 'llama-2-7b': 7, 'llama-2-70b': 70
    }
    for key, size in model_sizes.items():
        if key in model_name.lower():
            return size
    return 13  # Default medium size

def get_workload_context(workload: str) -> int:
    """Get typical context length for workload"""
    contexts = {
        'chatbot': 2048, 'summarization': 8192,
        'contentgen': 4096, 'multidoc': 16384
    }
    return contexts.get(workload, 4096)

def get_blis_search_space(model: str, workload: str) -> Dict:
    model_size = get_model_size(model)
    context_len = get_workload_context(workload)

    # Adaptive TP based on model size
    if model_size < 10:
        tp_values = [1, 2, 4]
    elif model_size < 50:
        tp_values = [2, 4, 8]
    else:
        tp_values = [4, 8]

    # Adaptive batch sizes based on model and context
    if model_size * context_len < 20000:  # Small model or context
        batch_values = [128, 256, 384, 512, 768, 1024]
    else:  # Large model or context
        batch_values = [32, 64, 96, 128, 192, 256]

    # Adaptive KV blocks based on context length
    kv_multiplier = context_len / 2048
    kv_values = [
        int(500000 * kv_multiplier),
        int(1000000 * kv_multiplier),
        int(2000000 * kv_multiplier)
    ]

    return {
        "model": [model],
        "hardware": ["H100", "A100-SXM"],
        "tp": tp_values,
        "total_kv_blocks": kv_values,
        "max_num_running_reqs": batch_values,
        "max_num_scheduled_tokens": [
            min(2048 * tp, context_len) for tp in tp_values
        ],
        "long_prefill_token_threshold": [0, min(1024, context_len // 4)],
        "max_model_len": [context_len],
        "block_size_in_tokens": [16, 32] if model_size < 20 else [16],
        "workload": [workload]
    }

def get_vidur_search_space(model: str, workload: str) -> Dict:
    model_size = get_model_size(model)
    context_len = get_workload_context(workload)

    # Similar adaptive logic for Vidur
    if model_size < 10:
        tp_values = [1, 2, 4]
        devices = ["a100", "h100"]
    elif model_size < 50:
        tp_values = [2, 4, 8]
        devices = ["h100", "a100"]
    else:
        tp_values = [4, 8]
        devices = ["h100"]  # Only H100 for very large models

    batch_values = (
        [64, 128, 256, 512] if model_size < 20
        else [32, 64, 128, 256]
    )

    return {
        "model": [model],
        "device": devices,
        "tp": tp_values,
        "num_replicas": [1, 2] if model_size < 50 else [1],
        "scheduler_type": ["sarathi", "vllm"] if context_len > 4096 else ["vllm", "orca"],
        "batch_size_cap": batch_values,
        "max_tokens_in_batch": [
            min(4096 * tp, context_len * 4) for tp in tp_values
        ],
        "memory_margin_fraction": [0.1, 0.15],
        "block_size": [16, 32],
        "watermark_fraction": [0.01, 0.05]
    }
```

## Output Format

### JSON Output Structure

```json
{
  "simulator": "blis",
  "model": "meta-llama/llama-3.1-8b-instruct",
  "workload": "chatbot",
  "search_time_seconds": 342.5,
  "configurations_tested": 127,
  "best_result": {
    "max_qps": 45.3,
    "configuration": {
      "hardware": "H100",
      "tp": 4,
      "total_kv_blocks": 2000000,
      "max_num_running_reqs": 512,
      "max_num_scheduled_tokens": 4096,
      "long_prefill_token_threshold": 2048
    },
    "metrics": {
      "ttft_mean_ms": 12.5,
      "ttft_p90_ms": 18.2,
      "ttft_p95_ms": 21.3,
      "ttft_p99_ms": 26.7,
      "itl_mean_ms": 5.2,
      "itl_p95_ms": 8.1,
      "e2e_mean_ms": 234.5,
      "e2e_p99_ms": 487.3,
      "responses_per_sec": 45.3,
      "tokens_per_sec": 12450.2
    },
    "constraints_met": true
  },
  "all_viable_configs": [
    {
      "qps": 45.3,
      "config": { "..." }
    }
  ]
}
```

### Console Output

```
Configuration Search for BLIS
Model: meta-llama/llama-3.1-8b-instruct
Workload: chatbot
Latency Constraints:
  - TTFT P90 < 20ms
  - ITL P95 < 10ms
  - E2E P99 < 500ms

Searching... (Phase 1: Grid Search)
  ✓ H100, TP=8, batch=1024: Viable at QPS=10
  ✗ H100, TP=4, batch=512: Latency violation
  ...

Found 12 viable configurations

Searching... (Phase 2: Binary Search on QPS)
  Config 1/12: H100, TP=8, batch=1024
    Testing QPS=500.0... ✗ (latency violation)
    Testing QPS=250.0... ✗ (latency violation)
    Testing QPS=125.0... ✓
    Testing QPS=187.5... ✗
    Testing QPS=156.2... ✓
    ...
    Max QPS: 163.5

  Config 2/12: ...

===============================
BEST CONFIGURATION FOUND
===============================
Simulator: BLIS
Max QPS: 163.5 requests/second
Hardware: H100
Tensor Parallelism: 8
Batch Config:
  - max_num_running_reqs: 1024
  - max_num_scheduled_tokens: 8192
  - total_kv_blocks: 2000000

Achieved Metrics:
  TTFT P90: 18.2ms (target: 20ms) ✓
  ITL P95: 8.1ms (target: 10ms) ✓
  E2E P99: 487.3ms (target: 500ms) ✓

Token Throughput: 45,123 tokens/sec
===============================
```

## Runtime Estimates

With the improved approach:
- **Initial sampling**: 25 configs × 30s = 12.5 minutes (with 4 parallel workers)
- **Bayesian optimization**: 75 iterations × 30s / 4 workers = 9.4 minutes
- **Final validation**: 3 runs × 30s = 1.5 minutes
- **Total estimated time**: ~25-30 minutes (vs 32+ hours originally)

## Implementation Steps

### Step 1: Project Setup
- Create directory structure
- Set up Python virtual environment
- Create requirements.txt with dependencies:
  ```
  numpy>=1.21.0
  scipy>=1.7.0        # For Gaussian Process
  scikit-learn>=1.0.0 # For Latin Hypercube Sampling
  matplotlib>=3.4.0   # For optimization visualization
  ```

### Step 2: Implement Core Infrastructure
- `adapters/base.py`: Abstract base adapter with timeout support
- `config/hardware_specs.py`: Hardware specifications (TFLOPS, memory)
- `utils/timeout_manager.py`: Process management with timeouts
- `utils/cache.py`: Results caching system

### Step 3: Implement Performance Prediction
- `search/performance_predictor.py`:
  - Roofline model implementation
  - Memory requirement estimation
  - Feasibility checking
- `config/search_space.py`: Adaptive search space generation

### Step 4: Implement Bayesian Optimization
- `search/bayesian_optimizer.py`:
  - Gaussian Process surrogate model
  - Expected Improvement acquisition
  - Latin Hypercube initial sampling
- `search/sampling.py`: LHS implementation

### Step 5: Implement Parallel Execution
- `parallel/executor.py`: Process pool management
- `parallel/job_manager.py`: Job queue and retry logic
- Handle failures gracefully with retry mechanism

### Step 6: Implement Adapters
- `adapters/blis_adapter.py`:
  - Command building with all parameters
  - Robust output parsing with regex
  - Model info extraction
- `adapters/vidur_adapter.py`:
  - Command building
  - JSON output directory scanning
  - Metrics extraction from complex format

### Step 7: Implement Search Orchestrator
- `orchestrator.py`: Main search logic
  - Three-phase approach
  - Progress tracking
  - Result validation

### Step 8: Implement CLI and Output
- `main.py`:
  - Argument parsing with defaults
  - Progress display
  - Clean error handling
- `utils/result_formatter.py`:
  - JSON export
  - Human-readable summaries

### Step 9: Testing and Validation
- Unit tests for each component
- Integration tests with mock simulators
- End-to-end tests with real simulators
- Performance benchmarks

## Verification Plan

### Manual Testing Steps

1. **BLIS Search Test (Small Model):**
```bash
cd config_search
python main.py \
  --simulator blis \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload chatbot \
  --ttft-p90-ms 20 \
  --itl-p95-ms 10 \
  --e2e-p99-ms 500 \
  --blis-binary ../inference-sim/simulation_worker \
  --num-workers 4 \
  --max-iterations 50 \
  --output blis_8b_results.json
```

2. **Vidur Search Test (Large Model):**
```bash
python main.py \
  --simulator vidur \
  --model meta-llama/Llama-2-70b-hf \
  --workload summarization \
  --ttft-p90-ms 50 \
  --e2e-p99-ms 2000 \
  --vidur-path ../vidur \
  --num-workers 4 \
  --max-iterations 75 \
  --output vidur_70b_results.json
```

3. **Performance Comparison Test:**
```bash
# Run with optimizations
python main.py \
  --simulator blis \
  --model microsoft/phi-4 \
  --workload chatbot \
  --ttft-p90-ms 15 \
  --blis-binary ../inference-sim/simulation_worker \
  --use-bayesian-opt \
  --use-performance-predictor \
  --num-workers 8 \
  --output optimized_results.json

# Run without optimizations (baseline)
python main.py \
  --simulator blis \
  --model microsoft/phi-4 \
  --workload chatbot \
  --ttft-p90-ms 15 \
  --blis-binary ../inference-sim/simulation_worker \
  --use-grid-search \
  --num-workers 1 \
  --output baseline_results.json
```

4. **Validate Results:**
- Compare search times (should be ~10x faster with optimizations)
- Check configuration diversity in results
- Verify max QPS is similar or better with Bayesian optimization
- Test best config stability with multiple runs
- Ensure all latency constraints are satisfied

### Success Criteria

#### Functionality
- ✓ Successfully finds viable configurations for both simulators
- ✓ Handles simulation failures gracefully (timeout, crashes)
- ✓ Binary search converges adaptively based on QPS range
- ✓ Results are reproducible with same seed
- ✓ JSON output contains all required fields

#### Performance
- ✓ Search completes in <30 minutes for typical workloads
- ✓ Parallel execution shows linear speedup up to 4-8 workers
- ✓ Performance predictor eliminates >50% of infeasible configs
- ✓ Bayesian optimization finds good configs in <100 iterations

#### Quality
- ✓ Found configurations meet all latency constraints
- ✓ Max QPS within 5% across multiple validation runs
- ✓ Search explores diverse configuration space
- ✓ Results include top-10 alternative configurations

#### Robustness
- ✓ Handles models of different sizes (2B to 70B parameters)
- ✓ Works with all preset workload types
- ✓ Gracefully handles missing coefficients/models
- ✓ Provides helpful error messages for common issues

### Automated Testing

```python
# test_config_search.py
import unittest
from unittest.mock import Mock, patch

class TestConfigSearch(unittest.TestCase):
    def test_bayesian_optimizer_convergence(self):
        """Test that Bayesian optimizer converges to optimum"""
        # Mock simulator that has optimum at specific config
        # Verify optimizer finds it in reasonable iterations

    def test_performance_predictor_accuracy(self):
        """Test roofline model predictions"""
        # Compare predicted vs actual feasibility
        # Should correctly identify OOM configs

    def test_parallel_executor_failure_handling(self):
        """Test handling of simulation failures"""
        # Mock failing simulations
        # Verify retry logic and graceful degradation

    def test_adaptive_search_space(self):
        """Test search space adaptation for different models"""
        # Verify small models get different configs than large
        # Check context-aware parameter ranges
```

## Dependencies

### Python Requirements

**Core Dependencies:**
```
numpy>=1.21.0          # Array operations and math
scipy>=1.7.0           # Gaussian Process, optimization
scikit-learn>=1.0.0    # Latin Hypercube Sampling, utilities
matplotlib>=3.4.0      # Optimization visualization (optional)
```

**Standard Library:**
- `subprocess` - Run simulator commands with timeout
- `multiprocessing` - Parallel simulation execution
- `json` - Parse Vidur output, write results
- `argparse` - CLI argument parsing
- `typing` - Type hints
- `abc` - Abstract base classes
- `re` - Parse BLIS text output
- `pathlib` - File path handling
- `dataclasses` - Configuration objects
- `logging` - Structured logging
- `time` - Performance tracking
- `concurrent.futures` - Alternative to multiprocessing

### External Tools
- BLIS binary (pre-built: `inference-sim/simulation_worker`)
- Vidur Python module (existing: `vidur/`)
  - Note: Vidur has its own dependencies (pandas, wandb, etc.)
  - These are isolated to Vidur's environment

### Hardware Requirements
- CPU: 4+ cores recommended for parallel execution
- Memory: 8GB+ RAM for Gaussian Process models
- Disk: 1GB for caching results

## Critical Files

### To Create:
- `config_search/main.py` - Main entry point
- `config_search/orchestrator.py` - Search orchestrator
- `config_search/adapters/base.py` - Abstract adapter
- `config_search/adapters/blis_adapter.py` - BLIS adapter
- `config_search/adapters/vidur_adapter.py` - Vidur adapter
- `config_search/search/binary_search.py` - Binary search logic
- `config_search/search/grid_search.py` - Grid search logic
- `config_search/config/search_space.py` - Search space definitions
- `config_search/config/constraints.py` - Constraint validation
- `config_search/utils/metrics_parser.py` - Metrics parsing
- `config_search/README.md` - Usage documentation

### To Read/Reference:
- `inference-sim/CLI_REFERENCE.md` - BLIS CLI options
- `vidur/CLI_REFERENCE.md` - Vidur CLI options
- `inference-sim/coefficients.yaml` - Model/hardware combinations
- `vidur/vidur/metrics/metrics_store.py` - Vidur output format

## Error Handling and Edge Cases

### Common Failure Modes

1. **Simulation Timeouts**:
   - Default 5-minute timeout per simulation
   - Automatically retry with 2x timeout on first failure
   - Mark config as infeasible after 3 failures

2. **Out of Memory**:
   - Detect OOM from stderr patterns
   - Reduce batch size and retry
   - Skip larger batch sizes for that hardware

3. **Missing Model Support**:
   - Check coefficients.yaml for BLIS support
   - Fallback to roofline model if coefficients missing
   - Warn user about reduced accuracy

4. **Invalid Configurations**:
   - TP > number of GPUs available
   - Batch size > model max length
   - Memory requirements > device capacity
   - Pre-filter these before simulation

5. **Parsing Failures**:
   - BLIS: Regex patterns for each metric with fallbacks
   - Vidur: Try multiple output directory patterns
   - Return partial metrics if some parsing fails

### Recovery Strategies

```python
def run_with_retry(adapter, config, qps, max_retries=3):
    timeout = 300  # 5 minutes

    for attempt in range(max_retries):
        try:
            metrics = adapter.run_simulation(
                adapter.build_command(config, qps),
                timeout=timeout * (attempt + 1)  # Increase timeout
            )
            if metrics:
                return metrics

        except SimulationError as e:
            if "out of memory" in str(e).lower():
                # Try with reduced batch size
                config = reduce_batch_size(config)
                if not config:
                    return None
            elif "timeout" in str(e).lower():
                logging.warning(f"Timeout on attempt {attempt + 1}")
                continue
            else:
                logging.error(f"Unknown error: {e}")

    return None  # All retries failed

def reduce_batch_size(config):
    """Reduce batch-related parameters by 50%"""
    reducible_params = [
        'max_num_running_reqs', 'batch_size_cap',
        'max_num_scheduled_tokens', 'max_tokens_in_batch'
    ]

    for param in reducible_params:
        if param in config:
            config[param] = max(1, config[param] // 2)

    return config if any(config.get(p, 0) > 16 for p in reducible_params) else None
```

### Edge Case Handling

1. **Empty Search Space**:
   - All configs filtered by performance predictor
   - Relax constraints and warn user
   - Suggest alternative hardware/models

2. **All Simulations Fail**:
   - Check simulator installation
   - Verify model names match expected format
   - Test with known-good configuration

3. **Convergence Issues**:
   - Bayesian optimizer stuck in local optimum
   - Inject random exploration configs
   - Increase initial sampling size

4. **Resource Exhaustion**:
   - Monitor system memory during search
   - Reduce parallel workers if needed
   - Checkpoint results periodically

## Implemented Optimizations

1. ✓ **Parallel Simulation Execution**: 4-8 concurrent simulations
2. ✓ **Caching**: Results cache prevents duplicate runs
3. ✓ **Smart Initialization**: Adaptive initial QPS based on model/hardware
4. ✓ **Bayesian Optimization**: Replaces naive grid search
5. ✓ **Performance Prediction**: Roofline model filters infeasible configs
6. ✓ **Adaptive Search Space**: Model-aware parameter ranges
7. ✓ **Robust Error Handling**: Timeouts and retry mechanisms

## Future Enhancements

1. **Multi-Objective Optimization**:
   - Pareto frontier for cost vs performance
   - Consider memory usage, power, and latency together
   - Multi-fidelity optimization (quick vs accurate simulations)

2. **Transfer Learning**:
   - Use results from similar models to warm-start search
   - Build meta-model across multiple workloads
   - Online learning from production metrics

3. **Advanced Scheduling**:
   - Predict optimal scheduler type from workload
   - Auto-tune scheduler parameters
   - Hybrid scheduling strategies

4. **Cloud Integration**:
   - Deploy search on cloud with auto-scaling
   - Cost-aware configuration search
   - Integration with cloud provider APIs

5. **Real-time Adaptation**:
   - Continuous optimization during production
   - A/B testing framework for configs
   - Automatic rollback on performance degradation

6. **Visualization Dashboard**:
   - Real-time search progress visualization
   - Interactive parameter exploration
   - Performance surface plots
