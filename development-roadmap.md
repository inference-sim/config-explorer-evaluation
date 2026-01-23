# Configuration Search Tool - Development Roadmap

## Overview
This document breaks down the comprehensive plan in `config-search-plan.md` into sequential, manageable development steps. Each step builds on the previous one and adds clear, testable capabilities.

---

## Step 1: Basic Infrastructure & Single Simulation
**Goal:** Establish foundation to run a single simulator and parse results

### Work Blocks:
1. **Project Structure**
   - Create directory layout (`config_search/` with subdirectories)
   - Set up Python virtual environment
   - Create `requirements.txt` with core dependencies (numpy, scipy, scikit-learn)

2. **Base Adapter Interface** (`adapters/base.py`)
   - Abstract `SimulatorAdapter` class
   - Methods: `build_command()`, `run_simulation()`, `parse_output()`
   - Basic timeout handling (subprocess with timeout)

3. **BLIS Adapter** (`adapters/blis_adapter.py`)
   - Implement command building for BLIS CLI
   - Parse BLIS text output (regex patterns for TTFT, ITL, E2E metrics)
   - Extract model info from coefficients.yaml

4. **Vidur Adapter** (`adapters/vidur_adapter.py`)
   - Implement command building for Vidur CLI
   - Find latest output directory
   - Parse JSON metrics from Vidur output

### Capability Added:
✅ Can run BLIS or Vidur with a given configuration and get back metrics as a Python dict

### Test:
```bash
python -c "from adapters.blis_adapter import BlisAdapter; \
adapter = BlisAdapter('./inference-sim/simulation_worker'); \
config = {'model': 'meta-llama/llama-3.1-8b-instruct', 'hardware': 'H100', 'tp': 1, ...}; \
metrics = adapter.run_simulation(adapter.build_command(config, qps=5.0)); \
print(metrics)"
```

---

## Step 2: Binary Search for Max QPS
**Goal:** Find the maximum QPS a single configuration can handle while meeting latency constraints

### Work Blocks:
1. **Constraint Validator** (`config/constraints.py`)
   - `meets_constraints()` function
   - Check if metrics satisfy latency thresholds (TTFT P90, ITL P95, etc.)
   - Handle missing metrics gracefully

2. **Binary Search Implementation** (`search/binary_search.py`)
   - `binary_search_qps(config, adapter, constraints)` function
   - Adaptive convergence threshold (1% of current QPS or 0.5 req/s)
   - Early stopping if fails at low QPS
   - Return maximum viable QPS

3. **Basic CLI** (`main.py`)
   - Argument parsing (simulator, model, workload, constraints)
   - Run binary search on single config
   - Print results

### Capability Added:
✅ Given a configuration, automatically find the maximum QPS it can sustain

### Test:
```bash
python main.py \
  --simulator blis \
  --model meta-llama/llama-3.1-8b-instruct \
  --hardware H100 --tp 2 \
  --max-num-running-reqs 256 \
  --ttft-p90-ms 20 \
  --output single_config_result.json
```

---

## Step 3: Search Space & Multi-Config Evaluation
**Goal:** Define search spaces and evaluate multiple configurations sequentially

### Work Blocks:
1. **Search Space Definitions** (`config/search_space.py`)
   - `get_blis_search_space(model, workload)` - returns dict of parameter ranges
   - `get_vidur_search_space(model, workload)` - returns dict of parameter ranges
   - Adaptive ranges based on model size (use `get_model_size()` helper)
   - Adaptive ranges based on workload context length

2. **Configuration Generator** (`search/grid_search.py`)
   - Generate all combinations from search space (itertools.product)
   - `generate_configs(search_space)` - returns list of config dicts

3. **Sequential Evaluator** (`orchestrator.py` - basic version)
   - Loop through configs
   - Run binary search for each
   - Track best config and QPS
   - Simple progress logging

### Capability Added:
✅ Can search across multiple configurations to find the best one (sequential evaluation)

### Test:
```bash
python main.py \
  --simulator blis \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload chatbot \
  --ttft-p90-ms 20 \
  --search-mode grid \
  --output grid_search_results.json
```

---

## Step 4: Parallel Execution
**Goal:** Speed up search by evaluating multiple configurations concurrently

### Work Blocks:
1. **Parallel Executor** (`parallel/executor.py`)
   - `ParallelSimulationExecutor` class using `multiprocessing.Pool`
   - `submit(config, qps, adapter)` - submit job to pool
   - `collect_results()` - gather completed jobs
   - Results caching by config hash

2. **Job Manager** (`parallel/job_manager.py`)
   - Retry logic with exponential backoff
   - `run_simulation_with_retry()` wrapper
   - Handle timeout and OOM errors

3. **Update Orchestrator** (`orchestrator.py`)
   - Replace sequential loop with parallel submission
   - Submit batches of N configs (N = num_workers)
   - Collect results as they complete

### Capability Added:
✅ 4-8x speedup through concurrent simulation execution

### Test:
```bash
python main.py \
  --simulator blis \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload chatbot \
  --ttft-p90-ms 20 \
  --search-mode grid \
  --num-workers 4 \
  --output parallel_results.json

# Compare runtime to sequential version
```

---

## Step 5: Performance Prediction & Filtering
**Goal:** Pre-filter infeasible configurations to avoid wasting simulation time

### Work Blocks:
1. **Hardware Specifications** (`config/hardware_specs.py`)
   - Device memory, TFLOPS for [H100, A100-SXM, A40, etc.]
   - `get_device_memory(hardware)`, `get_device_tflops(hardware)`

2. **Performance Predictor** (`search/performance_predictor.py`)
   - `PerformancePredictor` class
   - `estimate_kv_cache_size(model_params, seq_len, batch_size)` - memory estimation
   - `estimate_tflops(model_params, batch_size)` - compute estimation
   - `is_feasible(config, model_info)` - roofline-based feasibility check
   - Memory check: model + KV cache < 90% device memory
   - Compute check: required TFLOPS < device capacity

3. **Update Orchestrator**
   - Filter configs through predictor before submission
   - Log how many configs were filtered
   - Track filtering accuracy (false positives/negatives)

### Capability Added:
✅ Skip 50%+ of infeasible configs, reducing search time significantly

### Test:
```bash
python main.py \
  --simulator blis \
  --model meta-llama/llama-3.3-70b-instruct \
  --workload multidoc \
  --ttft-p90-ms 50 \
  --use-performance-predictor \
  --output filtered_results.json

# Check log: "Filtered to X feasible configs from Y total"
```

---

## Step 6: Bayesian Optimization
**Goal:** Intelligently explore configuration space instead of exhaustive grid search

### Work Blocks:
1. **Latin Hypercube Sampling** (`search/sampling.py`)
   - `latin_hypercube_sample(search_space, n_samples)` using scikit-learn
   - Generate initial diverse configs (20-30 samples)

2. **Gaussian Process Model** (`search/bayesian_optimizer.py`)
   - `BayesianOptimizer` class
   - Fit GP model on (config features → max QPS) using scipy
   - Features: hardware specs, TP, batch size, scheduler params
   - `predict(config)` - returns (mean, std) for expected QPS

3. **Acquisition Function**
   - Expected Improvement (EI) implementation
   - `suggest_next(results_history)` - proposes next config to evaluate
   - Balance exploration vs exploitation

4. **Update Orchestrator**
   - Phase 1: Evaluate initial Latin Hypercube samples
   - Phase 2: Iteratively suggest configs via acquisition function
   - Update GP model after each evaluation
   - Convergence criteria (iterations or time limit)

### Capability Added:
✅ Find near-optimal configurations in 50-100 evaluations instead of 1000+ grid points

### Test:
```bash
python main.py \
  --simulator blis \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload chatbot \
  --ttft-p90-ms 20 \
  --search-mode bayesian \
  --max-iterations 75 \
  --output bayesian_results.json

# Compare to grid search: similar max QPS, much fewer configs evaluated
```

---

## Step 7: Polish & Production-Ready Features
**Goal:** Robustness, validation, output quality, and user experience

### Work Blocks:
1. **Enhanced Error Handling** (`utils/timeout_manager.py`)
   - Detect OOM from stderr patterns
   - Auto-reduce batch size and retry on OOM
   - Timeout escalation (increase timeout on retry)
   - Process cleanup (kill zombie processes)

2. **Result Validation** (update `orchestrator.py`)
   - `_validate_best_config()` - run best config 3 times
   - Average metrics across validation runs
   - Warn if validation fails (use original metrics)
   - Safety margin: return QPS * 0.95 if validation close to threshold

3. **Output Formatting** (`utils/result_formatter.py`)
   - JSON output with all details (see schema in plan)
   - Human-readable console summary
   - Top-10 alternative configs
   - Search metadata (time, configs tested, convergence)

4. **Comprehensive CLI** (enhance `main.py`)
   - All constraint options (TTFT P90/P95, ITL P95, E2E P99)
   - Search mode selection (grid/bayesian)
   - Time limits and iteration caps
   - Verbose logging option
   - Resume from checkpoint (optional)

5. **Documentation** (`README.md`)
   - Installation instructions
   - Usage examples for both simulators
   - Explanation of constraints
   - Interpretation of results
   - Troubleshooting common errors

### Capability Added:
✅ Production-ready tool with robust error handling, validation, and great UX

### Test:
```bash
# Full end-to-end test
python main.py \
  --simulator blis \
  --model meta-llama/llama-3.1-70b-instruct \
  --workload summarization \
  --ttft-p90-ms 30 \
  --itl-p95-ms 15 \
  --e2e-p99-ms 1000 \
  --search-mode bayesian \
  --use-performance-predictor \
  --num-workers 8 \
  --max-iterations 100 \
  --time-limit 1800 \
  --output final_results.json

# Verify:
# - Search completes in <30 minutes
# - Best config validated 3x
# - JSON output well-formatted
# - Console output clear and informative
# - Handles failures gracefully
```

---

## Step 8 (Optional): Advanced Features
**Goal:** Nice-to-have enhancements for power users

### Work Blocks:
1. **Multi-Objective Optimization**
   - Consider cost, memory, and latency together
   - Pareto frontier visualization

2. **Transfer Learning**
   - Save GP models across searches
   - Warm-start from similar model/workload

3. **Visualization Dashboard**
   - Real-time search progress
   - Parameter sensitivity plots
   - Performance surface visualization

### Capability Added:
✅ Advanced capabilities for research and power users

---

## Development Timeline Estimate

| Step | Effort | Dependencies |
|------|--------|--------------|
| 1. Infrastructure | 2-3 days | None |
| 2. Binary Search | 1-2 days | Step 1 |
| 3. Search Space | 1-2 days | Step 2 |
| 4. Parallel Execution | 2-3 days | Step 3 |
| 5. Performance Prediction | 2-3 days | Step 3 |
| 6. Bayesian Optimization | 3-4 days | Steps 4-5 |
| 7. Polish | 2-3 days | Steps 1-6 |
| **Total** | **13-20 days** | |

---

## Testing Strategy

### Per-Step Testing
Each step includes a test command to verify the new capability works in isolation.

### Integration Testing
After Step 7, run full end-to-end tests:
- Small model (8B) on chatbot workload
- Large model (70B) on multidoc workload
- Both BLIS and Vidur simulators
- Compare grid vs Bayesian search
- Verify performance predictor accuracy

### Success Metrics
- Search completes in <30 minutes (vs 32+ hours naive approach)
- Finds configs within 5% of optimal QPS
- >50% of infeasible configs filtered by predictor
- Handles failures gracefully (timeouts, OOM)
- Results reproducible with same seed

---

## Key Simplifications from Original Plan

1. **Incremental Capability**: Each step adds one clear capability, making progress tangible
2. **Test-Driven**: Every step has a concrete test command
3. **Parallel Development**: Steps 4-5 can be developed concurrently (both depend on Step 3)
4. **Defer Complexity**: Advanced features (Step 8) moved to optional phase
5. **Clear Dependencies**: Each step's dependencies explicitly stated

This roadmap transforms the comprehensive but complex original plan into an actionable, sequential development process.
