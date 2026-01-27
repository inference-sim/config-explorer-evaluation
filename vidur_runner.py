"""
Vidur Runner - Run Vidur simulations with configuration

This module provides a Vidur-specific interface matching the BLIS runner API.

Key components:
- run_vidur(): Run single simulation at specific QPS (equivalent to run_blis)
"""
import os
import subprocess
import sys
import shutil
import time
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import glob

from capacity_planner import calculate_total_kv_blocks


def generate_synthetic_trace(config: Dict, output_path: str, qps: float):
    """
    Generate a synthetic trace file from workload parameters with constant inter-arrival times.

    Creates a CSV file with format: arrived_at,num_prefill_tokens,num_decode_tokens

    Args:
        config: Config with workload parameters (prompt_tokens, output_tokens, etc.)
        output_path: Path to write trace CSV
        qps: Queries per second (for constant inter-arrival time)
    """
    import numpy as np

    num_requests = config.get('num_requests', 1000)
    inter_arrival_time = 1.0 / qps  # Constant spacing

    # Get workload parameters with defaults
    prompt_mean = config.get('prompt_tokens', 512)
    prompt_std = config.get('prompt_tokens_stdev', 100)
    prompt_min = config.get('prompt_tokens_min', 1)
    prompt_max = config.get('prompt_tokens_max', 2048)

    output_mean = config.get('output_tokens', 128)
    output_std = config.get('output_tokens_stdev', 50)
    output_min = config.get('output_tokens_min', 1)
    output_max = config.get('output_tokens_max', 512)

    # Generate arrival times (constant spacing)
    arrival_times = np.arange(num_requests) * inter_arrival_time

    # Generate token counts from normal distribution, clipped to min/max
    np.random.seed(42)  # Reproducibility
    prompt_tokens = np.random.normal(prompt_mean, prompt_std, num_requests)
    prompt_tokens = np.clip(prompt_tokens, prompt_min, prompt_max).astype(int)

    output_tokens = np.random.normal(output_mean, output_std, num_requests)
    output_tokens = np.clip(output_tokens, output_min, output_max).astype(int)

    # Write to CSV with arrival times (Vidur format)
    with open(output_path, 'w') as f:
        f.write('arrived_at,num_prefill_tokens,num_decode_tokens\n')
        for i in range(num_requests):
            f.write(f'{arrival_times[i]:.6f},{prompt_tokens[i]},{output_tokens[i]}\n')


def parse_request_metrics_csv(csv_path: str) -> Optional[Dict]:
    """
    Parse Vidur's request_metrics.csv and extract key metrics.

    Columns in request_metrics.csv:
    - prefill_time_execution_plus_preemption: TTFT (execution time only, excludes scheduling delay) in seconds
    - decode_time_execution_plus_preemption_normalized: TPOT/ITL in seconds per token
    - request_e2e_time: E2E latency in seconds
    - request_scheduling_delay: Scheduling delay in seconds

    Args:
        csv_path: Path to request_metrics.csv

    Returns:
        Dictionary with metrics in BLIS-compatible format (milliseconds)
    """
    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"Warning: Empty CSV file: {csv_path}", file=sys.stderr)
            return None

        # Extract metrics and convert seconds to milliseconds for compatibility with BLIS
        metrics = {
            # TTFT (Time to First Token) - use execution time only
            'ttft_mean_ms': df['prefill_time_execution_plus_preemption'].mean() * 1000,
            'ttft_p90_ms': df['prefill_time_execution_plus_preemption'].quantile(0.90) * 1000,
            'ttft_p95_ms': df['prefill_time_execution_plus_preemption'].quantile(0.95) * 1000,
            'ttft_p99_ms': df['prefill_time_execution_plus_preemption'].quantile(0.99) * 1000,

            # ITL/TPOT (Inter-Token Latency / Time Per Output Token)
            'itl_mean_ms': df['decode_time_execution_plus_preemption_normalized'].mean() * 1000,
            'itl_p90_ms': df['decode_time_execution_plus_preemption_normalized'].quantile(0.90) * 1000,
            'itl_p95_ms': df['decode_time_execution_plus_preemption_normalized'].quantile(0.95) * 1000,
            'itl_p99_ms': df['decode_time_execution_plus_preemption_normalized'].quantile(0.99) * 1000,

            # E2E Latency
            'e2e_mean_ms': df['request_e2e_time'].mean() * 1000,
            'e2e_p90_ms': df['request_e2e_time'].quantile(0.90) * 1000,
            'e2e_p95_ms': df['request_e2e_time'].quantile(0.95) * 1000,
            'e2e_p99_ms': df['request_e2e_time'].quantile(0.99) * 1000,

            # Scheduling Delay
            'scheduling_delay_mean_ms': df['request_scheduling_delay'].mean() * 1000,
            'scheduling_delay_p90_ms': df['request_scheduling_delay'].quantile(0.90) * 1000,
            'scheduling_delay_p95_ms': df['request_scheduling_delay'].quantile(0.95) * 1000,
            'scheduling_delay_p99_ms': df['request_scheduling_delay'].quantile(0.99) * 1000,

            # Request stats
            'total_requests': len(df),
            'completed_requests': len(df),
            'failed_requests': 0,  # Vidur doesn't track failures in same way
        }

        # Calculate throughput (requests per second)
        if 'completed_at' in df.columns and 'arrived_at' in df.columns:
            total_duration = df['completed_at'].max() - df['arrived_at'].min()
            if total_duration > 0:
                metrics['responses_per_sec'] = len(df) / total_duration

            # Token throughput
            if 'request_num_tokens' in df.columns:
                total_tokens = df['request_num_tokens'].sum()
                metrics['tokens_per_sec'] = total_tokens / total_duration

        # Validate metrics - check if we got valid latency data
        # If all key metrics are 0 or NaN, the simulation didn't produce valid results
        import math
        key_metrics = ['e2e_p95_ms', 'ttft_p90_ms', 'itl_mean_ms']
        valid_data = False
        for metric_key in key_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                if not math.isnan(value) and value > 0:
                    valid_data = True
                    break

        if not valid_data:
            print(f"Warning: Invalid metrics (all zeros or NaN) in {csv_path}", file=sys.stderr)
            return None

        return metrics

    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error parsing CSV {csv_path}: {e}", file=sys.stderr)
        return None


def run_vidur(
    config: Dict,
    qps: float,
    trace_file: Optional[str] = None,
    num_requests: Optional[int] = None,
    timeout: int = 300,
    verbose: bool = True
) -> Dict:
    """
    Run Vidur simulation with given configuration and QPS.

    Args:
        config: Configuration dictionary containing:
            - model: Model identifier
            - hardware: Hardware type (e.g., 'H100')
            - tp: Tensor parallelism degree
            - batch_size: Maximum number of concurrent requests
            - max_scheduled_tokens: Maximum tokens per iteration
            - max_model_len: Maximum sequence length
            - gpu_memory_utilization: GPU memory utilization fraction
            - block_size: Block size in tokens
            - num_requests: Number of requests to simulate (optional, default: 1000)
        qps: Queries per second (arrival rate)
        trace_file: Optional path to trace file (CSV with arrived_at,num_prefill_tokens,num_decode_tokens)
        num_requests: Number of requests to simulate (optional, overrides config value)
        timeout: Timeout in seconds (default: 300)
        verbose: If True, print detailed simulation info (default: True)

    Returns:
        Dictionary with simulation results including:
        - ttft_p90_ms: 90th percentile time to first token
        - itl_p95_ms: 95th percentile inter-token latency
        - e2e_p90_ms: 90th percentile end-to-end latency
        - etc.
        Or None if simulation failed
    """
    # Get num_requests from config if not provided as argument
    if num_requests is None:
        num_requests = config.get('num_requests', 1000)

    # Create local tmp directory in current path
    local_tmp = Path('./tmp').resolve()  # Use absolute path
    local_tmp.mkdir(exist_ok=True)

    # Create unique subdirectory for this simulation (include PID for parallel safety)
    timestamp = int(time.time() * 1000000)  # microsecond precision
    pid = os.getpid()
    temp_dir = local_tmp / f'vidur_sim_{timestamp}_{pid}'
    temp_dir.mkdir(exist_ok=True)
    output_dir = str(temp_dir)  # Already absolute since local_tmp is absolute

    # Map hardware names
    device_map = {
        'H100': 'h100',
        'A100': 'a100',
        'A40': 'a40',
    }
    device = device_map.get(config['hardware'], config['hardware'].lower())

    # Generate trace file if not provided
    if not trace_file:
        actual_trace_file = os.path.join(output_dir, f'synthetic_trace_qps{qps}.csv')
        # Update config num_requests for trace generation
        trace_config = {**config, 'num_requests': num_requests}
        generate_synthetic_trace(trace_config, actual_trace_file, qps=qps)
    else:
        # Convert trace_file to absolute path if relative
        actual_trace_file = str(Path(trace_file).resolve())

    # Check if Vidur is installed as a package or available in subdirectory
    vidur_dir = Path(__file__).parent / 'vidur'
    vidur_main = vidur_dir / 'vidur' / 'main.py'

    if not vidur_main.exists():
        print(f"Error: Vidur not found at {vidur_main}", file=sys.stderr)
        print(f"Please ensure vidur/ subdirectory is present", file=sys.stderr)
        return None

    # Try to import vidur to see if Vidur is properly installed
    vidur_installed = False
    try:
        import vidur.main
        vidur_installed = True
    except ImportError:
        pass

    # Initialize environment variable (will be set if vidur not installed)
    run_env = None

    # Build vidur.main command with explicit arguments
    if vidur_installed:
        base_cmd = [sys.executable, '-m', 'vidur.main']
    else:
        if verbose:
            print(f"Note: Vidur not installed as package, running from {vidur_dir}")

        run_env = os.environ.copy()
        pythonpath = str(vidur_dir)
        if 'PYTHONPATH' in run_env:
            pythonpath = f"{pythonpath}:{run_env['PYTHONPATH']}"
        run_env['PYTHONPATH'] = pythonpath

        base_cmd = [sys.executable, str(vidur_main)]

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

    if verbose:
        print(f"  Calculated total_kv_blocks: {total_kv_blocks:,}")

    # Build full command with all required arguments
    cmd = base_cmd + [
        # Model and hardware
        '--replica_config_model_name', config['model'],
        '--replica_config_device', device,
        '--replica_config_tensor_parallel_size', str(config.get('tp', 1)),
        '--cluster_config_num_replicas', '1',

        # Scheduler
        '--replica_scheduler_config_type', 'vllm',
        '--vllm_scheduler_config_max_tokens_in_batch', str(config.get('max_scheduled_tokens', 8192)),
        '--vllm_scheduler_config_num_blocks', str(total_kv_blocks),
        '--vllm_scheduler_config_watermark_blocks_fraction', '0.0',

        # Random Forest execution time predictor
        '--random_forrest_execution_time_predictor_config_prediction_max_batch_size', str(config.get('batch_size', 256)),

        # Request generator - use trace_replay for both lengths AND arrival times
        '--length_generator_config_type', 'trace',
        '--request_generator_config_type', 'trace_replay',
        '--trace_request_generator_config_trace_file', actual_trace_file,
        '--trace_request_generator_config_max_tokens', str(config.get('max_model_len', 4096)),

        # Output
        '--metrics_config_output_dir', output_dir,

        # Disable unnecessary features for speed
        '--no-metrics_config_save_table_to_wandb',
        '--no-metrics_config_store_plots',
        '--no-metrics_config_enable_chrome_trace',
    ]

    if verbose:
        print(f"\nRunning Vidur simulation:")
        print(f"  QPS: {qps}")
        print(f"  Num Requests: {num_requests}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Max Scheduled Tokens: {config['max_scheduled_tokens']}")
        print(f"  Max Model Length: {config['max_model_len']}")
        print(f"  Total KV Blocks: {total_kv_blocks:,}")
        print(f"  Command: {' '.join(cmd)}\n")

    # Run Vidur simulation
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            cwd=vidur_dir,  # Run from vidur directory to access ./data/profiling
            env=run_env
        )

        if result.returncode != 0:
            if verbose:
                print(f"Vidur simulation failed with return code {result.returncode}", file=sys.stderr)
                print(f"STDERR: {result.stderr}", file=sys.stderr)
            return None

    except subprocess.TimeoutExpired:
        print(f"Vidur simulation timed out after {timeout}s", file=sys.stderr)
        # Clean up temporary directory
        try:
            shutil.rmtree(str(temp_dir))
        except:
            pass
        return None
    except Exception as e:
        print(f"Error running Vidur simulation: {e}", file=sys.stderr)
        # Clean up temporary directory
        try:
            shutil.rmtree(str(temp_dir))
        except:
            pass
        return None

    # Parse results from output directory
    # Find request_metrics.csv in timestamped subdirectory
    glob_pattern = f"{output_dir}/*/request_metrics.csv"
    metrics_files = glob.glob(glob_pattern)

    if not metrics_files:
        print(f"Error: No request_metrics.csv found in {output_dir}", file=sys.stderr)
        # Clean up temporary directory
        try:
            shutil.rmtree(str(temp_dir))
        except:
            pass
        return None

    # Parse the metrics
    metrics_file = metrics_files[0]
    metrics = parse_request_metrics_csv(metrics_file)

    if metrics:
        metrics['qps'] = qps

    # Clean up temporary directory after parsing results
    try:
        shutil.rmtree(str(temp_dir))
    except Exception as e:
        if verbose:
            print(f"Warning: Could not clean up {temp_dir}: {e}", file=sys.stderr)

    return metrics
