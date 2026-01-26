"""
BLIS Runner - Run BLIS simulations with configuration
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from capacity_planner import calculate_total_kv_blocks


def run_blis(
    config: Dict,
    qps: float,
    trace_file: Optional[str] = None,
    num_requests: Optional[int] = None,
    timeout: int = 300,
    verbose: bool = True
) -> Dict:
    """
    Run BLIS simulation with given configuration and QPS.

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
            - vllm_version: vLLM Docker image version (optional, default: 'vllm/vllm-openai:v0.8.4')
            - num_requests: Number of requests to simulate (optional, default: 500)
            - model_config_folder_base: Base path for model configs (optional, enables roofline model)
            - hardware_config: Path to hardware config JSON file (optional, enables roofline model)
                              Note: model_config_folder is automatically constructed as base/{model_name}
            - prefix_tokens: Number of prefix tokens (optional, default: 0)
            - prompt_tokens: Mean prompt tokens (optional, for distribution workload)
            - prompt_tokens_stdev: Prompt tokens std dev (optional)
            - prompt_tokens_min: Minimum prompt tokens (optional, default: 2)
            - prompt_tokens_max: Maximum prompt tokens (optional, default: 7000)
            - output_tokens: Mean output tokens (optional, for distribution workload)
            - output_tokens_stdev: Output tokens std dev (optional)
            - output_tokens_min: Minimum output tokens (optional, default: 2)
            - output_tokens_max: Maximum output tokens (optional, default: 7000)
            - slos: List of SLO constraints (optional, for qps_search)
                    e.g., [{"metric": "e2e_p95_ms", "threshold_ms": 1000}]
        qps: Queries per second (arrival rate)
        trace_file: Optional path to trace file (CSV with prompt_tokens,output_tokens)
        num_requests: Number of requests to simulate (optional, overrides config value, default: 500)
        timeout: Timeout in seconds (default: 300)
        verbose: If True, print detailed simulation info (default: True)

    Returns:
        Dictionary with simulation results including:
        - ttft_p90_ms: 90th percentile time to first token
        - itl_p95_ms: 95th percentile inter-token latency
        - e2e_p90_ms: 90th percentile end-to-end latency
        - throughput_qps: Achieved throughput
        - etc.
    """
    # Get num_requests from config if not provided as argument
    if num_requests is None:
        num_requests = config.get('num_requests', 500)

    # Calculate total_kv_blocks from max_model_len and gpu_memory_utilization
    total_kv_blocks = calculate_total_kv_blocks(
        model=config['model'],
        hardware=config['hardware'],
        tp=config['tp'],
        max_model_len=config['max_model_len'],
        gpu_memory_utilization=config['gpu_memory_utilization'],
        block_size=config.get('block_size', 16)
    )

    total_kv_blocks = int(total_kv_blocks * 0.8)

    # Build command
    blis_binary = Path("/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/config-explorer-evaluation/simulation_worker")
    defaults_file = Path("/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/config-explorer-evaluation/inference-sim/defaults.yaml")

    if not blis_binary.exists():
        raise FileNotFoundError(
            f"BLIS binary not found at {blis_binary}. "
            f"Please build it first: cd inference-sim && go build -o ../simulation_worker main.go"
        )

    if not defaults_file.exists():
        raise FileNotFoundError(
            f"defaults.yaml not found at {defaults_file}. "
            f"Please create it by combining coefficients.yaml and workloads.yaml"
        )

    # Get vllm_version from config, default to v0.8.4
    vllm_version = config.get('vllm_version', 'vllm/vllm-openai:v0.8.4')

    cmd = [
        str(blis_binary), 'run',
        '--model', config['model'],
        '--hardware', config['hardware'],
        '--tp', str(config['tp']),
        '--vllm-version', vllm_version,
        '--rate', str(qps),
        '--max-prompts', str(num_requests),
        '--max-num-running-reqs', str(config['batch_size']),
        '--max-num-scheduled-tokens', str(config['max_scheduled_tokens']),
        '--max-model-len', str(config['max_model_len']),
        '--total-kv-blocks', str(total_kv_blocks),
        '--block-size-in-tokens', str(config.get('block_size', 16)),
        '--defaults-filepath', str(defaults_file),
        '--log', 'error',  # Reduce log verbosity
    ]

    # Add model config folder and hardware config for roofline
    # Model config folder is constructed from model name: model.split("/")[1].lower()
    if 'hardware_config' in config:
        # Extract model folder name from model identifier
        model_folder_name = config['model'].split("/")[1].lower()

        # Get model config base path from config
        if 'model_config_folder_base' in config:
            model_config_base = Path(config['model_config_folder_base'])
        else:
            # Default to relative path if not specified
            model_config_base = Path(__file__).parent / "model_configs"

        model_config_folder = model_config_base / model_folder_name

        cmd.extend(['--model-config-folder', str(model_config_folder)])
        cmd.extend(['--hardware-config', config['hardware_config']])

    # Add trace file or use distribution
    if trace_file:
        cmd.extend(['--workload', 'traces'])
        cmd.extend(['--workload-traces-filepath', trace_file])
    else:
        # Use distribution workload with default parameters
        cmd.extend(['--workload', 'distribution'])
        if 'prefix_tokens' in config:
            cmd.extend(['--prefix-tokens', str(config['prefix_tokens'])])
        if 'prompt_tokens' in config:
            cmd.extend(['--prompt-tokens', str(config['prompt_tokens'])])
        if 'prompt_tokens_stdev' in config:
            cmd.extend(['--prompt-tokens-stdev', str(config['prompt_tokens_stdev'])])
        if 'prompt_tokens_min' in config:
            cmd.extend(['--prompt-tokens-min', str(config['prompt_tokens_min'])])
        if 'prompt_tokens_max' in config:
            cmd.extend(['--prompt-tokens-max', str(config['prompt_tokens_max'])])
        if 'output_tokens' in config:
            cmd.extend(['--output-tokens', str(config['output_tokens'])])
        if 'output_tokens_stdev' in config:
            cmd.extend(['--output-tokens-stdev', str(config['output_tokens_stdev'])])
        if 'output_tokens_min' in config:
            cmd.extend(['--output-tokens-min', str(config['output_tokens_min'])])
        if 'output_tokens_max' in config:
            cmd.extend(['--output-tokens-max', str(config['output_tokens_max'])])

    if verbose:
        print(f"\nRunning BLIS simulation:")
        print(f"  QPS: {qps}")
        print(f"  Num Requests: {num_requests}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Max Scheduled Tokens: {config['max_scheduled_tokens']}")
        print(f"  Max Model Length: {config['max_model_len']}")
        print(f"  GPU Memory Utilization: {config['gpu_memory_utilization']}")
        print(f"  Total KV Blocks: {total_kv_blocks}")
        print(f"  Command: {' '.join(cmd)}\n")

    # Run simulation
    try:
        # print(cmd)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False  # Don't raise on non-zero exit
        )

        # Check for errors
        if result.returncode != 0:
            print(f"BLIS simulation failed with return code {result.returncode}", file=sys.stderr)
            print(f"STDERR: {result.stderr}", file=sys.stderr)
            print(f"STDOUT: {result.stdout}", file=sys.stderr)
            return None

        # Parse JSON output
        # BLIS outputs metrics after "=== Simulation Metrics ===" header
        output = result.stdout.strip()

        # Find the JSON block after the header
        metrics = {}
        if '=== Simulation Metrics ===' in output:
            # Split by the header and take the part after it
            json_part = output.split('=== Simulation Metrics ===')[1].strip()
            try:
                metrics = json.loads(json_part)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON from BLIS output: {e}", file=sys.stderr)
                print(f"JSON part: {json_part[:500]}...", file=sys.stderr)
                return None
        else:
            # Try to find JSON in the output (fallback)
            for line in output.split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        metrics = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

        if not metrics:
            print(f"Warning: Could not find metrics in BLIS output", file=sys.stderr)
            print(f"Output: {output[:500]}...", file=sys.stderr)
            return None

        # Add calculated total_kv_blocks to results
        metrics['total_kv_blocks'] = total_kv_blocks
        metrics['qps'] = qps

        return metrics

    except subprocess.TimeoutExpired:
        print(f"BLIS simulation timed out after {timeout}s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error running BLIS simulation: {e}", file=sys.stderr)
        return None


def main():
    """
    Example usage of BLIS runner.
    """
    # Example configuration
    config = {
        'model': 'meta-llama/llama-3.1-8b-instruct',
        'hardware': 'H100',
        'tp': 1,
        'batch_size': 128,
        'max_scheduled_tokens': 4096,
        'max_model_len': 4096,
        'gpu_memory_utilization': 0.90,
        'block_size': 16,
        # Optional: workload parameters for distribution mode
        'prefix_tokens': 0,
        'prompt_tokens': 800,
        'prompt_tokens_stdev': 300,
        'prompt_tokens_min': 100,
        'prompt_tokens_max': 2000,
        'output_tokens': 400,
        'output_tokens_stdev': 200,
        'output_tokens_min': 50,
        'output_tokens_max': 1000,
    }

    # Run simulation at 10 QPS
    qps = 10.0
    print(f"Testing BLIS runner with QPS={qps}\n")

    metrics = run_blis(config, qps, num_requests=100)

    if metrics:
        print("\n" + "="*60)
        print("Simulation Results:")
        print("="*60)
        print(json.dumps(metrics, indent=2))
    else:
        print("\nSimulation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
