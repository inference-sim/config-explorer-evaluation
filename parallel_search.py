"""
Parallel Config Search - Evaluate multiple vLLM configs in parallel to find best one
"""
import argparse
import json
import sys
import yaml
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple
from qps_search import find_max_qps, display_metrics_category
from capacity_planner import calculate_total_kv_blocks


def evaluate_config(args: Tuple[Dict, list, Optional[str], float, float, float, int]) -> Dict:
    """
    Evaluate a single config by finding max QPS meeting SLOs.

    Args:
        args: Tuple containing (config, slos, trace_file, qps_min, qps_max, qps_granularity, config_id)

    Returns:
        Dictionary with config, max_qps, metrics, and total_kv_blocks
    """
    config, slos, trace_file, qps_min, qps_max, qps_granularity, config_id = args

    print(f"\n[Config {config_id}] Starting evaluation...")
    print(f"[Config {config_id}] batch_size={config['batch_size']}, "
          f"max_num_scheduled_tokens={config["max_scheduled_tokens"]}, "
          f"max_model_len={config['max_model_len']}, "
          f"gpu_mem_util={config['gpu_memory_utilization']}, "
          f"block_size={config['block_size']} ")

    try:
        # Calculate total KV blocks for this config (quiet mode)
        total_kv_blocks = calculate_total_kv_blocks(
            model=config['model'],
            hardware=config['hardware'],
            tp=config['tp'],
            max_model_len=config['max_model_len'],
            gpu_memory_utilization=config['gpu_memory_utilization'],
            block_size=config.get('block_size', 16),
            verbose=False  # Suppress detailed logs in parallel mode
        )

        # Run binary search to find max QPS (quiet mode)
        max_qps, metrics = find_max_qps(
            config=config,
            slos=slos,
            trace_file=trace_file,
            qps_min=qps_min,
            qps_max=qps_max,
            qps_granularity=qps_granularity,
            verbose=False  # Suppress detailed logs in parallel mode
        )

        # Add total_kv_blocks to metrics
        if metrics:
            metrics['total_kv_blocks'] = total_kv_blocks

        print(f"\n[Config {config_id}] ✅ Complete - Max QPS: {max_qps:.2f}")

        return {
            'config_id': config_id,
            'config': config,
            'max_qps': max_qps,
            'metrics': metrics,
            'success': True
        }

    except Exception as e:
        print(f"\n[Config {config_id}] ❌ Failed: {str(e)}")
        return {
            'config_id': config_id,
            'config': config,
            'max_qps': -1,
            'metrics': {},
            'success': False,
            'error': str(e)
        }


def load_config_space(yaml_path: str) -> Tuple[Dict, List[Dict], list]:
    """
    Load config space from YAML file.

    Supports two formats:
    1. Explicit configs: List of complete configs under "configs:" key
    2. Grid search: Lists of values for each parameter (generates Cartesian product)

    Args:
        yaml_path: Path to YAML file

    Returns:
        Tuple of (base_config, configs_list, slos)
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}")
        sys.exit(1)

    # Extract base parameters
    base_config = {
        'model': data.get('model'),
        'hardware': data.get('hardware', 'H100'),
        'tp': data.get('tp', 1),
        'vllm_version': data.get('vllm_version', 'vllm/vllm-openai:v0.8.4'),
        'num_requests': data.get('num_requests', 500),
    }

    # Add optional workload parameters if present
    optional_params = [
        'prefix_tokens', 'prompt_tokens', 'prompt_tokens_stdev',
        'prompt_tokens_min', 'prompt_tokens_max',
        'output_tokens', 'output_tokens_stdev',
        'output_tokens_min', 'output_tokens_max'
    ]
    for param in optional_params:
        if param in data:
            base_config[param] = data[param]

    # Get SLOs
    slos = data.get('slos', [])
    if not slos:
        print("Error: No SLOs defined in YAML file")
        sys.exit(1)

    # Check if using explicit configs or grid search format
    if 'configs' in data and data['configs']:
        # Format 1: Explicit configs (backward compatible)
        configs = data['configs']
        merged_configs = []
        for cfg in configs:
            merged = {**base_config, **cfg}
            merged_configs.append(merged)
        print(f"Using explicit config format: {len(merged_configs)} configs")

    else:
        # Format 2: Grid search - generate Cartesian product
        # Config parameters that can be swept
        grid_params = {
            'batch_size': data.get('batch_size'),
            'max_scheduled_tokens': data.get('max_scheduled_tokens'),
            'max_model_len': data.get('max_model_len'),
            'gpu_memory_utilization': data.get('gpu_memory_utilization'),
            'block_size': data.get('block_size', [16]),  # default if not specified
        }

        # Convert single values to lists for consistency
        for key, value in grid_params.items():
            if value is not None and not isinstance(value, list):
                grid_params[key] = [value]

        # Check that required parameters are present
        required = ['batch_size', 'max_scheduled_tokens', 'max_model_len', 'gpu_memory_utilization']
        missing = [p for p in required if not grid_params.get(p)]
        if missing:
            print(f"Error: Missing required grid search parameters: {', '.join(missing)}")
            print(f"Provide either 'configs:' list or parameter lists for grid search")
            sys.exit(1)

        # Generate all combinations
        param_names = ['batch_size', 'max_scheduled_tokens', 'max_model_len',
                       'gpu_memory_utilization', 'block_size']
        param_values = [grid_params[name] for name in param_names]

        merged_configs = []
        for combination in product(*param_values):
            config = {**base_config}
            for name, value in zip(param_names, combination):
                config[name] = value
            merged_configs.append(config)

        print(f"Using grid search format: {len(merged_configs)} configs generated from:")
        for name in param_names:
            if grid_params[name]:
                print(f"  {name}: {grid_params[name]}")

    if not merged_configs:
        print("Error: No configs generated")
        sys.exit(1)

    return base_config, merged_configs, slos


def display_results(results: List[Dict], slos: list):
    """
    Display all results and highlight the best config.

    Args:
        results: List of result dictionaries
        slos: List of SLO constraints
    """
    print("\n" + "="*80)
    print("PARALLEL CONFIG SEARCH RESULTS")
    print("="*80)

    # Sort results by max_qps (descending)
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    successful_results.sort(key=lambda x: x['max_qps'], reverse=True)

    if not successful_results:
        print("\n❌ All configs failed!")
        for result in failed_results:
            print(f"\nConfig {result['config_id']}: {result.get('error', 'Unknown error')}")
        return

    # Display all results
    print(f"\nEvaluated {len(results)} configs:")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Failed: {len(failed_results)}")

    print("\n" + "-"*80)
    print("All Configs (sorted by Max QPS):")
    print("-"*80)

    for i, result in enumerate(successful_results, 1):
        config = result['config']
        max_qps = result['max_qps']
        metrics = result['metrics']

        rank_marker = "🏆" if i == 1 else f" {i}."

        print(f"\n{rank_marker} Config {result['config_id']}:")
        print(f"   Max QPS: {max_qps:.2f}")
        print(f"   batch_size: {config['batch_size']}")
        print(f"   max_scheduled_tokens: {config['max_scheduled_tokens']}")
        print(f"   max_model_len: {config['max_model_len']}")
        print(f"   gpu_memory_utilization: {config['gpu_memory_utilization']}")
        print(f"   block_size: {config.get('block_size', 16)}")
        print(f"   total_kv_blocks: {metrics.get('total_kv_blocks', 0):,}")

        # Show SLO metrics
        print(f"   SLO Metrics:")
        for slo in slos:
            metric_value = metrics.get(slo['metric'], 0)
            status = "✅" if metric_value <= slo['threshold_ms'] else "❌"
            print(f"     {status} {slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")

    # Highlight best config
    best = successful_results[0]
    print("\n" + "="*80)
    print("BEST CONFIG")
    print("="*80)
    print(f"\nConfig {best['config_id']} achieves highest QPS: {best['max_qps']:.2f}")
    print(f"\nOptimal Configuration:")
    print(f"  batch_size: {best['config']['batch_size']}")
    print(f"  max_scheduled_tokens: {best['config']['max_scheduled_tokens']}")
    print(f"  max_model_len: {best['config']['max_model_len']}")
    print(f"  gpu_memory_utilization: {best['config']['gpu_memory_utilization']}")
    print(f"  block_size: {best['config'].get('block_size', 16)}")
    print(f"  total_kv_blocks: {best['metrics'].get('total_kv_blocks', 0):,}")

    print(f"\nSLO Compliance:")
    for slo in slos:
        metric_value = best['metrics'].get(slo['metric'], 0)
        status = "✅" if metric_value <= slo['threshold_ms'] else "❌"
        print(f"  {status} {slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")

    # Display all BLIS metrics (reusing Step 2 implementation)
    print(f"\nBest Config BLIS Metrics:")
    metrics = best['metrics']

    # Display latency metrics using helper function from qps_search.py
    display_metrics_category(metrics, 'e2e_', 'End-to-End Latency')
    display_metrics_category(metrics, 'ttft_', 'Time to First Token (TTFT)')
    display_metrics_category(metrics, 'itl_', 'Inter-Token Latency (ITL)')

    # Throughput Metrics
    print(f"\n  Throughput:")
    if 'responses_per_sec' in metrics:
        print(f"    Responses/sec: {metrics['responses_per_sec']:.2f}")
    if 'tokens_per_sec' in metrics:
        print(f"    Tokens/sec: {metrics['tokens_per_sec']:.2f}")

    # Request Metrics
    print(f"\n  Requests:")
    if 'completed_requests' in metrics:
        print(f"    Completed: {metrics['completed_requests']}")
    if 'failed_requests' in metrics:
        print(f"    Failed: {metrics['failed_requests']}")
    if 'total_requests' in metrics:
        print(f"    Total: {metrics['total_requests']}")

    print("\n" + "="*80)


def save_results(results: List[Dict], output_file: str):
    """
    Save results to JSON file.

    Args:
        results: List of result dictionaries
        output_file: Path to output JSON file
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️  Failed to save results: {e}")


def main():
    """
    Main entry point with argparse CLI.
    """
    parser = argparse.ArgumentParser(
        description='Parallel config search to find optimal vLLM configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Grid search format (recommended)
  python parallel_search.py --configs examples/configs_grid_search.yaml

  # Explicit configs format
  python parallel_search.py --configs examples/configs_explicit.yaml

  # With trace file
  python parallel_search.py -c examples/configs_explicit.yaml --trace traces/chat.csv

  # Specify number of workers
  python parallel_search.py -c examples/configs_explicit.yaml --num-workers 4

  # Save results to JSON
  python parallel_search.py -c examples/configs_explicit.yaml --output results.json

YAML Config Formats:

Format 1 - Grid Search (generates all combinations):
  batch_size: [128, 256, 512]
  max_scheduled_tokens: [2048, 4096]
  max_model_len: [4096, 8192]
  gpu_memory_utilization: [0.90]
  # Generates 3 × 2 × 2 × 1 = 12 configs

Format 2 - Explicit Configs (specify each one):
  configs:
    - batch_size: 128
      max_scheduled_tokens: 4096
      max_model_len: 4096
      gpu_memory_utilization: 0.90
    - batch_size: 256
      max_scheduled_tokens: 8192
      max_model_len: 8192
      gpu_memory_utilization: 0.90
        '''
    )

    # Required arguments
    parser.add_argument(
        '-c', '--configs',
        required=True,
        type=str,
        help='Path to YAML file with config space'
    )

    # Optional arguments
    parser.add_argument(
        '-t', '--trace',
        type=str,
        default=None,
        help='Path to trace file (CSV with prompt_tokens, output_tokens)'
    )

    parser.add_argument(
        '-n', '--num-workers',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: CPU count = {cpu_count()})'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output JSON file to save results (optional)'
    )

    # Search parameters
    parser.add_argument(
        '--qps-min',
        type=float,
        default=0.1,
        help='Minimum QPS to search (default: 0.1)'
    )

    parser.add_argument(
        '--qps-max',
        type=float,
        default=100.0,
        help='Maximum QPS to search (default: 100.0)'
    )

    parser.add_argument(
        '--qps-granularity',
        type=float,
        default=0.01,
        help='QPS granularity/step size (default: 0.01)'
    )

    args = parser.parse_args()

    # Determine number of workers
    num_workers = args.num_workers if args.num_workers else cpu_count()

    # Load config space
    print(f"\nLoading config space from: {args.configs}")
    base_config, configs, slos = load_config_space(args.configs)

    print(f"\n" + "="*80)
    print("PARALLEL CONFIG SEARCH")
    print("="*80)
    print(f"\nBase Configuration:")
    print(f"  Model: {base_config['model']}")
    print(f"  Hardware: {base_config['hardware']}")
    print(f"  TP: {base_config['tp']}")
    print(f"  Num Requests: {base_config['num_requests']}")

    print(f"\nSLO Constraints:")
    for slo in slos:
        print(f"  {slo['metric']} < {slo['threshold_ms']} ms")

    print(f"\nSearch Configuration:")
    print(f"  Number of configs to evaluate: {len(configs)}")
    print(f"  Parallel workers: {num_workers}")
    print(f"  QPS search range: [{args.qps_min}, {args.qps_max}]")
    print(f"  QPS granularity: {args.qps_granularity}")

    if args.trace:
        print(f"  Trace file: {args.trace}")

    print(f"\n" + "="*80)
    print("Starting parallel evaluation...")
    print("="*80)

    # Prepare arguments for each config
    eval_args = [
        (config, slos, args.trace, args.qps_min, args.qps_max, args.qps_granularity, i+1)
        for i, config in enumerate(configs)
    ]

    # Run parallel evaluation
    with Pool(num_workers) as pool:
        results = pool.map(evaluate_config, eval_args)

    # Display results
    display_results(results, slos)

    # Save results to file if requested
    if args.output:
        save_results(results, args.output)

    # Return best config
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best = max(successful_results, key=lambda x: x['max_qps'])
        print(f"\n✅ Search completed successfully!")
        print(f"   Best config achieves {best['max_qps']:.2f} QPS\n")
        return 0
    else:
        print(f"\n❌ All configs failed!\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
