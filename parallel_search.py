"""
Parallel Config Search - Evaluate multiple vLLM configs in parallel to find best one

Supports both BLIS (fast and accurate, recommended) and Vidur (ML-based) simulators.
"""
import argparse
import json
import shutil
import sys
import time
import yaml
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import both simulators and supporting modules
from vidur_runner import run_vidur
from blis_runner import run_blis
from capacity_planner import calculate_total_kv_blocks

# Import find_max_qps and display_metrics_category from qps_search
# (these work for both BLIS and Vidur)
import qps_search
from qps_search import display_metrics_category

# Global variable set by command-line argument
USE_VIDUR = False


def evaluate_config(args: Tuple[Dict, list, Optional[str], float, float, float, int, bool]) -> Dict:
    """
    Evaluate a single config by finding max QPS meeting SLOs.

    Args:
        args: Tuple containing (config, slos, trace_file, qps_min, qps_max, qps_granularity, config_id, use_vidur)
              Note: trace_file only used for Vidur; BLIS uses distribution mode

    Returns:
        Dictionary with config, max_qps, metrics, and total_kv_blocks
    """
    config, slos, trace_file, qps_min, qps_max, qps_granularity, config_id, use_vidur = args

    start_time = time.time()

    print(f"\n[Config {config_id}] Starting evaluation...")
    print(f"[Config {config_id}] batch_size={config['batch_size']}, "
          f"tp={config['tp']}, "
          f"max_num_scheduled_tokens={config["max_scheduled_tokens"]}, "
          f"max_model_len={config['max_model_len']}, "
          f"gpu_mem_util={config['gpu_memory_utilization']}, "
          f"block_size={config['block_size']} ")

    try:
        # Calculate total KV blocks for BLIS only (Vidur calculates internally)
        if not use_vidur:
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
        # BLIS uses distribution mode (no trace file), Vidur uses trace file
        actual_trace = trace_file if use_vidur else None

        # Use qps_search.find_max_qps for both BLIS and Vidur
        # Set the global flag to match our simulator choice
        qps_search.USE_VIDUR = use_vidur
        max_qps, metrics = qps_search.find_max_qps(
            config=config,
            slos=slos,
            trace_file=actual_trace,
            qps_min=qps_min,
            qps_max=qps_max,
            qps_granularity=qps_granularity,
            verbose=False  # Suppress detailed logs in parallel mode
        )

        # Add total_kv_blocks to metrics (BLIS only)
        if not use_vidur and metrics:
            metrics['total_kv_blocks'] = total_kv_blocks

        elapsed_time = time.time() - start_time
        print(f"\n[Config {config_id}] ✅ Complete - Max QPS: {max_qps:.2f} (Runtime: {elapsed_time:.2f}s)")

        return {
            'config_id': config_id,
            'config': config,
            'max_qps': max_qps,
            'metrics': metrics,
            'runtime_seconds': elapsed_time,
            'success': True
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n[Config {config_id}] ❌ Simulation failed: {str(e)} (Runtime: {elapsed_time:.2f}s)")
        return {
            'config_id': config_id,
            'config': config,
            'runtime_seconds': elapsed_time,
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
        'vllm_version': data.get('vllm_version', 'vllm/vllm-openai:v0.8.4'),
        'num_requests': data.get('num_requests', 500),
    }

    # TP can be in base_config (fixed for explicit configs) or in grid_params (swept)
    if 'tp' in data and not isinstance(data.get('tp'), list):
        base_config['tp'] = data['tp']
    elif 'tp' not in data:
        # Default TP value if not specified anywhere
        base_config['tp'] = 1

    # Add optional workload and roofline parameters if present
    optional_params = [
        'model_config_folder_base', 'hardware_config',
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
            'tp': data.get('tp'),
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
        required = ['batch_size', 'tp', 'max_scheduled_tokens', 'max_model_len', 'gpu_memory_utilization']
        missing = [p for p in required if not grid_params.get(p)]
        if missing:
            print(f"Error: Missing required grid search parameters: {', '.join(missing)}")
            print(f"Provide either 'configs:' list or parameter lists for grid search")
            sys.exit(1)

        # Generate all combinations
        param_names = ['tp', 'batch_size', 'max_scheduled_tokens', 'max_model_len',
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
    Display successful results and highlight the best config.

    Note: Only successful configs should be passed to this function.
    Failed configs are filtered out before calling this function.

    Args:
        results: List of successful result dictionaries
        slos: List of SLO constraints
    """
    print("\n" + "="*80)
    print("PARALLEL CONFIG SEARCH RESULTS")
    print("="*80)

    # Sort results by max_qps (descending)
    results.sort(key=lambda x: x['max_qps'], reverse=True)

    print(f"\nSuccessful configs: {len(results)}")

    print("\n" + "-"*80)
    print("Ranked by Max QPS:")
    print("-"*80)

    for i, result in enumerate(results, 1):
        config = result['config']
        max_qps = result['max_qps']
        metrics = result['metrics']

        rank_marker = "🏆" if i == 1 else f" {i}."

        print(f"\n{rank_marker} Config {result['config_id']}:")
        print(f"   Max QPS: {max_qps:.2f}")
        print(f"   tp: {config['tp']}")
        print(f"   batch_size: {config['batch_size']}")
        print(f"   max_scheduled_tokens: {config['max_scheduled_tokens']}")
        print(f"   max_model_len: {config['max_model_len']}")
        print(f"   gpu_memory_utilization: {config['gpu_memory_utilization']}")
        print(f"   block_size: {config.get('block_size', 16)}")
        if 'total_kv_blocks' in metrics:
            print(f"   total_kv_blocks: {metrics['total_kv_blocks']:,}")

        # Show SLO metrics
        print(f"   SLO Metrics:")
        for slo in slos:
            metric_value = metrics.get(slo['metric'], 0)
            status = "✅" if metric_value <= slo['threshold_ms'] else "❌"
            print(f"     {status} {slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")

    # Highlight best config
    best = results[0]
    print("\n" + "="*80)
    print("BEST CONFIG")
    print("="*80)
    print(f"\nConfig {best['config_id']} achieves highest QPS: {best['max_qps']:.2f}")
    print(f"\nOptimal Configuration:")
    print(f"  tp: {best['config']['tp']}")
    print(f"  batch_size: {best['config']['batch_size']}")
    print(f"  max_scheduled_tokens: {best['config']['max_scheduled_tokens']}")
    print(f"  max_model_len: {best['config']['max_model_len']}")
    print(f"  gpu_memory_utilization: {best['config']['gpu_memory_utilization']}")
    print(f"  block_size: {best['config'].get('block_size', 16)}")
    if 'total_kv_blocks' in best['metrics']:
        print(f"  total_kv_blocks: {best['metrics']['total_kv_blocks']:,}")

    print(f"\nSLO Compliance:")
    for slo in slos:
        metric_value = best['metrics'].get(slo['metric'], 0)
        status = "✅" if metric_value <= slo['threshold_ms'] else "❌"
        print(f"  {status} {slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")

    # Display all metrics (from either BLIS or Vidur)
    simulator_name = "Vidur" if USE_VIDUR else "BLIS"
    print(f"\nBest Config {simulator_name} Metrics:")
    metrics = best['metrics']

    # Display latency metrics using helper function
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


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (can be dict, list, numpy type, etc.)

    Returns:
        JSON-serializable version of the object
    """
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (bool, np.bool_)):
        # Handle both Python bool and numpy bool (np.bool_ is still valid)
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        # Use np.integer base class (works with NumPy 1.x and 2.x)
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        # Use np.floating base class (works with NumPy 1.x and 2.x)
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_results(successful_results: List[Dict], failed_results: List[Dict],
                 output_file: str, slos: list, base_config: Dict, total_search_time: float):
    """
    Save comprehensive results to JSON file, including both successful and failed configs.

    Args:
        successful_results: List of successful result dictionaries
        failed_results: List of failed result dictionaries
        output_file: Path to output JSON file
        slos: List of SLO constraints
        base_config: Base configuration used for all configs
        total_search_time: Total runtime in seconds for the search
    """
    import datetime

    # Sort successful results by max_qps
    successful_results.sort(key=lambda x: x['max_qps'], reverse=True)

    # Build comprehensive output
    simulator_name = "Vidur" if USE_VIDUR else "BLIS"

    output = {
        'metadata': {
            'simulator': simulator_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'model': base_config.get('model'),
            'hardware': base_config.get('hardware'),
            'num_requests': base_config.get('num_requests'),
            'slo_constraints': slos
        },
        'summary': {
            'total_configs_evaluated': len(successful_results) + len(failed_results),
            'successful_configs': len(successful_results),
            'failed_configs': len(failed_results),
            'best_config_id': successful_results[0]['config_id'] if successful_results else None,
            'best_max_qps': float(successful_results[0]['max_qps']) if successful_results else None,
            'total_search_runtime_seconds': float(total_search_time)
        },
        'successful_configs': [
            {
                'rank': i + 1,
                'config_id': result['config_id'],
                'max_qps': float(result['max_qps']),
                'runtime_seconds': float(result.get('runtime_seconds', 0)),
                'configuration': {
                    'tp': result['config']['tp'],
                    'batch_size': result['config']['batch_size'],
                    'max_scheduled_tokens': result['config']['max_scheduled_tokens'],
                    'max_model_len': result['config']['max_model_len'],
                    'gpu_memory_utilization': result['config']['gpu_memory_utilization'],
                    'block_size': result['config'].get('block_size', 16)
                },
                'slo_metrics': {
                    slo['metric']: {
                        'value_ms': float(result['metrics'].get(slo['metric'], 0)),
                        'threshold_ms': slo['threshold_ms'],
                        'passes': bool(result['metrics'].get(slo['metric'], 0) <= slo['threshold_ms'])
                    }
                    for slo in slos
                },
                'all_metrics': convert_to_json_serializable(result['metrics'])
            }
            for i, result in enumerate(successful_results)
        ],
        'failed_configs': [
            {
                'config_id': result['config_id'],
                'runtime_seconds': float(result.get('runtime_seconds', 0)),
                'configuration': {
                    'tp': result['config']['tp'],
                    'batch_size': result['config']['batch_size'],
                    'max_scheduled_tokens': result['config']['max_scheduled_tokens'],
                    'max_model_len': result['config']['max_model_len'],
                    'gpu_memory_utilization': result['config']['gpu_memory_utilization'],
                    'block_size': result['config'].get('block_size', 16)
                },
                'error': result.get('error', 'Unknown error')
            }
            for result in failed_results
        ]
    }

    try:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✅ Comprehensive results saved to: {output_file}")
        print(f"   - {len(successful_results)} successful configs")
        print(f"   - {len(failed_results)} failed configs")
    except Exception as e:
        print(f"\n⚠️  Failed to save results: {e}")


def cleanup_tmp_directories(verbose: bool = True):
    """
    Clean up temporary vidur_sim directories in ./tmp.

    This removes all vidur_sim_* directories created during the experiment.
    """
    tmp_dir = Path('./tmp')
    if not tmp_dir.exists():
        return

    # Find all vidur_sim directories
    vidur_dirs = list(tmp_dir.glob('vidur_sim_*'))

    if vidur_dirs:
        if verbose:
            print(f"\n🧹 Cleaning up {len(vidur_dirs)} temporary directories...")

        for dir_path in vidur_dirs:
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Could not remove {dir_path}: {e}")

        if verbose:
            print(f"✅ Cleanup complete")


def main():
    """
    Main entry point with argparse CLI.
    """
    parser = argparse.ArgumentParser(
        description='Parallel config search to find optimal vLLM configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Grid search with BLIS (fast and accurate, recommended)
  python parallel_search.py --configs examples/configs_grid_search.yaml --simulator blis

  # Grid search with Vidur (ML-based)
  python parallel_search.py --configs examples/configs_grid_search.yaml --simulator vidur

  # Explicit configs format
  python parallel_search.py --configs examples/configs_explicit.yaml --simulator blis

  # With trace file (only used for Vidur)
  python parallel_search.py -c examples/configs_explicit.yaml --simulator vidur --trace traces/chat.csv

  # Specify number of workers
  python parallel_search.py -c examples/configs_explicit.yaml --simulator blis --num-workers 4

  # Save results to JSON
  python parallel_search.py -c examples/configs_explicit.yaml --simulator blis --output results.json

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
        '-s', '--simulator',
        type=str,
        choices=['blis', 'vidur'],
        required=True,
        help='Simulator to use: blis (fast and accurate, recommended) or vidur (ML-based)'
    )

    parser.add_argument(
        '-t', '--trace',
        type=str,
        default=None,
        help='Path to trace file (only used for Vidur; BLIS uses distribution mode)'
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

    # Set global simulator selection
    global USE_VIDUR
    USE_VIDUR = (args.simulator == 'vidur')

    # Also set for qps_search module
    qps_search.USE_VIDUR = USE_VIDUR

    # Determine number of workers
    num_workers = args.num_workers if args.num_workers else cpu_count()

    # Load config space
    print(f"\nLoading config space from: {args.configs}")
    base_config, configs, slos = load_config_space(args.configs)

    print(f"\n" + "="*80)
    print("PARALLEL CONFIG SEARCH")
    print("="*80)
    simulator_name = "Vidur" if USE_VIDUR else "BLIS"
    print(f"\nSimulator: {simulator_name}")
    print(f"\nBase Configuration:")
    print(f"  Model: {base_config['model']}")
    print(f"  Hardware: {base_config['hardware']}")
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
        (config, slos, args.trace, args.qps_min, args.qps_max, args.qps_granularity, i+1, USE_VIDUR)
        for i, config in enumerate(configs)
    ]

    # Run parallel evaluation
    search_start_time = time.time()
    with Pool(num_workers) as pool:
        all_results = pool.map(evaluate_config, eval_args)
    total_search_time = time.time() - search_start_time

    # Filter out failed configs immediately
    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]

    # Report failures if any
    if failed_results:
        print(f"\n⚠️  {len(failed_results)} config(s) failed (excluded from results):")
        for result in failed_results:
            print(f"   Config {result['config_id']}: Simulation failed")

    # Check if we have any successful results
    if not successful_results:
        print(f"\n❌ All {len(all_results)} configs failed! No results to display.\n")
        print(f"Total Search Runtime: {total_search_time:.2f} seconds\n")
        # Still save failed results if output file requested
        if args.output:
            save_results([], failed_results, args.output, slos, base_config, total_search_time)
        return 1

    # Display only successful results
    display_results(successful_results, slos)

    # Save comprehensive results (both successful and failed) to file if requested
    if args.output:
        save_results(successful_results, failed_results, args.output, slos, base_config, total_search_time)

    # Clean up temporary directories (Vidur only)
    if USE_VIDUR:
        cleanup_tmp_directories(verbose=True)

    # Report best config
    best = max(successful_results, key=lambda x: x['max_qps'])
    print(f"\n✅ Search completed successfully!")
    print(f"   Best config achieves {best['max_qps']:.2f} QPS")
    print(f"   Total Search Runtime: {total_search_time:.2f} seconds\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
