"""
QPS Search - Binary search to find maximum QPS meeting SLO threshold

Supports both BLIS (fast and accurate) and Vidur (ML-based) simulators.
Use --simulator argument to switch between them.
"""
import argparse
import sys
import yaml
import numpy as np
from typing import Dict, Tuple, Optional

# Import both simulators
from vidur_runner import run_vidur
from blis_runner import run_blis

# Global variable set by command-line argument
USE_VIDUR = False


# Binary search configuration
MIN_RPS_FOR_BINARY_SEARCH = 0.1
MAX_RPS_FOR_BINARY_SEARCH = 100.0
RPS_GRANULARITY_FOR_BINARY_SEARCH = 0.01


def display_metrics_category(metrics: Dict, prefix: str, category_name: str) -> None:
    """
    Display metrics for a specific category (e2e, ttft, or itl).

    Args:
        metrics: Dictionary of all metrics
        prefix: Metric prefix (e.g., 'e2e_', 'ttft_', 'itl_')
        category_name: Display name for the category
    """
    if not any(k.startswith(prefix) for k in metrics):
        return

    print(f"\n  {category_name}:")
    # Define metric order
    metric_suffixes = ['mean_ms', 'p90_ms', 'p95_ms', 'p99_ms']
    metric_labels = ['Mean', 'P90', 'P95', 'P99']

    for suffix, label in zip(metric_suffixes, metric_labels):
        key = f"{prefix}{suffix}"
        if key in metrics:
            print(f"    {label}: {metrics[key]:.2f} ms")


def violates_slo(metrics: Optional[Dict], slos: list) -> tuple:
    """
    Check if system violates any SLO.

    Args:
        metrics: Simulation metrics from simulator
        slos: List of SLO constraints, each with 'metric' and 'threshold_ms'
              e.g., [{"metric": "e2e_p95_ms", "threshold_ms": 1000}]

    Returns:
        Tuple of (violated, violated_slos) where:
        - violated: True if any SLO violated, False if all met
        - violated_slos: List of violated SLO descriptions
    """
    if metrics is None:
        return True, ["Simulation failed"]

    violated_slos = []
    for slo in slos:
        metric = slo['metric']
        threshold = slo['threshold_ms']
        value = metrics.get(metric, float('inf'))

        if value > threshold:
            violated_slos.append(f"{metric} ({value:.2f}ms > {threshold}ms)")

    return len(violated_slos) > 0, violated_slos


def find_max_qps(
    config: Dict,
    slos: list,
    trace_file: Optional[str] = None,
    qps_min: float = MIN_RPS_FOR_BINARY_SEARCH,
    qps_max: float = MAX_RPS_FOR_BINARY_SEARCH,
    qps_granularity: float = RPS_GRANULARITY_FOR_BINARY_SEARCH,
    verbose: bool = True
) -> Tuple[float, Dict]:
    """
    Find maximum QPS where all SLO constraints are met using binary search.

    Args:
        config: Configuration dictionary for simulator, including:
                - num_requests: Number of requests per simulation (optional, default: 500)
                - Other config parameters (model, hardware, etc.)
        slos: List of SLO constraints, each with 'metric' and 'threshold_ms'
              e.g., [{"metric": "e2e_p95_ms", "threshold_ms": 1000},
                     {"metric": "ttft_p90_ms", "threshold_ms": 500}]
        trace_file: Optional path to trace file (only used for Vidur; BLIS uses distribution mode)
        qps_min: Minimum QPS to search (default: 0.1)
        qps_max: Maximum QPS to search (default: 100.0)
        qps_granularity: QPS step size (default: 0.01)
        verbose: If True, print detailed iteration logs (default: True)

    Returns:
        Tuple of (max_qps, metrics) where max_qps is the highest QPS meeting all SLOs
    """
    # Validate required config fields
    required_fields = ['model', 'hardware', 'tp', 'batch_size', 'max_scheduled_tokens',
                       'max_model_len', 'gpu_memory_utilization']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Config missing required fields: {', '.join(missing_fields)}")

    # Get num_requests from config, default to 500
    num_requests = config.get('num_requests', 500)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting Binary Search for Max QPS")
        print(f"{'='*60}")
        print(f"SLO Constraints:")
        for slo in slos:
            print(f"  {slo['metric']} < {slo['threshold_ms']} ms")
        print(f"Search Range: [{qps_min}, {qps_max}] QPS")
        print(f"Granularity: {qps_granularity} QPS")
        print(f"Requests per simulation: {num_requests}")
        print(f"{'='*60}\n")

    # Create discrete QPS values array
    # Note: np.arange(0.1, 100.0, 0.01) creates values up to 99.99 (not 100.0)
    # This is fine - we can adjust qps_max to 100.01 if exactly 100.0 QPS is needed
    qps_values = np.arange(qps_min, qps_max, qps_granularity).tolist()

    # Binary search on indices
    low_idx = 0
    high_idx = len(qps_values) - 1
    max_qps = -1
    best_metrics = {}
    iteration = 0

    while low_idx <= high_idx:
        iteration += 1
        mid_idx = (low_idx + high_idx) // 2
        test_qps = qps_values[mid_idx]

        if verbose:
            print(f"Iteration {iteration}: Testing QPS = {test_qps:.2f} (index {mid_idx}/{len(qps_values)-1})")

        # Run simulation
        # BLIS uses distribution mode (no trace file), Vidur uses trace file
        actual_trace = trace_file if USE_VIDUR else None
        if USE_VIDUR:
            metrics = run_vidur(config, test_qps, actual_trace, num_requests, verbose=verbose)
        else:
            metrics = run_blis(config, test_qps, actual_trace, num_requests, verbose=verbose)

        # Handle simulation failure (returns None when invalid metrics)
        if metrics is None:
            if verbose:
                simulator_name = "Vidur" if USE_VIDUR else "BLIS"
                print(f"  ❌ {simulator_name} simulation failed (invalid metrics)")
                print(f"  Treating as SLO violation, searching lower")
            high_idx = mid_idx - 1
            continue

        # Check if any SLO violated
        slo_violated, violated_list = violates_slo(metrics, slos)

        if verbose:
            # Print all SLO metrics
            for slo in slos:
                metric_value = metrics.get(slo['metric'], float('inf'))
                status = "❌" if metric_value > slo['threshold_ms'] else "✅"
                print(f"  {status} {slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")

        if slo_violated:
            if verbose:
                print(f"  ❌ SLO violated: {', '.join(violated_list)}")
                print(f"  Searching lower")
            high_idx = mid_idx - 1
        else:
            if verbose:
                print(f"  ✅ All SLOs met, searching higher")
            max_qps = test_qps
            best_metrics = metrics
            low_idx = mid_idx + 1

        if verbose:
            print()

    # Handle edge case: SLO met at all QPS values
    if max_qps == -1:
        if verbose:
            print(f"⚠️  SLO violated at all tested QPS values up to {qps_max}")
        # All simulations failed or violated SLOs
        simulator_name = "Vidur" if USE_VIDUR else "BLIS"
        raise RuntimeError(f"All {simulator_name} simulations failed or violated SLOs")

    # Final validation: ensure we have valid metrics
    if not best_metrics or max_qps <= 0:
        raise RuntimeError("Binary search completed but no valid configuration found")

    if verbose:
        print(f"{'='*60}")
        print(f"Binary Search Complete")
        print(f"{'='*60}")
        print(f"Max QPS meeting all SLOs: {max_qps:.2f} QPS")
        print(f"\nSLO Metrics at Max QPS:")
        for slo in slos:
            metric_value = best_metrics.get(slo['metric'], 0)
            status = "✅" if metric_value <= slo['threshold_ms'] else "❌"
            print(f"  {status} {slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")
        print(f"\nThroughput: {best_metrics.get('responses_per_sec', 0):.2f} QPS")
        if 'total_kv_blocks' in best_metrics:
            print(f"Total KV Blocks: {best_metrics['total_kv_blocks']:,}")
        print(f"{'='*60}\n")

    return max_qps, best_metrics


def main():
    """
    Main entry point with argparse CLI.
    """
    import json
    import time

    parser = argparse.ArgumentParser(
        description='Binary search to find maximum QPS meeting SLO constraints (BLIS: accurate, Vidur: ML-based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use BLIS simulator (fast and accurate, recommended)
  python qps_search.py --config test_config.json --simulator blis

  # Use Vidur simulator (ML-based)
  python qps_search.py --config test_config.json --simulator vidur

  # With YAML config (uses first config from YAML file)
  python qps_search.py -c examples/configs_grid_search.yaml --simulator blis

  # With trace file (only used for Vidur)
  python qps_search.py -c test_config.json --simulator vidur --trace traces/chat.csv

  # Custom search parameters
  python qps_search.py -c test_config.json --simulator blis --qps-max 50.0

  # Save results to file
  python qps_search.py -c test_config.json --simulator blis --output results.json

Config file format (JSON or YAML):
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "num_requests": 500,
    "slos": [
      {"metric": "e2e_p95_ms", "threshold_ms": 1000},
      {"metric": "ttft_mean_ms", "threshold_ms": 150}
    ],
    ...
  }

Available SLO metrics:
  End-to-End: e2e_mean_ms, e2e_p90_ms, e2e_p95_ms, e2e_p99_ms
  TTFT:       ttft_mean_ms, ttft_p90_ms, ttft_p95_ms, ttft_p99_ms
  ITL:        itl_mean_ms, itl_p90_ms, itl_p95_ms, itl_p99_ms

Simulators (required):
  blis:  Fast and accurate coefficient-based simulator (recommended)
  vidur: ML-based simulator with Random Forest latency prediction
        '''
    )

    # Required arguments
    parser.add_argument(
        '-c', '--config',
        required=True,
        type=str,
        help='Path to config file (JSON or YAML). For YAML grid search, uses first value from each parameter list.'
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

    # Search parameters
    parser.add_argument(
        '--qps-min',
        type=float,
        default=MIN_RPS_FOR_BINARY_SEARCH,
        help=f'Minimum QPS to search (default: {MIN_RPS_FOR_BINARY_SEARCH})'
    )
    parser.add_argument(
        '--qps-max',
        type=float,
        default=MAX_RPS_FOR_BINARY_SEARCH,
        help=f'Maximum QPS to search (default: {MAX_RPS_FOR_BINARY_SEARCH})'
    )
    parser.add_argument(
        '--qps-granularity',
        type=float,
        default=RPS_GRANULARITY_FOR_BINARY_SEARCH,
        help=f'QPS granularity/step size (default: {RPS_GRANULARITY_FOR_BINARY_SEARCH})'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to save results JSON file (e.g., results.json)'
    )

    args = parser.parse_args()

    # Set global simulator selection
    global USE_VIDUR
    USE_VIDUR = (args.simulator == 'vidur')

    # Load config (supports both JSON and YAML)
    try:
        with open(args.config, 'r') as f:
            # Detect file format by extension
            if args.config.endswith(('.yaml', '.yml')):
                data = yaml.safe_load(f)
                # For YAML files, we need to pick a single config
                # If 'configs' key exists, use the first one, otherwise use the data as-is
                if 'configs' in data and isinstance(data['configs'], list) and len(data['configs']) > 0:
                    # YAML with explicit configs list - merge base config with first config
                    base_config = {k: v for k, v in data.items() if k not in ['configs', 'slos']}
                    config = {**base_config, **data['configs'][0]}
                    print(f"Note: YAML file has {len(data['configs'])} configs, using first one for single-config search")
                elif 'configs' in data:
                    print(f"Error: YAML 'configs' key exists but is empty or invalid")
                    sys.exit(1)
                else:
                    # YAML with grid search format - need to pick specific values
                    # Extract base params and use first value from each list
                    config = {}
                    grid_params = ['tp', 'batch_size', 'max_scheduled_tokens', 'max_model_len',
                                   'gpu_memory_utilization', 'block_size']
                    for key, value in data.items():
                        if key in grid_params and isinstance(value, list):
                            config[key] = value[0]  # Use first value
                        elif key not in ['slos']:
                            config[key] = value
                    print(f"Note: YAML file uses grid search format, using first value from each parameter list")
            else:
                # JSON format
                config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        print(f"Error: Invalid config file: {e}")
        sys.exit(1)

    # Get SLO configuration from config file or data (for YAML)
    if args.config.endswith(('.yaml', '.yml')):
        # For YAML files, slos might be in the original data, not in config
        with open(args.config, 'r') as f:
            data = yaml.safe_load(f)
            slos = data.get('slos', [])
    else:
        slos = config.get('slos', [])

    if not slos:
        print("Error: No SLOs defined in config file")
        print("Please add 'slos' field to config.json:")
        print('  "slos": [')
        print('    {"metric": "e2e_p95_ms", "threshold_ms": 1000}')
        print('  ]')
        sys.exit(1)

    simulator_name = "Vidur" if USE_VIDUR else "BLIS"
    print(f"\n{'='*60}")
    print(f"QPS Search - {simulator_name} Simulator")
    print(f"{'='*60}")

    print(f"\nConfiguration:")
    print(f"  Model: {config['model']}")
    print(f"  Hardware: {config['hardware']}")
    print(f"  TP: {config['tp']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Max Scheduled Tokens: {config['max_scheduled_tokens']}")
    print(f"  Max Model Length: {config['max_model_len']}")
    print(f"  GPU Memory Utilization: {config['gpu_memory_utilization']}")

    # Show roofline model parameters if present
    if 'hardware_config' in config:
        print(f"  Hardware Config: {config['hardware_config']}")
        if 'model_config_folder_base' in config:
            print(f"  Model Config Base: {config['model_config_folder_base']}")
        print(f"  Roofline Model: ENABLED")
    else:
        print(f"  Roofline Model: DISABLED (using pre-trained coefficients)")

    if args.trace:
        print(f"  Trace File: {args.trace}")

    # Get num_requests from config, default to 500
    num_requests = config.get('num_requests', 500)
    print(f"  Num Requests: {num_requests}")

    print(f"\nSearch Parameters:")
    print(f"  QPS Range: [{args.qps_min}, {args.qps_max}]")
    print(f"  QPS Granularity: {args.qps_granularity}")

    print(f"\nSLO Constraints:")
    for slo in slos:
        print(f"  {slo['metric']} < {slo['threshold_ms']} ms")

    # Run binary search with error handling
    start_time = time.time()
    try:
        max_qps, metrics = find_max_qps(
            config=config,
            slos=slos,
            trace_file=args.trace,
            qps_min=args.qps_min,
            qps_max=args.qps_max,
            qps_granularity=args.qps_granularity
        )
    except RuntimeError as e:
        simulator_name = "Vidur" if USE_VIDUR else "BLIS"
        print("\n" + "="*60)
        print(f"❌ {simulator_name} Binary Search Failed")
        print("="*60)
        print(f"\nError: {str(e)}")
        print(f"\nPossible causes:")
        print(f"  - Configuration parameters may be invalid")
        print(f"  - SLO thresholds may be too strict")
        print(f"  - All QPS values violated SLOs or failed simulation")
        print()
        sys.exit(1)
    except Exception as e:
        simulator_name = "Vidur" if USE_VIDUR else "BLIS"
        print("\n" + "="*60)
        print(f"❌ Unexpected Error in {simulator_name} Simulation")
        print("="*60)
        print(f"\nError: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        print()
        sys.exit(1)

    if max_qps > 0:
        elapsed_time = time.time() - start_time

        # Save results to file if --output specified
        if args.output:
            # Extract only SLO metrics from the full metrics
            slo_metrics = {}
            for slo in slos:
                metric_name = slo['metric']
                if metric_name in metrics:
                    slo_metrics[metric_name] = metrics[metric_name]
                    slo_metrics[f"{metric_name}_threshold_ms"] = slo['threshold_ms']

            # Create result entry for this simulator
            simulator_result = {
                "max_qps": round(max_qps, 2),
                **slo_metrics,
                "total_kv_blocks": metrics.get('total_kv_blocks', None),
                "runtime_seconds": round(elapsed_time, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Create config section with all parameters except the following
            excluded_fields = {'model_config_folder_base', 'hardware_config', 'slos', 'vllm_version'}
            config_summary = {
                k: v for k, v in config.items()
                if k not in excluded_fields
            }
            # Add slos separately to the config summary
            config_summary['slos'] = slos

            # Load existing data if file exists, otherwise create new structure
            try:
                with open(args.output, 'r') as f:
                    output_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                output_data = {}

            # Initialize config_file entry if it doesn't exist
            if args.config not in output_data:
                output_data[args.config] = {
                    "config": config_summary,
                    "results": {}
                }

            # Add or update simulator result
            output_data[args.config]["results"][args.simulator] = simulator_result

            try:
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\n✅ Results saved to: {args.output}")
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to save results to {args.output}: {e}")

        print("\n" + "="*60)
        print("✅ Search completed successfully!")
        print("="*60)
        print(f"\nResults:")
        print(f"  Max QPS: {max_qps:.2f}")
        print(f"  Total Runtime: {elapsed_time:.2f} seconds")
        print(f"\nSLO Metrics at Max QPS:")
        for slo in slos:
            metric_value = metrics.get(slo['metric'], 0)
            status = "✅" if metric_value <= slo['threshold_ms'] else "❌"
            print(f"  {status} {slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")

        simulator_name = "Vidur" if USE_VIDUR else "BLIS"
        print(f"\nAll {simulator_name} Metrics:")

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

        # Configuration
        print(f"\n  Configuration:")
        if 'total_kv_blocks' in metrics:
            print(f"    Total KV Blocks: {metrics['total_kv_blocks']:,}")
        print(f"    QPS: {metrics.get('qps', max_qps):.2f}")

        print()
    else:
        print("\n❌ Search failed - no configuration meets all SLOs")
        sys.exit(1)


if __name__ == "__main__":
    main()
