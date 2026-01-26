"""
QPS Search - Binary search to find maximum QPS meeting SLO threshold
"""
import argparse
import sys
import numpy as np
from typing import Dict, Tuple, Optional
from blis_runner import run_blis


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
    metric_suffixes = ['mean_ms', 'median_ms', 'p50_ms', 'p90_ms', 'p95_ms', 'p99_ms', 'max_ms']
    metric_labels = ['Mean', 'Median', 'P50', 'P90', 'P95', 'P99', 'Max']

    for suffix, label in zip(metric_suffixes, metric_labels):
        key = f"{prefix}{suffix}"
        if key in metrics:
            print(f"    {label}: {metrics[key]:.2f} ms")


def violates_slo(metrics: Optional[Dict], slos: list) -> tuple:
    """
    Check if system violates any SLO.

    Args:
        metrics: Simulation metrics from BLIS
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
    qps_granularity: float = RPS_GRANULARITY_FOR_BINARY_SEARCH
) -> Tuple[float, Dict]:
    """
    Find maximum QPS where all SLO constraints are met using binary search.

    Args:
        config: Configuration dictionary for BLIS, including:
                - num_requests: Number of requests per simulation (optional, default: 500)
                - Other BLIS config parameters (model, hardware, etc.)
        slos: List of SLO constraints, each with 'metric' and 'threshold_ms'
              e.g., [{"metric": "e2e_p95_ms", "threshold_ms": 1000},
                     {"metric": "ttft_p90_ms", "threshold_ms": 500}]
        trace_file: Optional path to trace file
        qps_min: Minimum QPS to search (default: 0.1)
        qps_max: Maximum QPS to search (default: 100.0)
        qps_granularity: QPS step size (default: 0.01)

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

        print(f"Iteration {iteration}: Testing QPS = {test_qps:.2f} (index {mid_idx}/{len(qps_values)-1})")

        # Run simulation
        metrics = run_blis(config, test_qps, trace_file, num_requests)

        # Check if any SLO violated
        slo_violated, violated_list = violates_slo(metrics, slos)

        if metrics:
            # Print all SLO metrics
            for slo in slos:
                metric_value = metrics.get(slo['metric'], float('inf'))
                status = "❌" if metric_value > slo['threshold_ms'] else "✅"
                print(f"  {status} {slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")

        if slo_violated:
            print(f"  ❌ SLO violated: {', '.join(violated_list)}")
            print(f"  Searching lower")
            high_idx = mid_idx - 1
        else:
            print(f"  ✅ All SLOs met, searching higher")
            max_qps = test_qps
            best_metrics = metrics
            low_idx = mid_idx + 1
        print()

    # Handle edge case: SLO met at all QPS values
    if max_qps == -1:
        print(f"⚠️  SLO violated at all tested QPS values up to {qps_max}")
        max_qps = qps_max
        # Run at max to get metrics
        best_metrics = run_blis(config, qps_max, trace_file, num_requests) or {}

    print(f"{'='*60}")
    print(f"Binary Search Complete")
    print(f"{'='*60}")
    print(f"Max QPS meeting all SLOs: {max_qps:.2f} QPS")
    if best_metrics:
        print(f"\nSLO Metrics at Max QPS:")
        for slo in slos:
            metric_value = best_metrics.get(slo['metric'], 0)
            status = "✅" if metric_value <= slo['threshold_ms'] else "❌"
            print(f"  {status} {slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")
        print(f"\nThroughput: {best_metrics.get('responses_per_sec', 0):.2f} QPS")
        print(f"Total KV Blocks: {best_metrics.get('total_kv_blocks', 0)}")
    print(f"{'='*60}\n")

    return max_qps, best_metrics


def main():
    """
    Main entry point with argparse CLI.
    """
    import json

    parser = argparse.ArgumentParser(
        description='Binary search to find maximum QPS meeting SLO constraints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  python qps_search.py --config test_config.json

  # With trace file
  python qps_search.py -c test_config.json --trace traces/chat.csv

  # Custom search parameters
  python qps_search.py -c test_config.json --qps-max 50.0

Config file format:
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "num_requests": 500,
    "slos": [
      {"metric": "e2e_p95_ms", "threshold_ms": 1000},
      {"metric": "ttft_mean_ms", "threshold_ms": 150}
    ],
    ...
  }

Available SLO metrics (any BLIS metric):
  End-to-End: e2e_mean_ms, e2e_median_ms, e2e_p50_ms, e2e_p90_ms, e2e_p95_ms, e2e_p99_ms, e2e_max_ms
  TTFT:       ttft_mean_ms, ttft_median_ms, ttft_p50_ms, ttft_p90_ms, ttft_p95_ms, ttft_p99_ms, ttft_max_ms
  ITL:        itl_mean_ms, itl_median_ms, itl_p50_ms, itl_p90_ms, itl_p95_ms, itl_p99_ms, itl_max_ms
        '''
    )

    # Required arguments
    parser.add_argument(
        '-c', '--config',
        required=True,
        type=str,
        help='Path to config JSON file'
    )

    # Optional arguments
    parser.add_argument(
        '-t', '--trace',
        type=str,
        default=None,
        help='Path to trace file (CSV with prompt_tokens, output_tokens)'
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

    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)

    # Get SLO configuration from config file
    slos = config.get('slos', [])

    if not slos:
        print("Error: No SLOs defined in config file")
        print("Please add 'slos' field to config.json:")
        print('  "slos": [')
        print('    {"metric": "e2e_p95_ms", "threshold_ms": 1000}')
        print('  ]')
        sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Model: {config['model']}")
    print(f"  Hardware: {config['hardware']}")
    print(f"  TP: {config['tp']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Max Scheduled Tokens: {config['max_scheduled_tokens']}")
    print(f"  Max Model Length: {config['max_model_len']}")
    print(f"  GPU Memory Utilization: {config['gpu_memory_utilization']}")

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

    # Run binary search
    max_qps, metrics = find_max_qps(
        config=config,
        slos=slos,
        trace_file=args.trace,
        qps_min=args.qps_min,
        qps_max=args.qps_max,
        qps_granularity=args.qps_granularity
    )

    if max_qps > 0:
        print("\n" + "="*60)
        print("✅ Search completed successfully!")
        print("="*60)
        print(f"\nResults:")
        print(f"  Max QPS: {max_qps:.2f}")
        print(f"\nSLO Metrics at Max QPS:")
        for slo in slos:
            metric_value = metrics.get(slo['metric'], 0)
            status = "✅" if metric_value <= slo['threshold_ms'] else "❌"
            print(f"  {status} {slo['metric']}: {metric_value:.2f} ms (SLO: {slo['threshold_ms']} ms)")

        print(f"\nAll BLIS Metrics:")

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
        print(f"    Total KV Blocks: {metrics.get('total_kv_blocks', 0)}")
        print(f"    QPS: {metrics.get('qps', max_qps):.2f}")

        print()
    else:
        print("\n❌ Search failed - no configuration meets all SLOs")
        sys.exit(1)


if __name__ == "__main__":
    main()
