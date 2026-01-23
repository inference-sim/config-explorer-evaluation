#!/usr/bin/env python3
"""
Test script for BLIS runner - Step 1 implementation
"""
import json
import sys
from pathlib import Path

from blis_runner import run_blis


def load_config(config_file: str):
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_blis.py <config.json> [qps] [num_requests]")
        print("\nExample:")
        print("  python test_blis.py test_config.json 5.0 100")
        sys.exit(1)

    config_file = sys.argv[1]
    qps = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    num_requests = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    # Load configuration
    print(f"Loading configuration from {config_file}...")
    config = load_config(config_file)

    print(f"\nConfiguration:")
    print(json.dumps(config, indent=2))

    # Run BLIS simulation
    print(f"\n{'='*60}")
    print(f"Running BLIS simulation at {qps} QPS with {num_requests} requests")
    print(f"{'='*60}")

    metrics = run_blis(config, qps, num_requests=num_requests)

    if metrics:
        print(f"\n{'='*60}")
        print("✅ Simulation completed successfully!")
        print(f"{'='*60}")
        print("\nResults:")
        print(json.dumps(metrics, indent=2))

        # Extract key metrics
        if 'e2e_p90_ms' in metrics or 'e2e_latency_p90' in metrics:
            p90_key = 'e2e_p90_ms' if 'e2e_p90_ms' in metrics else 'e2e_latency_p90'
            print(f"\nKey Metrics:")
            print(f"  P90 End-to-End Latency: {metrics[p90_key]:.2f} ms")
            if 'ttft_p90_ms' in metrics or 'ttft_p90' in metrics:
                ttft_key = 'ttft_p90_ms' if 'ttft_p90_ms' in metrics else 'ttft_p90'
                print(f"  P90 TTFT: {metrics[ttft_key]:.2f} ms")
            if 'throughput_qps' in metrics or 'throughput' in metrics:
                tput_key = 'throughput_qps' if 'throughput_qps' in metrics else 'throughput'
                print(f"  Throughput: {metrics[tput_key]:.2f} QPS")

        return 0
    else:
        print(f"\n{'='*60}")
        print("❌ Simulation failed!")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
