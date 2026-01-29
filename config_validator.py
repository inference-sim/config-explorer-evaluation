#!/usr/bin/env python3
"""
Config Validator - Validate top-N simulator configs against real vLLM

Validates simulator predictions by running real vLLM+GuideLLM benchmarks
and comparing SLO compliance and prediction accuracy.

ARCHITECTURE: Creates a separate pod per config for clean GPU memory state.
All pods are spun up in parallel and validated concurrently using ThreadPoolExecutor.
"""

import json
import time
import os
import shutil
import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from vllm_validation_utils import (
    KubernetesManager,
    build_vllm_command_string,
    build_guidellm_command_string,
    wait_for_vllm_in_pod,
    wait_for_guidellm_completion,
    extract_metric_from_guidellm
)

# Constants
POD_VLLM_LOG = "/tmp/vllm.log"
POD_GUIDELLM_LOG = "/tmp/guidellm.log"
POD_OUTPUT_DIR = "/tmp/validation_results"


def load_simulator_results(results_file):
    """Load simulator results JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)


def get_top_n_configs(results_data, top_n):
    """Return first N from successful_configs"""
    return results_data.get("successful_configs", [])[:top_n]


def build_validation_config(sim_config, metadata):
    """Merge sim_config.configuration + metadata.workload + metadata.model"""
    config = {**sim_config['configuration'], **metadata['workload']}
    config['model'] = metadata['model']
    return config


def save_file_from_pod(k8s, pod_path, local_dir, local_filename=None):
    """Helper to download and save a file from pod"""
    local_filename = local_filename or os.path.basename(pod_path)
    local_path = os.path.join(local_dir, local_filename)

    try:
        content = k8s.read_file_from_pod(pod_path)
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(local_path, mode) as f:
            f.write(content)
        return local_path
    except Exception as e:
        print(f"⚠️  Failed to copy {local_filename}: {e}")
        return None


def download_pod_results(k8s, temp_dir):
    """Download logs and results from pod to local directory"""
    os.makedirs(temp_dir, exist_ok=True)

    save_file_from_pod(k8s, POD_VLLM_LOG, temp_dir, "vllm.log")
    save_file_from_pod(k8s, POD_GUIDELLM_LOG, temp_dir, "guidellm.log")

    benchmarks_path = None
    result_files = k8s.list_files_from_pod(POD_OUTPUT_DIR)
    for pod_file in result_files:
        if pod_file:
            local_path = save_file_from_pod(k8s, pod_file, temp_dir)
            if local_path and os.path.basename(local_path) == "benchmarks.json":
                benchmarks_path = local_path

    return benchmarks_path


def build_slo_metric_result(slo, sim_config, real_ms):
    """Build SLO metric result dict"""
    metric_name = slo['metric']
    threshold_ms = slo['threshold_ms']
    simulator_ms = sim_config['slo_metrics'][metric_name]['value_ms']

    if real_ms is None:
        return {
            "threshold_ms": threshold_ms,
            "simulator_ms": simulator_ms,
            "real_ms": None,
            "difference_ms": None,
            "error_percent": None,
            "passes": False,
            "error": "Could not extract metric from results"
        }

    passes = real_ms <= threshold_ms
    difference_ms = real_ms - simulator_ms
    error_percent = (difference_ms / simulator_ms) * 100 if simulator_ms > 0 else None

    return {
        "threshold_ms": threshold_ms,
        "simulator_ms": simulator_ms,
        "real_ms": real_ms,
        "difference_ms": difference_ms,
        "error_percent": error_percent,
        "passes": passes
    }


def parse_guidellm_results(benchmarks_path, slo_constraints, sim_config):
    """Parse GuideLLM results and check SLO compliance"""
    if not benchmarks_path or not os.path.exists(benchmarks_path):
        raise RuntimeError("benchmarks.json not found in results")

    with open(benchmarks_path, 'r') as f:
        guidellm_results = json.load(f)

    benchmark_duration = None
    if guidellm_results.get("benchmarks") and len(guidellm_results["benchmarks"]) > 0:
        benchmark_duration = guidellm_results["benchmarks"][0].get("duration")

    slo_metrics = {}
    all_slos_met = True

    for slo in slo_constraints:
        metric_name = slo['metric']
        real_ms = extract_metric_from_guidellm(guidellm_results, metric_name)
        slo_metrics[metric_name] = build_slo_metric_result(slo, sim_config, real_ms)

        if not slo_metrics[metric_name]['passes']:
            all_slos_met = False

    return slo_metrics, all_slos_met, benchmark_duration


def validate_single_config_in_pod(sim_config, config, max_qps, slo_constraints, args, temp_dir):
    """
    Validate single config in its own dedicated pod.
    Creates pod -> starts vLLM -> runs GuideLLM -> downloads results -> deletes pod
    Thread-safe for parallel execution.
    """
    config_id = sim_config['config_id']
    rank = sim_config['rank']
    prefix = f"[Config {config_id}]"

    print(f"\n{prefix} {'='*60}")
    print(f"{prefix} Validating Config {config_id} (Rank {rank})")
    print(f"{prefix} {'='*60}")
    print(f"{prefix} Simulator: {args.simulator.upper()}")
    print(f"{prefix} Max QPS: {max_qps}")
    print(f"{prefix} Configuration: {sim_config['configuration']}")

    result = {
        "config_id": config_id,
        "rank": rank,
        "configuration": sim_config['configuration'],
        "max_qps": max_qps,
        "meets_slos": None,
        "slo_metrics": {},
        "status": "PENDING",
        "output_dir": temp_dir
    }

    deployment_name = f"vllm-val-config{config_id}-{int(time.time())}"
    k8s = None

    try:
        print(f"{prefix} [1/8] Creating pod {deployment_name}...")
        k8s = KubernetesManager(args.namespace, args.image, deployment_name)
        k8s.create_deployment()
        k8s.wait_for_pod_ready(timeout=600)
        print(f"{prefix} ✓ Pod ready")

        print(f"{prefix} [2/8] Setting up environment...")
        k8s.exec_command("apt-get update && apt-get install -y python3-pip curl")
        k8s.exec_command("pip install guidellm")
        print(f"{prefix} ✓ Environment setup complete")

        print(f"{prefix} [3/8] Starting vLLM server...")
        vllm_cmd = build_vllm_command_string(config, port=8000)
        k8s.exec_command(f"nohup {vllm_cmd} > {POD_VLLM_LOG} 2>&1 &", background=True)
        time.sleep(10)

        print(f"{prefix} [4/8] Waiting for vLLM health check...")
        wait_for_vllm_in_pod(k8s, timeout=args.startup_timeout)
        print(f"{prefix} ✓ vLLM server ready")

        print(f"{prefix} [5/8] Running GuideLLM at {max_qps} QPS...")
        k8s.exec_command(f"mkdir -p {POD_OUTPUT_DIR}")
        guidellm_cmd = build_guidellm_command_string(config, max_qps, POD_OUTPUT_DIR)
        k8s.exec_command(f"nohup {guidellm_cmd} > {POD_GUIDELLM_LOG} 2>&1 &", background=True)

        print(f"{prefix} [6/8] Waiting for GuideLLM completion...")
        wait_for_guidellm_completion(k8s, POD_OUTPUT_DIR, timeout=args.benchmark_timeout)
        print(f"{prefix} ✓ GuideLLM benchmark completed")

        print(f"{prefix} [7/8] Downloading results to {temp_dir}...")
        benchmarks_path = download_pod_results(k8s, temp_dir)
        print(f"{prefix} ✓ Results downloaded to {temp_dir}/")

        print(f"{prefix} [8/8] Parsing metrics and checking SLO compliance...")
        slo_metrics, all_slos_met, benchmark_duration = parse_guidellm_results(
            benchmarks_path, slo_constraints, sim_config
        )

        result['slo_metrics'] = slo_metrics
        result['meets_slos'] = all_slos_met
        result['status'] = "SUCCESS"
        if benchmark_duration is not None:
            result['benchmark_duration_seconds'] = round(benchmark_duration, 2)

        print(f"{prefix} ✓ Metrics parsed. SLOs met: {all_slos_met}")
        if benchmark_duration is not None:
            print(f"{prefix}   GuideLLM benchmark duration: {benchmark_duration:.2f}s")

        print(f"{prefix} ✓ Validation complete")
        print(f"{prefix} {'='*60}\n")

        return result

    except Exception as e:
        print(f"{prefix} ✗ Validation failed: {e}")
        result['status'] = "ERROR"
        result['error'] = str(e)
        result['meets_slos'] = False

        if k8s:
            print(f"{prefix} Attempting to save logs for debugging...")
            download_pod_results(k8s, temp_dir)

        print(f"{prefix} {'='*60}\n")
        return result

    finally:
        if k8s and not args.keep_deployment:
            try:
                print(f"{prefix} Deleting pod {deployment_name}...")
                k8s.delete_deployment()
                print(f"{prefix} ✓ Pod deleted")
            except Exception as e:
                print(f"{prefix} ⚠️  Failed to delete pod: {e}")


def generate_validation_report(all_results, output_file, simulator, simulator_runtime=0):
    """Generate validation report JSON with summary and per-config results"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_tested = len(all_results)
    meeting_slos = sum(1 for r in all_results if r.get('meets_slos') is True)
    failing_slos = sum(1 for r in all_results if r.get('meets_slos') is False)

    total_guidellm_runtime = sum(
        r.get('benchmark_duration_seconds', 0)
        for r in all_results
        if r.get('status') == 'SUCCESS'
    )

    report = {
        "timestamp": timestamp,
        "simulator": simulator.upper(),
        "summary": {
            "total_configs_tested": total_tested,
            "configs_meeting_slos": meeting_slos,
            "configs_failing_slos": failing_slos,
            "total_guidellm_runtime_seconds": round(total_guidellm_runtime, 2),
            "total_simulator_runtime_seconds": round(simulator_runtime, 2)
        },
        "results": all_results
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    runtime_minutes = total_guidellm_runtime / 60
    runtime_hours = total_guidellm_runtime / 3600

    print(f"\n{'='*60}")
    print("VALIDATION REPORT GENERATED")
    print(f"{'='*60}")
    print(f"Total configs tested: {total_tested}")
    print(f"Configs meeting SLOs: {meeting_slos}")
    print(f"Configs failing SLOs: {failing_slos}")
    print(f"Total GuideLLM runtime: {total_guidellm_runtime:.2f}s ({runtime_minutes:.2f}m / {runtime_hours:.2f}h)")
    print(f"\nReport saved to: {output_file}")
    print(f"{'='*60}\n")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Validate top-N simulator configs against real vLLM (creates separate pod per config)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate top 3 BLIS configs in parallel (creates 3 pods concurrently)
  python config_validator.py \\
    --simulator blis \\
    --results results/config_exp/blis_config_exploration_lowprefix.json \\
    --top-n 3 \\
    --namespace diya \\
    --output-file validation_report.json

  # Keep pods and logs after validation (for debugging)
  python config_validator.py \\
    --simulator blis \\
    --results results/config_exp/blis_config_exploration.json \\
    --top-n 3 \\
    --namespace diya \\
    --output-file validation_report.json \\
    --keep-deployment \\
    --keep-logs
        """
    )

    parser.add_argument("--simulator", required=True, choices=["blis", "vidur"],
                        help="Simulator type (blis or vidur)")
    parser.add_argument("--results", required=True,
                        help="Simulator results JSON file from parallel_search.py")
    parser.add_argument("--top-n", type=int, required=True,
                        help="Number of top configs to validate")
    parser.add_argument("--namespace", default="diya",
                        help="Kubernetes namespace")
    parser.add_argument("--image", default="vllm/vllm-openai:v0.14.0",
                        help="vLLM container image")
    parser.add_argument("--output-file", required=True,
                        help="Output file path for validation report")
    parser.add_argument("--startup-timeout", type=int, default=600,
                        help="vLLM startup timeout in seconds")
    parser.add_argument("--benchmark-timeout", type=int, default=3600,
                        help="GuideLLM benchmark timeout in seconds")
    parser.add_argument("--keep-deployment", action="store_true",
                        help="Keep pods after validation (for debugging)")
    parser.add_argument("--keep-logs", action="store_true",
                        help="Keep per-config log directories (for debugging)")

    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_base = os.path.splitext(args.output_file)[0]
    results_base_dir = f"{output_base}_configs"
    os.makedirs(results_base_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VALIDATING {args.simulator.upper()} CONFIGS")
    print(f"{'='*60}\n")

    results_data = load_simulator_results(args.results)
    configs = get_top_n_configs(results_data, args.top_n)
    metadata = results_data['metadata']
    slo_constraints = metadata['slo_constraints']

    print(f"Found {len(configs)} top configs to validate")
    print(f"Running all {len(configs)} validations in parallel...\n")

    # Prepare validation tasks
    validation_tasks = []
    for sim_config in configs:
        config = build_validation_config(sim_config, metadata)
        max_qps = sim_config['max_qps']
        temp_dir = os.path.join(results_base_dir, f"config_{sim_config['config_id']}_rank_{sim_config['rank']}")

        validation_tasks.append({
            'sim_config': sim_config,
            'config': config,
            'max_qps': max_qps,
            'temp_dir': temp_dir
        })

    # Run all validations in parallel using ThreadPoolExecutor
    all_results = []
    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(
                validate_single_config_in_pod,
                task['sim_config'],
                task['config'],
                task['max_qps'],
                slo_constraints,
                args,
                task['temp_dir']
            ): task['sim_config']
            for task in validation_tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_config):
            sim_config = future_to_config[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"✓ Config {sim_config['config_id']} (rank {sim_config['rank']}) completed")
            except Exception as e:
                print(f"✗ Config {sim_config['config_id']} (rank {sim_config['rank']}) failed with exception: {e}")

    # Sort results by rank for consistent reporting
    all_results.sort(key=lambda x: x['rank'])

    # Extract simulator runtime from input results
    simulator_runtime = results_data.get('summary', {}).get('total_search_runtime_seconds', 0)

    report_path = generate_validation_report(all_results, args.output_file, args.simulator, simulator_runtime)

    # Cleanup: Delete temporary config directories (keep only report)
    if not args.keep_logs:
        try:
            if os.path.exists(results_base_dir):
                shutil.rmtree(results_base_dir)
                print(f"✓ Cleaned up temporary directory: {results_base_dir}/")
        except Exception as e:
            print(f"⚠️  Failed to cleanup temporary directory: {e}")
    else:
        print(f"✓ Keeping logs in: {results_base_dir}/")

    print(f"\n✓ All validations complete. Report: {report_path}")


if __name__ == "__main__":
    main()
