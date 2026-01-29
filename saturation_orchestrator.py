import json
import subprocess
import time
import os
import sys
import requests
import math

# Import shared validation utilities
from vllm_validation_utils import (
    KubernetesManager,
    build_vllm_command_string,
    build_guidellm_command_string,
    wait_for_vllm_in_pod,
    wait_for_guidellm_completion,
    extract_metric_from_guidellm
)


def ensure_guidellm_installed():
    """Check if GuideLLM is installed, install if not"""
    try:
        import guidellm
        print("✓ GuideLLM is already installed")
    except ImportError:
        print("Installing GuideLLM...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "guidellm"])
        print("✓ GuideLLM installed successfully")


class VLLMServerManager:
    def __init__(self, config, port=8000, log_file=None, startup_timeout=600):
        """Build vLLM command from config"""
        self.port = port
        self.process = None
        self.log_file = log_file
        self.startup_timeout = startup_timeout
        self.model_name = config['model']

        # Map config to vLLM arguments
        self.command = [
            "vllm", "serve", config['model'],
            "--max-num-seqs", str(config['batch_size']),
            "--max-num-batched-tokens", str(config['max_scheduled_tokens']),
            "--max-model-len", str(config['max_model_len']),
            "--gpu-memory-utilization", str(config['gpu_memory_utilization']),
            "--block-size", str(config['block_size']),
            "--tensor-parallel-size", str(config['tp']),
            "--port", str(port)
        ]

    def _check_health(self):
        """Check if server health endpoint responds"""
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _check_models_endpoint(self):
        """Check if /v1/models endpoint is available"""
        try:
            response = requests.get(f"http://localhost:{self.port}/v1/models", timeout=5)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def start(self):
        """Start server in background and wait for full readiness"""
        # Open log file for stdout/stderr
        log_handle = open(self.log_file, 'w') if self.log_file else subprocess.DEVNULL

        # Start vLLM server in background
        self.process = subprocess.Popen(
            self.command,
            stdout=log_handle,
            stderr=subprocess.STDOUT
        )

        print(f"Starting vLLM server (timeout: {self.startup_timeout}s)...")

        # Poll health endpoint
        start_time = time.time()
        health_ok = False
        models_ok = False

        while time.time() - start_time < self.startup_timeout:
            # Step 1: Check health endpoint
            if not health_ok:
                health_ok = self._check_health()
                if health_ok:
                    print("✓ Health endpoint ready")
                time.sleep(2)
                continue

            # Step 2: Check models endpoint
            if not models_ok:
                models_ok = self._check_models_endpoint()
                if models_ok:
                    print(f"✓ vLLM server fully ready on port {self.port}")
                    return
                time.sleep(2)
                continue

        # Timeout occurred
        self.stop()
        raise TimeoutError(f"vLLM server failed to become ready within {self.startup_timeout}s")

    def stop(self):
        """Gracefully stop server"""
        if self.process:
            self.process.terminate()
            self.process.wait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def run_guidellm_benchmark(config, target_qps, output_dir, benchmark_timeout=3600):
    """Build and execute GuideLLM command with timeout"""
    # Build data JSON with flat structure
    data = {
        "prompt_tokens": config['prompt_tokens'],
        "output_tokens": config['output_tokens'],
        "prefix_tokens": config['prefix_tokens'],
        "prompt_tokens_min": config['prompt_tokens_min'],
        "prompt_tokens_max": config['prompt_tokens_max'],
        "prompt_tokens_stdev": config['prompt_tokens_stdev'],
        "output_tokens_min": config['output_tokens_min'],
        "output_tokens_max": config['output_tokens_max'],
        "output_tokens_stdev": config['output_tokens_stdev']
    }

    data_json = json.dumps(data)

    # Adjust max-requests to account for 10% warmup
    # If warmup is 0.1, then actual benchmark requests = max_requests * 0.9
    # So to get num_requests benchmark requests, we need max_requests = num_requests / 0.9
    warmup_fraction = 0.1
    adjusted_max_requests = math.ceil(config['num_requests'] / (1 - warmup_fraction))

    # Build GuideLLM command
    command = [
        "guidellm", "benchmark",
        "--target", "http://localhost:8000/v1",
        "--model", config['model'],
        "--profile", "constant",
        "--request-type", "text_completions",
        "--rate", str(target_qps),
        "--max-requests", str(adjusted_max_requests),
        "--warmup", str(warmup_fraction),
        "--data", data_json,
        "--output-dir", output_dir
    ]

    print(f"Running GuideLLM benchmark (timeout: {benchmark_timeout}s)...")
    print(f"  Warmup: {warmup_fraction} ({int(adjusted_max_requests * warmup_fraction)} requests)")
    print(f"  Benchmark: {config['num_requests']} requests")
    try:
        subprocess.run(command, check=True, timeout=benchmark_timeout)
        print(f"✓ GuideLLM benchmark completed")
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"GuideLLM benchmark exceeded timeout of {benchmark_timeout}s")


def generate_validation_report(simulator_name, config_name, simulator_max_qps, slo_constraints, benchmarks_path):
    """
    Print validation report showing SLO compliance at simulator's max QPS

    Args:
        simulator_name: Name of simulator (blis/vidur)
        config_name: Config identifier
        simulator_max_qps: QPS predicted by simulator
        slo_constraints: List of SLO constraints from config
        benchmarks_path: Path to GuideLLM benchmarks.json
    """
    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}")

    # Load GuideLLM results
    try:
        with open(benchmarks_path, 'r') as f:
            guidellm_results = json.load(f)
    except Exception as e:
        print(f"✗ Failed to load benchmarks.json: {e}")
        return

    # Check SLO compliance
    print(f"\nSimulator: {simulator_name.upper()}")
    print(f"Config: {config_name}")
    print(f"QPS Tested: {simulator_max_qps}")
    print(f"\nSLO Compliance at Simulator's Max QPS:")
    print(f"{'-'*60}")

    all_pass = True

    for slo in slo_constraints:
        metric_name = slo['metric']
        threshold_ms = slo['threshold_ms']

        # Extract real metric value
        real_value_ms = extract_metric_from_guidellm(guidellm_results, metric_name)

        if real_value_ms is None:
            print(f"✗ {metric_name}: Could not extract from results")
            all_pass = False
        else:
            passes = real_value_ms <= threshold_ms
            status_icon = "✓" if passes else "✗"

            print(f"{status_icon} {metric_name}:")
            print(f"    Threshold: {threshold_ms:.2f} ms")
            print(f"    Real vLLM: {real_value_ms:.2f} ms")
            print(f"    Status: {'PASS' if passes else 'FAIL'}")

            if not passes:
                all_pass = False

    print(f"{'-'*60}")
    print(f"\nOverall: {'✓ ALL SLOs MET' if all_pass else '✗ SOME SLOs FAILED'}")
    print(f"{'='*60}\n")


def run_pod_validation(config, max_qps, args):
    """Run validation in existing Kubernetes pod"""
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Connect to existing deployment
    k8s = KubernetesManager(args.namespace, args.image, deployment_name=args.deployment_name)
    k8s.find_pod_from_deployment()

    # 1. Setup environment
    k8s.exec_command("apt-get update && apt-get install -y python3-pip curl")
    k8s.exec_command("pip install guidellm")
    print("✓ Environment setup complete")

    # 2. Start vLLM server in background
    vllm_cmd = build_vllm_command_string(config, port=8000)
    k8s.exec_command(f"nohup {vllm_cmd} > /tmp/vllm.log 2>&1 &", background=True)
    time.sleep(10)

    # 3. Wait for vLLM ready
    wait_for_vllm_in_pod(k8s, timeout=args.startup_timeout)
    print("✓ vLLM server ready")

    # 4. Create output directory and run GuideLLM in background
    k8s.exec_command("mkdir -p /output/validation_results")
    guidellm_cmd = build_guidellm_command_string(config, max_qps, "/output/validation_results")
    k8s.exec_command(f"nohup {guidellm_cmd} > /tmp/guidellm.log 2>&1 &", background=True)
    print("Running GuideLLM benchmark...")

    # 5. Wait for GuideLLM completion
    wait_for_guidellm_completion(k8s, "/output/validation_results", timeout=args.benchmark_timeout)
    print("✓ GuideLLM benchmark completed")

    # 6. Copy logs
    vllm_log = k8s.read_file_from_pod("/tmp/vllm.log")
    with open(os.path.join(args.output_dir, "vllm.log"), "w") as f:
        f.write(vllm_log)

    guidellm_log = k8s.read_file_from_pod("/tmp/guidellm.log")
    with open(os.path.join(args.output_dir, "guidellm.log"), "w") as f:
        f.write(guidellm_log)

    # 7. Copy GuideLLM results
    result_files = k8s.list_files_from_pod("/output/validation_results")
    for pod_file in result_files:
        if pod_file:
            file_content = k8s.read_file_from_pod(pod_file)
            local_filename = os.path.basename(pod_file)
            local_path = os.path.join(args.output_dir, local_filename)

            # Write as text or binary depending on what we got back
            if isinstance(file_content, bytes):
                with open(local_path, "wb") as f:
                    f.write(file_content)
            else:
                with open(local_path, "w") as f:
                    f.write(file_content)

    print(f"✓ Results saved to {args.output_dir}/")


def run_k8s_validation(config, max_qps, args):
    """Run validation in Kubernetes pod (creates new deployment)"""
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    deployment_name = f"vllm-validation-{int(time.time())}"

    # 1. Create deployment
    k8s = KubernetesManager(args.namespace, args.image, deployment_name)
    k8s.create_deployment()
    k8s.wait_for_pod_ready(timeout=600)
    print("✓ Pod ready")

    # 2. Setup environment
    k8s.exec_command("apt-get update && apt-get install -y python3-pip curl")
    k8s.exec_command("pip install guidellm")
    print("✓ Environment setup complete")

    # 3. Start vLLM server in background
    vllm_cmd = build_vllm_command_string(config, port=8000)
    k8s.exec_command(f"nohup {vllm_cmd} > /tmp/vllm.log 2>&1 &", background=True)
    time.sleep(10)

    # 4. Wait for vLLM ready
    wait_for_vllm_in_pod(k8s, timeout=args.startup_timeout)
    print("✓ vLLM server ready")

    # 5. Create output directory and run GuideLLM in background
    k8s.exec_command("mkdir -p /output/validation_results")
    guidellm_cmd = build_guidellm_command_string(config, max_qps, "/output/validation_results")
    k8s.exec_command(f"nohup {guidellm_cmd} > /tmp/guidellm.log 2>&1 &", background=True)
    print("Running GuideLLM benchmark...")

    # 6. Wait for GuideLLM completion
    wait_for_guidellm_completion(k8s, "/output/validation_results", timeout=args.benchmark_timeout)
    print("✓ GuideLLM benchmark completed")

    # 7. Copy logs
    vllm_log = k8s.read_file_from_pod("/tmp/vllm.log")
    with open(os.path.join(args.output_dir, "vllm.log"), "w") as f:
        f.write(vllm_log)

    guidellm_log = k8s.read_file_from_pod("/tmp/guidellm.log")
    with open(os.path.join(args.output_dir, "guidellm.log"), "w") as f:
        f.write(guidellm_log)

    # 8. Copy GuideLLM results
    result_files = k8s.list_files_from_pod("/output/validation_results")
    for pod_file in result_files:
        if pod_file:
            file_content = k8s.read_file_from_pod(pod_file)
            local_filename = os.path.basename(pod_file)
            with open(os.path.join(args.output_dir, local_filename), "w") as f:
                f.write(file_content)

    print(f"✓ Results saved to {args.output_dir}/")

    # 9. Cleanup
    if not args.keep_deployment:
        k8s.delete_deployment()
        print("✓ Deployment cleaned up")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate simulation results against real vLLM")
    parser.add_argument("--results", required=True, help="Simulation results JSON file")
    parser.add_argument("--config-name", required=True, help="Config name key in results file")
    parser.add_argument("--simulator", required=True, choices=["blis", "vidur"],
                        help="Simulator type (blis or vidur)")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--startup-timeout", type=int, default=600,
                        help="vLLM server startup timeout in seconds (default: 600)")
    parser.add_argument("--benchmark-timeout", type=int, default=3600,
                        help="GuideLLM benchmark timeout in seconds (default: 3600)")
    parser.add_argument("--use-k8s", action="store_true",
                        help="Create new Kubernetes deployment")
    parser.add_argument("--deployment-name", type=str, default=None,
                        help="Existing deployment name to use (alternative to --use-k8s)")
    parser.add_argument("--namespace", default="diya",
                        help="Kubernetes namespace")
    parser.add_argument("--image", default="vllm/vllm-openai:v0.14.0",
                        help="vLLM container image")
    parser.add_argument("--keep-deployment", action="store_true",
                        help="Keep deployment after completion (only for --use-k8s)")

    args = parser.parse_args()

    # Validate arguments
    if not args.use_k8s and not args.deployment_name:
        parser.error("Either --use-k8s or --deployment-name must be specified")

    if args.use_k8s and args.deployment_name:
        parser.error("Cannot specify both --use-k8s and --deployment-name")

    # Load config and max QPS
    with open(args.results) as f:
        data = json.load(f)

    config = data[args.config_name]["config"]
    max_qps = data[args.config_name]["results"][args.simulator]["max_qps"]

    # Extract SLO constraints from config
    slo_constraints = config.get("slos", [])

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_k8s:
        # Create new deployment and run validation
        run_k8s_validation(config, max_qps, args)
    else:
        # Use existing pod
        run_pod_validation(config, max_qps, args)

    print(f"✓ Validation complete. Results in {args.output_dir}/")

    # Generate validation report
    benchmarks_path = os.path.join(args.output_dir, "benchmarks.json")
    if slo_constraints and os.path.exists(benchmarks_path):
        generate_validation_report(
            simulator_name=args.simulator,
            config_name=args.config_name,
            simulator_max_qps=max_qps,
            slo_constraints=slo_constraints,
            benchmarks_path=benchmarks_path
        )
    elif not slo_constraints:
        print(f"\n⚠️  No SLO constraints found in config")
    elif not os.path.exists(benchmarks_path):
        print(f"\n⚠️  benchmarks.json not found")


if __name__ == "__main__":
    main()
