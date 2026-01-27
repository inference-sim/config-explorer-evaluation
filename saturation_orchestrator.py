import json
import subprocess
import time
import os
import sys
import requests


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
    def __init__(self, config, port=8000, log_file=None, startup_timeout=300):
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
    adjusted_max_requests = int(config['num_requests'] / (1 - warmup_fraction))

    # Build GuideLLM command
    command = [
        "guidellm",
        "--target", "http://localhost:8000/v1",
        "--model", config['model'],
        "--profile", "constant",
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate simulation results against real vLLM")
    parser.add_argument("--results", required=True, help="Simulation results JSON file")
    parser.add_argument("--config-name", required=True, help="Config name key in results file")
    parser.add_argument("--simulator", required=True, choices=["blis", "vidur"],
                        help="Simulator type (blis or vidur)")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--startup-timeout", type=int, default=300,
                        help="vLLM server startup timeout in seconds (default: 300)")
    parser.add_argument("--benchmark-timeout", type=int, default=3600,
                        help="GuideLLM benchmark timeout in seconds (default: 3600)")

    args = parser.parse_args()

    # Ensure GuideLLM is installed
    ensure_guidellm_installed()

    # Load config and max QPS
    with open(args.results) as f:
        data = json.load(f)

    config = data[args.config_name]["config"]
    max_qps = data[args.config_name]["results"][args.simulator]["max_qps"]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up vLLM server log file
    vllm_log_file = os.path.join(args.output_dir, "vllm_server.log")

    # Start vLLM server in background and run benchmark
    with VLLMServerManager(
        config,
        port=args.vllm_port,
        log_file=vllm_log_file,
        startup_timeout=args.startup_timeout
    ) as server:
        run_guidellm_benchmark(
            config,
            max_qps,
            args.output_dir,
            benchmark_timeout=args.benchmark_timeout
        )

    print(f"✓ Validation complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
