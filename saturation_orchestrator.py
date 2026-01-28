import json
import subprocess
import time
import os
import sys
import requests
import math
from kubernetes import client, config
from kubernetes.stream import stream


def ensure_guidellm_installed():
    """Check if GuideLLM is installed, install if not"""
    try:
        import guidellm
        print("✓ GuideLLM is already installed")
    except ImportError:
        print("Installing GuideLLM...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "guidellm"])
        print("✓ GuideLLM installed successfully")


class KubernetesManager:
    def __init__(self, namespace, image, deployment_name=None, pod_name=None):
        """Initialize Kubernetes manager"""
        config.load_kube_config()
        self.namespace = namespace
        self.image = image
        self.deployment_name = deployment_name
        self.pod_name = pod_name

        self.apps_api = client.AppsV1Api()
        self.core_api = client.CoreV1Api()

    def create_deployment(self):
        """Create deployment from vllm_validation.yaml template"""
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": self.deployment_name},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": self.deployment_name}},
                "template": {
                    "metadata": {"labels": {"app": self.deployment_name}},
                    "spec": {
                        "terminationGracePeriodSeconds": 20,
                        "restartPolicy": "Always",
                        "affinity": {
                            "nodeAffinity": {
                                "requiredDuringSchedulingIgnoredDuringExecution": {
                                    "nodeSelectorTerms": [{
                                        "matchExpressions": [
                                            {
                                                "key": "nvidia.com/gpu.product",
                                                "operator": "In",
                                                "values": ["NVIDIA-H100-80GB-HBM3"]
                                            },
                                            {
                                                "key": "nvidia.com/gpu.memory",
                                                "operator": "Gt",
                                                "values": ["50000"]
                                            }
                                        ]
                                    }]
                                }
                            }
                        },
                        "containers": [{
                            "name": "vllm",
                            "image": self.image,
                            "imagePullPolicy": "IfNotPresent",
                            "resources": {"limits": {"nvidia.com/gpu": 1}},
                            "command": ["/bin/sh", "-lc"],
                            "args": ["sleep infinity"],
                            "ports": [{"containerPort": 8000, "name": "http"}],
                            "env": [
                                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
                                {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"}
                            ]
                        }]
                    }
                }
            }
        }

        self.apps_api.create_namespaced_deployment(
            namespace=self.namespace,
            body=deployment_manifest
        )
        print(f"✓ Deployment {self.deployment_name} created")

    def find_pod_from_deployment(self):
        """Find pod from deployment name"""
        pods = self.core_api.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"app={self.deployment_name}"
        )

        if not pods.items:
            raise RuntimeError(f"No pod found for deployment {self.deployment_name}")

        pod = pods.items[0]
        self.pod_name = pod.metadata.name
        print(f"✓ Found pod {self.pod_name} for deployment {self.deployment_name}")

    def wait_for_pod_ready(self, timeout=300):
        """Poll until pod status is Running"""
        start = time.time()

        while time.time() - start < timeout:
            pods = self.core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={self.deployment_name}"
            )

            if pods.items:
                pod = pods.items[0]
                self.pod_name = pod.metadata.name

                if pod.status.phase == "Running":
                    return

            time.sleep(2)

        raise TimeoutError(f"Pod failed to become ready within {timeout}s")

    def exec_command(self, cmd, background=False):
        """Execute command in pod"""
        if background:
            cmd = f"nohup {cmd} > /dev/null 2>&1 &"

        exec_command = ["/bin/sh", "-c", cmd]

        resp = stream(
            self.core_api.connect_get_namespaced_pod_exec,
            self.pod_name,
            self.namespace,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=True
        )

        return resp

    def copy_to_pod(self, local_path, pod_path):
        """Copy local file to pod"""
        with open(local_path, 'rb') as f:
            file_data = f.read()

        # Create parent directory in pod
        pod_dir = os.path.dirname(pod_path)
        self.exec_command(f"mkdir -p {pod_dir}")

        # Write file content
        exec_command = ["sh", "-c", f"cat > {pod_path}"]
        resp = stream(
            self.core_api.connect_get_namespaced_pod_exec,
            self.pod_name,
            self.namespace,
            command=exec_command,
            stderr=True,
            stdin=True,
            stdout=True,
            tty=False,
            _preload_content=False
        )

        resp.write_stdin(file_data)
        resp.close()

    def read_file_from_pod(self, pod_path):
        """Read file content from pod using base64 to avoid truncation"""
        # Use base64 to safely transfer any file content
        result = self.exec_command(f"base64 -w 0 {pod_path}")

        # Decode base64 to get original content
        import base64
        try:
            decoded = base64.b64decode(result.strip())
            # Try to decode as UTF-8 text
            return decoded.decode('utf-8')
        except:
            # If it fails, return as bytes
            return decoded

    def list_files_from_pod(self, pod_dir):
        """List files in directory in pod"""
        result = self.exec_command(f"find {pod_dir} -type f")
        return result.strip().split('\n') if result.strip() else []

    def delete_deployment(self):
        """Delete deployment and pod"""
        self.apps_api.delete_namespaced_deployment(
            name=self.deployment_name,
            namespace=self.namespace,
            body=client.V1DeleteOptions()
        )
        print(f"✓ Deployment {self.deployment_name} deleted")


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


def build_vllm_command_string(config, port=8000):
    """Build vLLM command as string for exec"""
    cmd = f"vllm serve {config['model']}"
    cmd += f" --max-num-seqs {config['batch_size']}"
    cmd += f" --max-num-batched-tokens {config['max_scheduled_tokens']}"
    cmd += f" --max-model-len {config['max_model_len']}"
    cmd += f" --gpu-memory-utilization {config['gpu_memory_utilization']}"
    cmd += f" --block-size {config['block_size']}"
    cmd += f" --tensor-parallel-size {config['tp']}"
    cmd += f" --port {port}"
    return cmd


def build_guidellm_command_string(config, max_qps, output_dir):
    """Build GuideLLM command as string for exec"""
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

    data_json = json.dumps(data).replace('"', '\\"')

    warmup_fraction = 0.1
    adjusted_max_requests = int(math.ceil(config['num_requests'] / (1 - warmup_fraction)))

    cmd = "guidellm benchmark"
    cmd += " --target http://localhost:8000/v1"
    cmd += f" --model {config['model']}"
    cmd += " --profile constant"
    cmd += " --request-type text_completions"
    cmd += f" --rate {max_qps}"
    cmd += f" --max-requests {adjusted_max_requests}"
    cmd += f" --warmup {warmup_fraction}"
    cmd += f' --data "{data_json}"'
    cmd += f" --output-dir {output_dir}"
    return cmd


def wait_for_vllm_in_pod(k8s, timeout=300):
    """Poll health endpoints in pod until ready"""
    start = time.time()
    health_ok = False
    models_ok = False

    while time.time() - start < timeout:
        if not health_ok:
            try:
                result = k8s.exec_command("curl -s -w '%{http_code}' localhost:8000/health")
                # Check for 200 HTTP status code
                if "200" in result:
                    health_ok = True
                    print("✓ Health endpoint ready")
            except Exception as e:
                pass  # Retry on error
            time.sleep(2)
            continue

        if not models_ok:
            try:
                result = k8s.exec_command("curl -s -w '%{http_code}' localhost:8000/v1/models")
                # Check for 200 HTTP status code and some content
                if "200" in result and len(result) > 10:
                    models_ok = True
                    print("✓ Models endpoint ready")
                    return
            except Exception as e:
                pass  # Retry on error
            time.sleep(2)

    raise TimeoutError("vLLM failed to start in pod")


def wait_for_guidellm_completion(k8s, output_dir, timeout=3600):
    """Poll output directory for JSON/CSV files indicating completion"""
    start = time.time()

    while time.time() - start < timeout:
        result = k8s.exec_command(f"ls {output_dir}/*.json {output_dir}/*.csv 2>/dev/null")

        if result and (".json" in result or ".csv" in result):
            return

        time.sleep(10)

    raise TimeoutError("GuideLLM benchmark did not complete in time")


def extract_metric_from_guidellm(guidellm_results, metric_name):
    """
    Extract metric from GuideLLM benchmarks.json

    Args:
        guidellm_results: Parsed JSON from benchmarks.json
        metric_name: Metric name like "e2e_p95_ms", "ttft_p90_ms"

    Returns:
        float: Metric value in milliseconds, or None if not found
    """
    # Metric name mapping
    mapping = {
        "e2e_p90_ms": ("request_latency", "p90", True),
        "e2e_p95_ms": ("request_latency", "p95", True),
        "e2e_p99_ms": ("request_latency", "p99", True),
        "ttft_p90_ms": ("time_to_first_token_ms", "p90", False),
        "ttft_p95_ms": ("time_to_first_token_ms", "p95", False),
        "ttft_p99_ms": ("time_to_first_token_ms", "p99", False),
        "itl_p90_ms": ("inter_token_latency_ms", "p90", False),
        "itl_p95_ms": ("inter_token_latency_ms", "p95", False),
        "itl_p99_ms": ("inter_token_latency_ms", "p99", False),
    }

    if metric_name not in mapping:
        return None

    metric_key, percentile_key, needs_conversion = mapping[metric_name]

    try:
        benchmarks = guidellm_results.get("benchmarks", [])
        if not benchmarks:
            return None

        metrics = benchmarks[0].get("metrics", {})
        metric_data = metrics.get(metric_key, {})
        total_data = metric_data.get("total", {})
        percentiles = total_data.get("percentiles", {})
        value = percentiles.get(percentile_key)

        if value is None:
            return None

        # Convert seconds to milliseconds if needed
        if needs_conversion:
            value = value * 1000

        return value

    except (KeyError, IndexError, TypeError):
        return None


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
