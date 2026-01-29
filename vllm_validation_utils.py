"""
Shared utilities for vLLM validation workflows

Common components used by saturation_orchestrator.py and config_validator.py
"""

import json
import time
import math
from kubernetes import client, config
from kubernetes.stream import stream


class KubernetesManager:
    """Kubernetes pod manager for vLLM validation"""

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
        result = self.exec_command(f"find {pod_dir} -type f 2>/dev/null || echo ''")
        return result.strip().split('\n') if result.strip() else []

    def delete_deployment(self):
        """Delete deployment and pod"""
        self.apps_api.delete_namespaced_deployment(
            name=self.deployment_name,
            namespace=self.namespace,
            body=client.V1DeleteOptions()
        )
        print(f"✓ Deployment {self.deployment_name} deleted")


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
        result = k8s.exec_command(f"ls {output_dir}/*.json {output_dir}/*.csv 2>/dev/null || echo ''")

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
