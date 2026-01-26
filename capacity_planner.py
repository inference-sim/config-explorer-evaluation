"""
Capacity Planner - Calculate total KV cache blocks
Uses config_explorer library from llm-d-benchmark as shown in CLAUDE.md
"""
import sys
import os
from contextlib import contextmanager

from config_explorer.capacity_planner import (
    get_model_info_from_hf,
    get_model_config_from_hf,
    total_kv_cache_blocks as llm_d_total_kv_cache_blocks,
    model_memory_req,
)


@contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# Hardware memory in GB
HARDWARE_MEMORY = {
    "H100": 80,
    "A100": 80,
    "A100-40GB": 40,
    "A40": 48,
    "V100": 32,
}


def calculate_total_kv_blocks(
    model: str,
    hardware: str,
    tp: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    block_size: int = 16,
    pp: int = 1,
    dp: int = 1,
    hf_token: str = None,
    verbose: bool = False,
) -> int:
    """
    Calculate total KV cache blocks using config_explorer library.

    This follows the usage pattern from CLAUDE.md lines 262-290.

    Args:
        model: Model identifier (e.g., 'meta-llama/llama-3.1-8b-instruct')
        hardware: Hardware type (e.g., 'H100')
        tp: Tensor parallelism degree
        max_model_len: Maximum sequence length (context_len)
        gpu_memory_utilization: GPU memory utilization fraction (e.g., 0.9)
        block_size: Number of tokens per block (default: 16)
        pp: Pipeline parallelism degree (default: 1)
        dp: Data parallelism degree (default: 1)
        hf_token: Optional HuggingFace token for gated models
        verbose: If True, print detailed capacity planning info (default: True)

    Returns:
        Number of KV cache blocks
    """
    # Get GPU memory in GB
    if hardware not in HARDWARE_MEMORY:
        raise ValueError(
            f"Unknown hardware type: {hardware}. Known types: {list(HARDWARE_MEMORY.keys())}"
        )
    gpu_memory = HARDWARE_MEMORY[hardware]

    # Get model information from HuggingFace (following CLAUDE.md pattern)
    if verbose:
        print(f"Fetching model info from HuggingFace: {model}")
        model_info = get_model_info_from_hf(model, hf_token=hf_token)
        model_config = get_model_config_from_hf(model, hf_token=hf_token)
    else:
        # Suppress HuggingFace library output when not verbose
        with suppress_stdout_stderr():
            model_info = get_model_info_from_hf(model, hf_token=hf_token)
            model_config = get_model_config_from_hf(model, hf_token=hf_token)

    # Calculate total KV cache blocks (following CLAUDE.md lines 277-288)
    total_blocks = llm_d_total_kv_cache_blocks(
        model_info=model_info,
        model_config=model_config,
        context_len=max_model_len,      # max_model_len
        gpu_memory=gpu_memory,           # H100 = 80 GiB
        gpu_mem_util=gpu_memory_utilization,  # gpu_memory_utilization
        batch_size=1,
        block_size=block_size,
        tp=tp,                           # tensor parallel size
        pp=pp,                           # pipeline parallel size
        dp=dp                            # data parallel size
    )

    # Calculate model memory for display
    model_memory = model_memory_req(model_info, model_config)

    # Print capacity planning results
    if verbose:
        print(f"\nCapacity Planning Results:")
        print(f"  Model: {model}")
        print(f"  Hardware: {hardware} ({gpu_memory}GB)")
        print(f"  Tensor Parallelism: {tp}")
        print(f"  Pipeline Parallelism: {pp}")
        print(f"  Data Parallelism: {dp}")
        print(f"  GPU Memory Utilization: {gpu_memory_utilization:.2%}")
        print(f"  Max Model Length (context): {max_model_len}")
        print(f"  Block Size: {block_size} tokens")
        print(f"  Model Memory: {model_memory:.2f} GiB")
        print(f"  Total KV Blocks: {total_blocks:,}")

    return int(total_blocks)


if __name__ == "__main__":
    # Example usage from CLAUDE.md
    # total_blocks = calculate_total_kv_blocks(
    #     model="meta-llama/llama-3.1-8b-instruct",
    #     hardware="H100",
    #     tp=1,
    #     max_model_len=8192,
    #     gpu_memory_utilization=0.90,
    #     block_size=16
    # )
    total_blocks = calculate_total_kv_blocks(
        model="codellama/CodeLlama-34b-Instruct-hf",
        hardware="H100",
        tp=1,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        block_size=16
    )
    
    print(f"\nResult: {total_blocks} KV cache blocks")
