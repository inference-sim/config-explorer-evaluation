#!/usr/bin/env python3
"""
Compare BLIS and Vidur simulator results.

Generates visualizations and summary metrics comparing two simulator outputs.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(blis_file: str, vidur_file: str) -> Tuple[Dict, Dict]:
    """Load BLIS and Vidur result JSON files."""
    with open(blis_file, 'r') as f:
        blis_data = json.load(f)
    with open(vidur_file, 'r') as f:
        vidur_data = json.load(f)
    return blis_data, vidur_data


def calculate_metrics(blis_data: Dict, vidur_data: Dict) -> Dict[str, Any]:
    """Calculate comparison metrics dynamically."""
    blis_summary = blis_data['summary']
    vidur_summary = vidur_data['summary']

    blis_success = blis_summary['successful_configs']
    blis_total = blis_summary['total_configs_evaluated']
    vidur_success = vidur_summary['successful_configs']
    vidur_total = vidur_summary['total_configs_evaluated']

    metrics = {
        'blis_success_rate': (blis_success / blis_total * 100) if blis_total > 0 else 0,
        'vidur_success_rate': (vidur_success / vidur_total * 100) if vidur_total > 0 else 0,
        'blis_best_qps': blis_summary['best_max_qps'],
        'vidur_best_qps': vidur_summary['best_max_qps'],
        'blis_total_runtime': blis_summary['total_search_runtime_seconds'],
        'vidur_total_runtime': vidur_summary['total_search_runtime_seconds'],
        'speedup': vidur_summary['total_search_runtime_seconds'] / blis_summary['total_search_runtime_seconds'] if blis_summary['total_search_runtime_seconds'] > 0 else 0,
        'qps_improvement': ((blis_summary['best_max_qps'] - vidur_summary['best_max_qps']) / vidur_summary['best_max_qps'] * 100) if vidur_summary['best_max_qps'] > 0 else 0,
        'blis_avg_runtime': blis_summary['total_search_runtime_seconds'] / blis_total if blis_total > 0 else 0,
        'vidur_avg_runtime': vidur_summary['total_search_runtime_seconds'] / vidur_total if vidur_total > 0 else 0,
    }

    return metrics


def plot_runtime_comparison(blis_configs: List[Dict], vidur_configs: List[Dict],
                            vidur_failed: List[Dict], blis_total: float, vidur_total: float,
                            output_dir: Path):
    """Plot 1: Runtime comparison - both total and average."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Total Runtime
    categories = ['BLIS', 'Vidur']
    total_runtimes = [blis_total, vidur_total]

    bars1 = ax1.bar(categories, total_runtimes, color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.6)

    ax1.set_ylabel('Total Runtime (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Search Runtime', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Average Runtime per Config
    blis_avg = np.mean([c['runtime_seconds'] for c in blis_configs]) if blis_configs else 0
    vidur_avg_success = np.mean([c['runtime_seconds'] for c in vidur_configs]) if vidur_configs else 0

    all_vidur = vidur_configs + vidur_failed
    vidur_avg_all = np.mean([c['runtime_seconds'] for c in all_vidur]) if all_vidur else 0

    x = np.arange(2)
    width = 0.35

    avg_categories = ['Successful Only', 'All Configs']
    blis_avgs = [blis_avg, blis_avg]  # BLIS has 100% success
    vidur_avgs = [vidur_avg_success, vidur_avg_all]

    bars2 = ax2.bar(x - width/2, blis_avgs, width,
                   label='BLIS', color='#2ecc71', alpha=0.8)
    bars3 = ax2.bar(x + width/2, vidur_avgs, width,
                   label='Vidur', color='#e74c3c', alpha=0.8)

    ax2.set_ylabel('Average Runtime (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Runtime per Config', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(avg_categories)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Runtime Comparison: BLIS vs Vidur', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '01_runtime_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_qps_comparison(blis_configs: List[Dict], vidur_configs: List[Dict],
                       vidur_failed: List[Dict], output_dir: Path):
    """Plot 2: Max QPS comparison - top configs sorted by performance."""
    # Create dict for Vidur QPS lookup
    vidur_qps_dict = {c['config_id']: c['max_qps'] for c in vidur_configs}

    # First, identify which parameters vary across configs
    varying_params = set()
    param_values = {
        'tp': set(),
        'batch_size': set(),
        'max_scheduled_tokens': set(),
        'max_model_len': set(),
        'gpu_memory_utilization': set(),
        'block_size': set()
    }

    for c in blis_configs:
        config = c['configuration']
        param_values['tp'].add(config.get('tp', 1))
        param_values['batch_size'].add(config.get('batch_size', 0))
        param_values['max_scheduled_tokens'].add(config.get('max_scheduled_tokens', 0))
        param_values['max_model_len'].add(config.get('max_model_len', 0))
        param_values['gpu_memory_utilization'].add(config.get('gpu_memory_utilization', 0))
        param_values['block_size'].add(config.get('block_size', 16))

    for param, values in param_values.items():
        if len(values) > 1:
            varying_params.add(param)

    # Prepare data: sort by BLIS QPS (descending) and take top 5
    config_data = []
    for c in blis_configs:
        config_id = c['config_id']
        config_params = c['configuration']

        # Create readable label with all varying parameters
        label_parts = []

        if 'tp' in varying_params:
            label_parts.append(f"TP={config_params.get('tp', 1)}")
        if 'batch_size' in varying_params:
            label_parts.append(f"BS={config_params.get('batch_size', 0)}")
        if 'max_scheduled_tokens' in varying_params:
            label_parts.append(f"MST={config_params.get('max_scheduled_tokens', 0)}")
        if 'max_model_len' in varying_params:
            label_parts.append(f"MML={config_params.get('max_model_len', 0)}")
        if 'gpu_memory_utilization' in varying_params:
            label_parts.append(f"GPU={config_params.get('gpu_memory_utilization', 0):.2f}")
        if 'block_size' in varying_params:
            label_parts.append(f"BLK={config_params.get('block_size', 16)}")

        # Format label with line breaks for readability (max 3 params per line)
        if len(label_parts) <= 3:
            label = ", ".join(label_parts)
        else:
            line1 = ", ".join(label_parts[:3])
            line2 = ", ".join(label_parts[3:])
            label = f"{line1}\n{line2}"

        config_data.append({
            'config_id': config_id,
            'label': label,
            'blis_qps': c['max_qps'],
            'vidur_qps': vidur_qps_dict.get(config_id, 0),
            'vidur_failed': config_id not in vidur_qps_dict
        })

    # Sort by BLIS QPS and take top 12
    config_data.sort(key=lambda x: x['blis_qps'], reverse=True)
    top_configs = config_data[:5]

    # Reverse for plotting (highest at top)
    top_configs.reverse()

    labels = [c['label'] for c in top_configs]
    blis_qps = [c['blis_qps'] for c in top_configs]
    vidur_qps = [c['vidur_qps'] for c in top_configs]

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(labels))
    height = 0.35

    # Plot bars
    bars1 = ax.barh(y_pos + height/2, blis_qps, height,
                    label='BLIS', color='#2ecc71', alpha=0.8)
    bars2 = ax.barh(y_pos - height/2, vidur_qps, height,
                    label='Vidur', color='#e74c3c', alpha=0.8)

    # Add value labels on bars
    for i, (bar, qps) in enumerate(zip(bars1, blis_qps)):
        if qps > 0:
            ax.text(qps + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{qps:.2f}',
                   va='center', fontsize=9, fontweight='bold')

    for i, (bar, qps) in enumerate(zip(bars2, vidur_qps)):
        if qps > 0:
            ax.text(qps + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{qps:.2f}',
                   va='center', fontsize=9, fontweight='bold')
        else:
            # Mark as failed
            ax.text(0.05, bar.get_y() + bar.get_height()/2,
                   'FAILED',
                   va='center', fontsize=8, style='italic', color='darkred')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Max QPS', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Configurations sorted by Max QPS',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    # Add note about config parameters
    note_text = 'BS = Batch Size, MST = Max Scheduled Tokens, MML = Max Model Length'
    ax.text(0.02, 0.98, note_text,
           transform=ax.transAxes, fontsize=8, style='italic',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / '02_qps_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_success_rate(blis_summary: Dict, vidur_summary: Dict, output_dir: Path):
    """Plot 3: Success rate comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # BLIS pie chart
    blis_success = blis_summary['successful_configs']
    blis_failed = blis_summary['failed_configs']

    ax1.pie([blis_success, blis_failed], labels=['Success', 'Failed'],
            autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
    ax1.set_title(f'BLIS Success Rate\n({blis_success}/{blis_summary["total_configs_evaluated"]} configs)',
                  fontsize=12, fontweight='bold')

    # Vidur pie chart
    vidur_success = vidur_summary['successful_configs']
    vidur_failed = vidur_summary['failed_configs']

    ax2.pie([vidur_success, vidur_failed], labels=['Success', 'Failed'],
            autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
    ax2.set_title(f'Vidur Success Rate\n({vidur_success}/{vidur_summary["total_configs_evaluated"]} configs)',
                  fontsize=12, fontweight='bold')

    plt.suptitle('Configuration Success Rates', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '03_success_rate.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_qps_distribution(blis_configs: List[Dict], vidur_configs: List[Dict], output_dir: Path):
    """Plot 4: QPS variation across all configuration parameters."""
    # Extract all parameters
    def extract_all_params(configs):
        data = {
            'tp': [],
            'batch_size': [],
            'max_scheduled_tokens': [],
            'max_model_len': [],
            'gpu_memory_utilization': [],
            'block_size': [],
            'qps': []
        }
        for c in configs:
            config = c['configuration']
            data['tp'].append(config.get('tp', 1))
            data['batch_size'].append(config.get('batch_size', 0))
            data['max_scheduled_tokens'].append(config.get('max_scheduled_tokens', 0))
            data['max_model_len'].append(config.get('max_model_len', 0))
            data['gpu_memory_utilization'].append(config.get('gpu_memory_utilization', 0))
            data['block_size'].append(config.get('block_size', 16))
            data['qps'].append(c['max_qps'])
        return data

    blis_data = extract_all_params(blis_configs)
    vidur_data = extract_all_params(vidur_configs)

    # Define all potential parameters to plot
    all_params = [
        ('tp', 'Tensor Parallelism (TP)', lambda x: f'{x}'),
        ('batch_size', 'Batch Size', lambda x: f'{x}'),
        ('max_scheduled_tokens', 'Max Scheduled Tokens', lambda x: f'{x}'),
        ('max_model_len', 'Max Model Length', lambda x: f'{x}'),
        ('gpu_memory_utilization', 'GPU Memory Utilization', lambda x: f'{x:.2f}'),
        ('block_size', 'Block Size', lambda x: f'{x}')
    ]

    # Filter to only show parameters with more than 1 unique value
    params = []
    for param_key, param_label, format_func in all_params:
        unique_values = set(blis_data[param_key])
        if len(unique_values) > 1:
            params.append((param_key, param_label, format_func))

    if not params:
        print("Warning: No varying parameters found for QPS distribution plot")
        return

    # Determine grid size based on number of varying parameters
    num_params = len(params)
    if num_params <= 2:
        nrows, ncols = 1, num_params
        figsize = (7 * num_params, 5)
    elif num_params <= 4:
        nrows, ncols = 2, 2
        figsize = (14, 10)
    else:
        nrows, ncols = 3, 2
        figsize = (14, 12)

    # Create subplot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    axes = axes.flatten()

    width = 0.35

    for idx, (param_key, param_label, format_func) in enumerate(params):
        ax = axes[idx]

        # Get unique values for this parameter
        param_values = sorted(list(set(blis_data[param_key])))

        # Group QPS by parameter value
        blis_by_param = {val: [] for val in param_values}
        vidur_by_param = {val: [] for val in param_values}

        for val, qps in zip(blis_data[param_key], blis_data['qps']):
            blis_by_param[val].append(qps)

        for val, qps in zip(vidur_data[param_key], vidur_data['qps']):
            vidur_by_param[val].append(qps)

        # Calculate means
        blis_means = [np.mean(blis_by_param[val]) if blis_by_param[val] else 0 for val in param_values]
        vidur_means = [np.mean(vidur_by_param[val]) if vidur_by_param[val] else 0 for val in param_values]

        # Plot bars
        positions = np.arange(len(param_values))
        ax.bar(positions - width/2, blis_means, width, label='BLIS',
               color='#2ecc71', alpha=0.8)
        ax.bar(positions + width/2, vidur_means, width, label='Vidur',
               color='#e74c3c', alpha=0.8)

        # Formatting
        ax.set_xlabel(param_label, fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg Max QPS', fontsize=10, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels([format_func(val) for val in param_values], rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars if there aren't too many
        if len(param_values) <= 4:
            for i, (b_mean, v_mean) in enumerate(zip(blis_means, vidur_means)):
                if b_mean > 0:
                    ax.text(i - width/2, b_mean + 0.1, f'{b_mean:.1f}',
                           ha='center', va='bottom', fontsize=8)
                if v_mean > 0:
                    ax.text(i + width/2, v_mean + 0.1, f'{v_mean:.1f}',
                           ha='center', va='bottom', fontsize=8)

    # Hide unused subplots
    for idx in range(num_params, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Max QPS Variation Across Configuration Parameters',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_dir / '04_qps_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def display_summary_metrics(blis_data: Dict, vidur_data: Dict, metrics: Dict):
    """Display comprehensive summary metrics."""
    print("\n" + "="*80)
    print("SIMULATOR COMPARISON SUMMARY")
    print("="*80)

    print("\n📊 SUCCESS RATES:")
    print(f"  BLIS:  {metrics['blis_success_rate']:.1f}% "
          f"({blis_data['summary']['successful_configs']}/{blis_data['summary']['total_configs_evaluated']} configs)")
    print(f"  Vidur: {metrics['vidur_success_rate']:.1f}% "
          f"({vidur_data['summary']['successful_configs']}/{vidur_data['summary']['total_configs_evaluated']} configs)")

    print("\n🚀 MAX QPS ACHIEVED:")
    print(f"  BLIS:  {metrics['blis_best_qps']:.2f} QPS")
    print(f"  Vidur: {metrics['vidur_best_qps']:.2f} QPS")
    print(f"  → BLIS achieves {metrics['qps_improvement']:+.1f}% higher QPS")

    print("\n⏱️  RUNTIME PERFORMANCE:")
    print(f"  BLIS Total:  {metrics['blis_total_runtime']:.2f}s "
          f"(avg: {metrics['blis_avg_runtime']:.2f}s per config)")
    print(f"  Vidur Total: {metrics['vidur_total_runtime']:.2f}s "
          f"(avg: {metrics['vidur_avg_runtime']:.2f}s per config)")
    print(f"  → BLIS is {metrics['speedup']:.1f}x faster")

    print("\n🏆 BEST CONFIGURATIONS:")

    # Find best configs
    blis_best = max(blis_data['successful_configs'], key=lambda x: x['max_qps'])
    config = blis_best['configuration']
    print(f"\n  BLIS Best:")
    print(f"    Max QPS: {blis_best['max_qps']:.2f}")
    print(f"    Runtime: {blis_best['runtime_seconds']:.2f}s")
    print(f"    Parameters: TP={config.get('tp', 1)}, "
          f"batch_size={config.get('batch_size')}, "
          f"max_scheduled_tokens={config.get('max_scheduled_tokens')}")

    if vidur_data['successful_configs']:
        vidur_best = max(vidur_data['successful_configs'], key=lambda x: x['max_qps'])
        config = vidur_best['configuration']
        print(f"\n  Vidur Best:")
        print(f"    Max QPS: {vidur_best['max_qps']:.2f}")
        print(f"    Runtime: {vidur_best['runtime_seconds']:.2f}s")
        print(f"    Parameters: TP={config.get('tp', 1)}, "
              f"batch_size={config.get('batch_size')}, "
              f"max_scheduled_tokens={config.get('max_scheduled_tokens')}")

    print("\n💡 KEY INSIGHTS:")
    if metrics['blis_success_rate'] > metrics['vidur_success_rate']:
        diff = metrics['blis_success_rate'] - metrics['vidur_success_rate']
        print(f"  • BLIS has {diff:.1f}% higher success rate")

    if metrics['qps_improvement'] > 0:
        print(f"  • BLIS finds configurations with {metrics['qps_improvement']:.1f}% better QPS")

    if metrics['speedup'] > 1:
        print(f"  • BLIS completes search {metrics['speedup']:.1f}x faster than Vidur")

    # Analyze failures if any
    if vidur_data['failed_configs']:
        print("\n⚠️  VIDUR FAILURE ANALYSIS:")
        failed = vidur_data['failed_configs']
        print(f"  Total failed: {len(failed)} configs")

        # Common failure patterns
        token_counts = {}
        for f in failed:
            tokens = f['configuration'].get('max_scheduled_tokens', 'unknown')
            token_counts[tokens] = token_counts.get(tokens, 0) + 1

        if token_counts:
            print(f"  Failures by max_scheduled_tokens:")
            for tokens, count in sorted(token_counts.items()):
                print(f"    {tokens}: {count} failures")

    print("\n" + "="*80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Compare BLIS and Vidur simulator results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_simulators.py blis_results.json vidur_results.json
  python compare_simulators.py blis_results.json vidur_results.json -o my_plots/
        """
    )

    parser.add_argument('blis_file', type=str,
                       help='Path to BLIS results JSON file')
    parser.add_argument('vidur_file', type=str,
                       help='Path to Vidur results JSON file')
    parser.add_argument('-o', '--output-dir', type=str, default='comparison_plots',
                       help='Output directory for plots (default: comparison_plots)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Loading results...")
    print(f"  BLIS:  {args.blis_file}")
    print(f"  Vidur: {args.vidur_file}")

    # Load data
    try:
        blis_data, vidur_data = load_results(args.blis_file, args.vidur_file)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing JSON: {e}")
        sys.exit(1)

    # Calculate metrics
    metrics = calculate_metrics(blis_data, vidur_data)

    # Extract config lists
    blis_configs = blis_data['successful_configs']
    vidur_configs = vidur_data['successful_configs']
    vidur_failed = vidur_data.get('failed_configs', [])

    print(f"\n📈 Generating visualizations...")

    # Generate all plots
    plot_runtime_comparison(blis_configs, vidur_configs, vidur_failed,
                           blis_data['summary']['total_search_runtime_seconds'],
                           vidur_data['summary']['total_search_runtime_seconds'],
                           output_dir)
    print(f"  ✓ Generated: {output_dir}/01_runtime_comparison.png")

    plot_qps_comparison(blis_configs, vidur_configs, vidur_failed, output_dir)
    print(f"  ✓ Generated: {output_dir}/02_qps_comparison.png")

    plot_success_rate(blis_data['summary'], vidur_data['summary'], output_dir)
    print(f"  ✓ Generated: {output_dir}/03_success_rate.png")

    plot_qps_distribution(blis_configs, vidur_configs, output_dir)
    print(f"  ✓ Generated: {output_dir}/04_qps_distribution.png")

    # Display summary metrics
    display_summary_metrics(blis_data, vidur_data, metrics)

    print(f"✅ Analysis complete! Plots saved to: {output_dir}/")


if __name__ == '__main__':
    main()
