#!/usr/bin/env python3
"""
Compare validation reports from BLIS and Vidur simulators.

Validates simulator predictions by comparing against real vLLM results.
Generates visualizations showing:
1. Adapted plots: runtime, QPS, SLO compliance, QPS distribution
2. New plots: SLO compliance comparison, SLO value accuracy
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_validation_reports(blis_file: str, vidur_file: str) -> Tuple[Dict, Dict]:
    """Load BLIS and Vidur validation report JSON files."""
    with open(blis_file, 'r') as f:
        blis_data = json.load(f)
    with open(vidur_file, 'r') as f:
        vidur_data = json.load(f)
    return blis_data, vidur_data


def calculate_metrics(blis_data: Dict, vidur_data: Dict) -> Dict[str, Any]:
    """Calculate comparison metrics from validation reports."""
    blis_summary = blis_data['summary']
    vidur_summary = vidur_data['summary']

    blis_configs = blis_data['results']
    vidur_configs = vidur_data['results']

    # SLO compliance metrics
    blis_meeting_slos = sum(1 for c in blis_configs if c.get('meets_slos') is True)
    blis_total = len(blis_configs)
    vidur_meeting_slos = sum(1 for c in vidur_configs if c.get('meets_slos') is True)
    vidur_total = len(vidur_configs)

    # QPS metrics
    blis_qps_values = [c['max_qps'] for c in blis_configs if c.get('max_qps')]
    vidur_qps_values = [c['max_qps'] for c in vidur_configs if c.get('max_qps')]

    blis_best_qps = max(blis_qps_values) if blis_qps_values else 0
    vidur_best_qps = max(vidur_qps_values) if vidur_qps_values else 0

    # Prediction accuracy: measure error in SLO metrics where available
    blis_errors = []
    for config in blis_configs:
        for metric_name, metric_data in config.get('slo_metrics', {}).items():
            if metric_data.get('error_percent') is not None:
                blis_errors.append(abs(metric_data['error_percent']))

    vidur_errors = []
    for config in vidur_configs:
        for metric_name, metric_data in config.get('slo_metrics', {}).items():
            if metric_data.get('error_percent') is not None:
                vidur_errors.append(abs(metric_data['error_percent']))

    metrics = {
        'blis_slo_compliance': (blis_meeting_slos / blis_total * 100) if blis_total > 0 else 0,
        'vidur_slo_compliance': (vidur_meeting_slos / vidur_total * 100) if vidur_total > 0 else 0,
        'blis_best_qps': blis_best_qps,
        'vidur_best_qps': vidur_best_qps,
        'blis_total_guidellm_runtime': blis_summary.get('total_guidellm_runtime_seconds', 0),
        'vidur_total_guidellm_runtime': vidur_summary.get('total_guidellm_runtime_seconds', 0),
        'blis_mean_error_percent': np.mean(blis_errors) if blis_errors else 0,
        'vidur_mean_error_percent': np.mean(vidur_errors) if vidur_errors else 0,
        'blis_max_error_percent': max(blis_errors) if blis_errors else 0,
        'vidur_max_error_percent': max(vidur_errors) if vidur_errors else 0,
    }

    return metrics


def plot_runtime_comparison(blis_data: Dict, vidur_data: Dict, output_dir: Path):
    """Plot 1: GuideLLM vs simulator config exploration runtime comparison."""
    fig, ax = plt.subplots(figsize=(11, 6))

    blis_summary = blis_data['summary']
    vidur_summary = vidur_data['summary']

    # Extract runtimes
    # GuideLLM runtime: actual vLLM validation benchmark time
    blis_guidellm_runtime = blis_summary.get('total_guidellm_runtime_seconds', 0)
    vidur_guidellm_runtime = vidur_summary.get('total_guidellm_runtime_seconds', 0)

    # Simulator runtime: config exploration/search time
    blis_simulator_runtime = blis_summary.get('total_simulator_runtime_seconds', 0)
    vidur_simulator_runtime = vidur_summary.get('total_simulator_runtime_seconds', 0)

    categories = ['BLIS', 'Vidur']
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, [blis_guidellm_runtime, vidur_guidellm_runtime], width,
                   label='GuideLLM (Real Validation)', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + width/2, [blis_simulator_runtime, vidur_simulator_runtime], width,
                   label='Simulator (Config Search)', color='#e67e22', alpha=0.85)

    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('GuideLLM vs Simulator Config Exploration Runtime Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '01_runtime_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_qps_comparison(blis_data: Dict, vidur_data: Dict, output_dir: Path):
    """Plot 2: Max QPS comparison - top configs sorted by performance."""
    blis_configs = blis_data['results']
    vidur_configs = vidur_data['results']

    # Match configs by rank
    blis_by_rank = {c['rank']: c for c in blis_configs}
    vidur_by_rank = {c['rank']: c for c in vidur_configs}

    # Build comparison data
    config_data = []
    max_rank = min(len(blis_configs), len(vidur_configs), 5)

    for rank in range(1, max_rank + 1):
        blis_config = blis_by_rank.get(rank)
        vidur_config = vidur_by_rank.get(rank)

        if blis_config:
            config = blis_config['configuration']
            label = f"Rank {rank}: BS={config.get('batch_size')}, MST={config.get('max_scheduled_tokens')}, MML={config.get('max_model_len')}"

            config_data.append({
                'label': label,
                'rank': rank,
                'blis_qps': blis_config.get('max_qps', 0),
                'vidur_qps': vidur_config.get('max_qps', 0) if vidur_config else 0,
            })

    if not config_data:
        print("Warning: No configs found for QPS comparison")
        return

    # Reverse for plotting (highest rank at top)
    config_data.reverse()

    labels = [c['label'] for c in config_data]
    blis_qps = [c['blis_qps'] for c in config_data]
    vidur_qps = [c['vidur_qps'] for c in config_data]

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(labels))
    height = 0.35

    bars1 = ax.barh(y_pos + height/2, blis_qps, height,
                    label='BLIS', color='#27ae60', alpha=0.85)
    bars2 = ax.barh(y_pos - height/2, vidur_qps, height,
                    label='Vidur', color='#e74c3c', alpha=0.85)

    # Add value labels on bars
    for bar, qps in zip(bars1, blis_qps):
        if qps > 0:
            ax.text(qps + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{qps:.2f}', va='center', fontsize=10, fontweight='bold')

    for bar, qps in zip(bars2, vidur_qps):
        if qps > 0:
            ax.text(qps + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{qps:.2f}', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Max QPS', fontsize=12, fontweight='bold')
    ax.set_title('Top Configurations: Simulator Max QPS Comparison',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    # Add abbreviation legend
    abbrev_text = 'BS = Batch Size\nMST = Max Scheduled Tokens\nMML = Max Model Length'
    ax.text(0.98, 0.02, abbrev_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / '02_qps_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_success_rate(blis_data: Dict, vidur_data: Dict, output_dir: Path):
    """Plot 3: SLO compliance rate comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    blis_summary = blis_data['summary']
    vidur_summary = vidur_data['summary']

    # Extract SLO pass/fail metrics
    blis_meeting = blis_summary['configs_meeting_slos']
    blis_failing = blis_summary['configs_failing_slos']
    vidur_meeting = vidur_summary['configs_meeting_slos']
    vidur_failing = vidur_summary['configs_failing_slos']

    # BLIS pie chart
    ax1.pie([blis_meeting, blis_failing], labels=['Meeting SLOs', 'Failing SLOs'],
            autopct='%1.1f%%', startangle=90, colors=['#27ae60', '#e74c3c'],
            textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax1.set_title(f'BLIS SLO Compliance\n({blis_meeting}/{blis_summary["total_configs_tested"]} configs)',
                  fontsize=12, fontweight='bold')

    # Vidur pie chart
    ax2.pie([vidur_meeting, vidur_failing], labels=['Meeting SLOs', 'Failing SLOs'],
            autopct='%1.1f%%', startangle=90, colors=['#27ae60', '#e74c3c'],
            textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title(f'Vidur SLO Compliance\n({vidur_meeting}/{vidur_summary["total_configs_tested"]} configs)',
                  fontsize=12, fontweight='bold')

    plt.suptitle('SLO Compliance Rates (Real vLLM)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / '03_success_rate.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_qps_distribution(blis_data: Dict, vidur_data: Dict, output_dir: Path):
    """Plot 4: QPS variation across configuration parameters."""
    blis_configs = blis_data['results']
    vidur_configs = vidur_data['results']

    # Extract all parameters
    def extract_all_params(configs):
        data = {
            'batch_size': [],
            'max_scheduled_tokens': [],
            'max_model_len': [],
            'qps': []
        }
        for c in configs:
            config = c['configuration']
            data['batch_size'].append(config.get('batch_size', 0))
            data['max_scheduled_tokens'].append(config.get('max_scheduled_tokens', 0))
            data['max_model_len'].append(config.get('max_model_len', 0))
            data['qps'].append(c['max_qps'])
        return data

    blis_param_data = extract_all_params(blis_configs)
    vidur_param_data = extract_all_params(vidur_configs)

    # Define parameters to plot
    params = [
        ('batch_size', 'Batch Size', lambda x: f'{x}'),
        ('max_scheduled_tokens', 'Max Scheduled Tokens', lambda x: f'{x}'),
        ('max_model_len', 'Max Model Length', lambda x: f'{x}'),
    ]

    # Grid size for 3 plots
    num_params = len(params)
    nrows, ncols = 1, 3
    figsize = (18, 5)

    # Create subplot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    axes = axes.flatten()

    width = 0.35

    for idx, (param_key, param_label, format_func) in enumerate(params):
        ax = axes[idx]

        # Get unique values for this parameter from both simulators
        blis_values = set(blis_param_data[param_key])
        vidur_values = set(vidur_param_data[param_key])
        param_values = sorted(list(blis_values | vidur_values))

        # Group QPS by parameter value
        blis_by_param = {val: [] for val in param_values}
        vidur_by_param = {val: [] for val in param_values}

        for val, qps in zip(blis_param_data[param_key], blis_param_data['qps']):
            if val in blis_by_param:
                blis_by_param[val].append(qps)

        for val, qps in zip(vidur_param_data[param_key], vidur_param_data['qps']):
            if val in vidur_by_param:
                vidur_by_param[val].append(qps)

        # Calculate means
        blis_means = [np.mean(blis_by_param[val]) if blis_by_param[val] else 0 for val in param_values]
        vidur_means = [np.mean(vidur_by_param[val]) if vidur_by_param[val] else 0 for val in param_values]

        # Plot bars
        positions = np.arange(len(param_values))
        ax.bar(positions - width/2, blis_means, width, label='BLIS',
               color='#27ae60', alpha=0.85)
        ax.bar(positions + width/2, vidur_means, width, label='Vidur',
               color='#e74c3c', alpha=0.85)

        # Formatting
        ax.set_xlabel(param_label, fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg Max QPS', fontsize=10, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels([format_func(val) for val in param_values], rotation=45, ha='right')
        ax.legend(fontsize=9)
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

    plt.suptitle('Max QPS Variation Across Configuration Parameters',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_dir / '04_qps_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_slo_values_comparison(blis_data: Dict, vidur_data: Dict, output_dir: Path):
    """Plot 6: SLO value comparison at top configs - Real vs Predicted."""
    blis_configs = blis_data['results']
    vidur_configs = vidur_data['results']

    # Match configs by rank and get top 3
    blis_by_rank = {c['rank']: c for c in blis_configs}
    vidur_by_rank = {c['rank']: c for c in vidur_configs}

    top_ranks = sorted(list(set(list(blis_by_rank.keys()) + list(vidur_by_rank.keys()))))[:3]

    # Collect all SLO metrics
    all_metrics = {}
    for rank in top_ranks:
        blis_config = blis_by_rank.get(rank)
        if blis_config:
            for metric_name, metric_data in blis_config.get('slo_metrics', {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []

    if not all_metrics:
        print("Warning: No SLO metrics found for comparison")
        return

    # Create subplots for each SLO metric
    num_metrics = len(all_metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6))

    if num_metrics == 1:
        axes = [axes]

    for ax_idx, (metric_name, _) in enumerate(all_metrics.items()):
        ax = axes[ax_idx]

        # Collect data for this metric
        x_pos = 0
        bar_width = 0.25
        x_labels = []
        colors_list = ['#3498db', '#27ae60', '#e74c3c']  # Real, BLIS, Vidur

        for rank in top_ranks:
            blis_config = blis_by_rank.get(rank)
            vidur_config = vidur_by_rank.get(rank)

            if blis_config and metric_name in blis_config.get('slo_metrics', {}):
                metric_data = blis_config['slo_metrics'][metric_name]
                real_ms = metric_data.get('real_ms', 0)
                simulator_ms = metric_data.get('simulator_ms', 0)
                threshold_ms = metric_data.get('threshold_ms', 0)

                # Plot bars: Real, BLIS, Vidur
                bars = []
                values = [real_ms, simulator_ms]
                labels = ['Real vLLM', 'BLIS Pred.']

                if vidur_config and metric_name in vidur_config.get('slo_metrics', {}):
                    vidur_metric = vidur_config['slo_metrics'][metric_name]
                    values.append(vidur_metric.get('simulator_ms', 0))
                    labels.append('Vidur Pred.')

                positions = [x_pos + i * bar_width for i in range(len(values))]
                for pos, val, color, label in zip(positions, values, colors_list[:len(values)], labels):
                    ax.bar(pos, val, bar_width, color=color, alpha=0.85, label=label if x_pos == 0 else "")

                # Add threshold line
                ax.axhline(y=threshold_ms, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                          label='Threshold' if x_pos == 0 else "")

                x_pos += len(values) * bar_width + 0.2
                x_labels.append(f'Rank {rank}')

        # Formatting
        ax.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        if ax_idx == 0:
            ax.legend(fontsize=9, loc='upper left')

    plt.suptitle('SLO Metric Values: Real vLLM vs Simulator Predictions',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '05_slo_values_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def display_summary_metrics(blis_data: Dict, vidur_data: Dict, metrics: Dict):
    """Display comprehensive summary metrics from validation reports."""
    blis_summary = blis_data['summary']
    vidur_summary = vidur_data['summary']

    print("\n" + "="*80)
    print("VALIDATION COMPARISON SUMMARY")
    print("="*80)

    print("\n📊 SLO COMPLIANCE RATES (Real vLLM):")
    print(f"  BLIS:  {metrics['blis_slo_compliance']:.1f}% "
          f"({blis_summary['configs_meeting_slos']}/{blis_summary['total_configs_tested']} configs)")
    print(f"  Vidur: {metrics['vidur_slo_compliance']:.1f}% "
          f"({vidur_summary['configs_meeting_slos']}/{vidur_summary['total_configs_tested']} configs)")

    print("\n🚀 MAX QPS PREDICTED:")
    print(f"  BLIS:  {metrics['blis_best_qps']:.2f} QPS")
    print(f"  Vidur: {metrics['vidur_best_qps']:.2f} QPS")
    qps_diff = ((metrics['blis_best_qps'] - metrics['vidur_best_qps']) / metrics['vidur_best_qps'] * 100) if metrics['vidur_best_qps'] > 0 else 0
    print(f"  → Difference: {qps_diff:+.1f}%")

    print("\n⏱️  VALIDATION BENCHMARK DURATION:")
    print(f"  BLIS Total:  {metrics['blis_total_guidellm_runtime']:.2f}s")
    print(f"  Vidur Total: {metrics['vidur_total_guidellm_runtime']:.2f}s")
    if metrics['blis_total_guidellm_runtime'] > 0:
        speedup = metrics['vidur_total_guidellm_runtime'] / metrics['blis_total_guidellm_runtime']
        print(f"  → {'BLIS' if speedup > 1 else 'Vidur'} benchmark took {speedup:.1f}x")

    print("\n🎯 PREDICTION ACCURACY:")
    print(f"  BLIS Mean Error:   {metrics['blis_mean_error_percent']:.1f}% (max: {metrics['blis_max_error_percent']:.1f}%)")
    print(f"  Vidur Mean Error:  {metrics['vidur_mean_error_percent']:.1f}% (max: {metrics['vidur_max_error_percent']:.1f}%)")

    if metrics['blis_mean_error_percent'] < metrics['vidur_mean_error_percent']:
        print(f"  → BLIS predictions are more accurate by {metrics['vidur_mean_error_percent'] - metrics['blis_mean_error_percent']:.1f}%")
    else:
        print(f"  → Vidur predictions are more accurate by {metrics['blis_mean_error_percent'] - metrics['vidur_mean_error_percent']:.1f}%")

    print("\n🏆 BEST CONFIGURATIONS (Highest SLO Compliance):")

    # Find best configs by SLO compliance
    blis_configs = blis_data['results']
    vidur_configs = vidur_data['results']

    blis_best = max(blis_configs, key=lambda x: x['max_qps']) if blis_configs else None
    vidur_best = max(vidur_configs, key=lambda x: x['max_qps']) if vidur_configs else None

    if blis_best:
        config = blis_best['configuration']
        print(f"\n  BLIS (Rank {blis_best['rank']}):")
        print(f"    Max QPS: {blis_best['max_qps']:.2f}")
        print(f"    SLOs Met: {blis_best['meets_slos']}")
        print(f"    Parameters: TP={config.get('tp')}, BS={config.get('batch_size')}, "
              f"MST={config.get('max_scheduled_tokens')}, MML={config.get('max_model_len')}")

    if vidur_best:
        config = vidur_best['configuration']
        print(f"\n  Vidur (Rank {vidur_best['rank']}):")
        print(f"    Max QPS: {vidur_best['max_qps']:.2f}")
        print(f"    SLOs Met: {vidur_best['meets_slos']}")
        print(f"    Parameters: TP={config.get('tp')}, BS={config.get('batch_size')}, "
              f"MST={config.get('max_scheduled_tokens')}, MML={config.get('max_model_len')}")

    print("\n💡 KEY INSIGHTS:")
    if metrics['blis_slo_compliance'] >= metrics['vidur_slo_compliance']:
        print(f"  • BLIS and Vidur have similar SLO compliance rates")
    else:
        diff = metrics['vidur_slo_compliance'] - metrics['blis_slo_compliance']
        print(f"  • Vidur's predictions have {diff:.1f}% higher SLO compliance")

    if abs(metrics['blis_mean_error_percent'] - metrics['vidur_mean_error_percent']) > 5:
        if metrics['blis_mean_error_percent'] < metrics['vidur_mean_error_percent']:
            print(f"  • BLIS has significantly more accurate predictions")
        else:
            print(f"  • Vidur has significantly more accurate predictions")

    print("\n" + "="*80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Compare validation reports from BLIS and Vidur simulators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_simulators.py blis_validation_report.json vidur_validation_report.json
  python compare_simulators.py blis_validation_report.json vidur_validation_report.json -o my_plots/
        """
    )

    parser.add_argument('blis_file', type=str,
                       help='Path to BLIS validation report JSON file')
    parser.add_argument('vidur_file', type=str,
                       help='Path to Vidur validation report JSON file')
    parser.add_argument('-o', '--output-dir', type=str, default='comparison_plots',
                       help='Output directory for plots (default: comparison_plots)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Loading validation reports...")
    print(f"  BLIS:  {args.blis_file}")
    print(f"  Vidur: {args.vidur_file}")

    # Load data
    try:
        blis_data, vidur_data = load_validation_reports(args.blis_file, args.vidur_file)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing JSON: {e}")
        sys.exit(1)

    # Calculate metrics
    metrics = calculate_metrics(blis_data, vidur_data)

    print(f"\n📈 Generating visualizations...")

    # Generate all plots
    plot_runtime_comparison(blis_data, vidur_data, output_dir)
    print(f"  ✓ Generated: {output_dir}/01_runtime_comparison.png")

    plot_qps_comparison(blis_data, vidur_data, output_dir)
    print(f"  ✓ Generated: {output_dir}/02_qps_comparison.png")

    plot_success_rate(blis_data, vidur_data, output_dir)
    print(f"  ✓ Generated: {output_dir}/03_success_rate.png")

    plot_qps_distribution(blis_data, vidur_data, output_dir)
    print(f"  ✓ Generated: {output_dir}/04_qps_distribution.png")

    plot_slo_values_comparison(blis_data, vidur_data, output_dir)
    print(f"  ✓ Generated: {output_dir}/05_slo_values_comparison.png")

    # Display summary metrics
    display_summary_metrics(blis_data, vidur_data, metrics)

    print(f"✅ Analysis complete! Plots saved to: {output_dir}/")


if __name__ == '__main__':
    main()
