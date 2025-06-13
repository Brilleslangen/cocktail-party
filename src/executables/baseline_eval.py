import os
import csv
import torch
import torchaudio
import hydra
import wandb
import numpy as np
from tqdm import tqdm
from scipy import stats
from tabulate import tabulate
from omegaconf import DictConfig

from src.data.dataset import AudioDataset
from src.evaluate.metrics import (
    compute_all_metrics_baseline,
    compute_binaural_metrics,
    compute_ild_error,
    compute_itd_error,
    compute_ipd_error
)
from src.evaluate.loss import compute_mask
from src.helpers import select_device


def compute_statistical_significance(values, reference_value=0.0, alpha=0.05):
    """
    Compute statistical significance using one-sample t-test.
    Tests if the mean of values is significantly different from reference_value.

    Args:
        values: List of metric values
        reference_value: Value to test against (default 0.0)
        alpha: Significance level (default 0.05)

    Returns:
        p-value and whether result is significant
    """
    if len(values) < 2:
        return 1.0, False

    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(values, reference_value)
    is_significant = p_value < alpha

    return p_value, is_significant


def evaluate_baseline_dataset(dataset_dir: str, sample_rate: int = 16000):
    """
    Evaluate baseline metrics on test dataset with proper binaural handling.

    Args:
        dataset_dir: Path to test dataset directory
        sample_rate: Audio sample rate

    Returns:
        results_list: List of dictionaries with per-file results
        summary_stats: Dictionary with summary statistics
    """
    device = select_device()
    test_ds = AudioDataset(dataset_dir, sample_rate)

    # Metric names
    basic_metric_names = ['SNR', 'SDR', 'SI-SNR', 'SI-SDR', 'ESTOI', 'PESQ',
                          'SNRi', 'SDRi', 'SI-SNRi', 'SI-SDRi']
    binaural_metric_names = ['ILD_MSE', 'ILD_corr', 'ITD_MSE', 'IPD_MSE', 'IPD_corr']

    results_list = []

    # Store metrics for summary stats
    all_metrics_left = {metric: [] for metric in basic_metric_names}
    all_metrics_right = {metric: [] for metric in basic_metric_names}
    all_metrics_weighted = {metric: [] for metric in basic_metric_names}
    all_binaural_metrics = {metric: [] for metric in binaural_metric_names}

    print(f"Evaluating {len(test_ds)} test samples...")

    for idx in tqdm(range(len(test_ds))):
        # Load audio
        mix, ref = test_ds[idx]  # [2, T], [2, T]
        mix = mix.to(device)
        ref = ref.to(device)

        # Get filename and duration
        filename = os.path.basename(test_ds.mix_files[idx])
        duration = mix.shape[-1] / sample_rate

        # Initialize result for this file
        result = {
            'filename': filename,
            'duration': duration
        }

        # Process each channel separately
        channel_metrics = []
        channel_energies = []

        for ch in range(2):  # Left and right channels
            mix_ch = mix[ch:ch + 1, :]  # [1, T]
            ref_ch = ref[ch:ch + 1, :]  # [1, T]

            # Create mask for valid samples
            lengths = torch.tensor([mix_ch.shape[-1]], device=device)
            mask = compute_mask(lengths, mix_ch.shape[-1], device)  # [1, T]

            # Compute all metrics for this channel
            metrics = compute_all_metrics_baseline(mix_ch, ref_ch, mask, sample_rate)
            channel_metrics.append(metrics)

            # Compute channel energy for weighting
            energy = (ref_ch ** 2).sum().item()
            channel_energies.append(energy)

        # Compute energy weights
        total_energy = sum(channel_energies)
        if total_energy > 0:
            weights = [e / total_energy for e in channel_energies]
        else:
            weights = [0.5, 0.5]  # Equal weights if no energy

        # Store per-channel and weighted metrics
        for metric in basic_metric_names:
            left_val = channel_metrics[0][metric].item()
            right_val = channel_metrics[1][metric].item()

            # Store individual channel values
            result[f'{metric}_L'] = left_val
            result[f'{metric}_R'] = right_val

            # Energy-weighted average
            weighted_val = weights[0] * left_val + weights[1] * right_val
            result[f'{metric}'] = weighted_val  # Default is weighted

            # Channel difference (for checking consistency)
            result[f'{metric}_diff'] = abs(left_val - right_val)

            # Collect for summary stats
            all_metrics_left[metric].append(left_val)
            all_metrics_right[metric].append(right_val)
            all_metrics_weighted[metric].append(weighted_val)

        # Compute binaural metrics
        binaural_metrics = compute_binaural_metrics(mix, ref, sample_rate)
        for metric, value in binaural_metrics.items():
            result[metric] = value
            all_binaural_metrics[metric].append(value)

        # Store energy weights for reference
        result['weight_L'] = weights[0]
        result['weight_R'] = weights[1]

        results_list.append(result)

    # Compute summary statistics
    summary_stats = {
        'left': {},
        'right': {},
        'weighted': {},
        'binaural': {}
    }

    # Basic metrics stats for each channel and weighted
    for metric in basic_metric_names:
        # Left channel
        values_l = all_metrics_left[metric]
        summary_stats['left'][metric] = compute_metric_stats(values_l)

        # Right channel
        values_r = all_metrics_right[metric]
        summary_stats['right'][metric] = compute_metric_stats(values_r)

        # Weighted average
        values_w = all_metrics_weighted[metric]
        summary_stats['weighted'][metric] = compute_metric_stats(values_w)

    # Binaural metrics stats
    for metric in binaural_metric_names:
        values = all_binaural_metrics[metric]
        summary_stats['binaural'][metric] = compute_metric_stats(values)

    return results_list, summary_stats


def compute_metric_stats(values):
    """Compute statistics for a metric."""
    stats_dict = {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values)
    }

    # Statistical significance test
    # For baseline, test if metrics are significantly different from 0
    p_value, is_significant = compute_statistical_significance(values, reference_value=0.0)

    stats_dict['p_value'] = p_value
    stats_dict['significant'] = is_significant

    return stats_dict


def save_detailed_results(results_list, output_path):
    """Save detailed per-file results to CSV."""
    if not results_list:
        return

    # Define column order
    basic_metrics = ['SNR', 'SDR', 'SI-SNR', 'SI-SDR', 'ESTOI', 'PESQ',
                     'SNRi', 'SDRi', 'SI-SNRi', 'SI-SDRi']
    binaural_metrics = ['ILD_MSE', 'ILD_corr', 'ITD_MSE', 'IPD_MSE', 'IPD_corr']

    headers = ['filename', 'duration', 'weight_L', 'weight_R']

    # Add weighted metrics (default)
    headers.extend(basic_metrics)

    # Add per-channel metrics
    for metric in basic_metrics:
        headers.extend([f'{metric}_L', f'{metric}_R', f'{metric}_diff'])

    # Add binaural metrics
    headers.extend(binaural_metrics)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()

        # Write each row with proper formatting
        for result in results_list:
            formatted_result = {
                'filename': result['filename'],
                'duration': f"{result['duration']:.2f}",
                'weight_L': f"{result['weight_L']:.3f}",
                'weight_R': f"{result['weight_R']:.3f}"
            }

            # Format metric values
            for key in result:
                if key not in ['filename', 'duration', 'weight_L', 'weight_R']:
                    formatted_result[key] = f"{result[key]:.3f}"

            writer.writerow(formatted_result)


def save_summary_stats(summary_stats, output_path):
    """Save summary statistics to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create comprehensive summary with all channels
    rows = []

    # Basic metrics
    basic_metrics = ['SNR', 'SDR', 'SI-SNR', 'SI-SDR', 'ESTOI', 'PESQ',
                     'SNRi', 'SDRi', 'SI-SNRi', 'SI-SDRi']

    # Headers
    headers = ['metric', 'channel'] + ['mean', 'std', 'min', 'max', 'median', 'p_value', 'significant']

    # Add rows for each metric and channel
    for metric in basic_metrics:
        # Left channel
        row = {'metric': metric, 'channel': 'left'}
        row.update(format_stats_row(summary_stats['left'][metric]))
        rows.append(row)

        # Right channel
        row = {'metric': metric, 'channel': 'right'}
        row.update(format_stats_row(summary_stats['right'][metric]))
        rows.append(row)

        # Weighted average
        row = {'metric': metric, 'channel': 'weighted'}
        row.update(format_stats_row(summary_stats['weighted'][metric]))
        rows.append(row)

    # Binaural metrics
    for metric, stats in summary_stats['binaural'].items():
        row = {'metric': metric, 'channel': 'binaural'}
        row.update(format_stats_row(stats))
        rows.append(row)

    # Write to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def format_stats_row(stats):
    """Format statistics for CSV row."""
    return {
        'mean': f"{stats['mean']:.3f}",
        'std': f"{stats['std']:.3f}",
        'min': f"{stats['min']:.3f}",
        'max': f"{stats['max']:.3f}",
        'median': f"{stats['median']:.3f}",
        'p_value': f"{stats['p_value']:.4f}",
        'significant': '*' if stats['significant'] else ''
    }


def print_summary_table(summary_stats):
    """Print formatted summary statistics using tabulate."""
    print("\n" + "=" * 100)
    print("BASELINE EVALUATION SUMMARY")
    print("=" * 100)

    # Basic metrics - weighted average
    print("\n📊 WEIGHTED AVERAGE METRICS (Energy-weighted L/R)")
    print("-" * 100)

    table_data = []
    headers = ['Metric', 'Mean ± Std', 'Min', 'Max', 'Median', 'p-value', 'Sig.']

    basic_metrics = ['SNR', 'SDR', 'SI-SNR', 'SI-SDR', 'ESTOI', 'PESQ',
                     'SNRi', 'SDRi', 'SI-SNRi', 'SI-SDRi']

    for metric in basic_metrics:
        stats = summary_stats['weighted'][metric]
        mean_std = f"{stats['mean']:.3f} ± {stats['std']:.3f}"
        sig = '***' if stats['p_value'] < 0.001 else \
            '**' if stats['p_value'] < 0.01 else \
                '*' if stats['p_value'] < 0.05 else ''

        row = [metric, mean_std, f"{stats['min']:.3f}", f"{stats['max']:.3f}",
               f"{stats['median']:.3f}", f"{stats['p_value']:.4f}", sig]
        table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Per-channel comparison
    print("\n🎧 PER-CHANNEL COMPARISON")
    print("-" * 100)

    channel_data = []
    channel_headers = ['Metric', 'Left (Mean±Std)', 'Right (Mean±Std)', 'L-R Diff']

    for metric in basic_metrics:
        left_stats = summary_stats['left'][metric]
        right_stats = summary_stats['right'][metric]

        left_str = f"{left_stats['mean']:.3f} ± {left_stats['std']:.3f}"
        right_str = f"{right_stats['mean']:.3f} ± {right_stats['std']:.3f}"
        diff = abs(left_stats['mean'] - right_stats['mean'])

        row = [metric, left_str, right_str, f"{diff:.3f}"]
        channel_data.append(row)

    print(tabulate(channel_data, headers=channel_headers, tablefmt='grid'))

    # Binaural metrics
    print("\n🔊 BINAURAL PRESERVATION METRICS")
    print("-" * 100)

    binaural_data = []
    binaural_headers = ['Metric', 'Mean ± Std', 'Min', 'Max', 'Interpretation']

    for metric, stats in summary_stats['binaural'].items():
        mean_std = f"{stats['mean']:.3f} ± {stats['std']:.3f}"

        # Add interpretation
        if 'MSE' in metric:
            interp = "Lower is better"
        elif 'corr' in metric:
            interp = "Higher is better (max=1)"
        else:
            interp = ""

        row = [metric, mean_std, f"{stats['min']:.3f}", f"{stats['max']:.3f}", interp]
        binaural_data.append(row)

    print(tabulate(binaural_data, headers=binaural_headers, tablefmt='grid'))

    print("\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001")
    print("=" * 100)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Main function to run baseline evaluation."""

    # Setup paths
    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            job_type="baseline-evaluation",
            name="baseline-binaural-evaluation",
            config=dict(cfg)
        )
        dataset_dir = run.use_artifact(cfg.dataset.artifact_name, type="dataset").download()
    else:
        # Assume dataset is already downloaded
        dataset_dir = "../datasets/static"

    test_dir = os.path.join(dataset_dir, "test")

    # Output paths
    detailed_csv = "outputs/baseline_evaluation_detailed.csv"
    summary_csv = "outputs/baseline_evaluation_summary.csv"

    # Run evaluation
    print("Starting enhanced binaural baseline evaluation...")
    results_list, summary_stats = evaluate_baseline_dataset(test_dir, cfg.dataset.sample_rate)

    # Save results
    print(f"\nSaving detailed results to: {detailed_csv}")
    save_detailed_results(results_list, detailed_csv)

    print(f"Saving summary statistics to: {summary_csv}")
    save_summary_stats(summary_stats, summary_csv)

    # Print summary table
    print_summary_table(summary_stats)

    # Log to W&B if enabled
    if cfg.wandb.enabled:
        # Log CSVs as artifacts
        artifact = wandb.Artifact("baseline-binaural-evaluation", type="evaluation")
        artifact.add_file(detailed_csv)
        artifact.add_file(summary_csv)
        run.log_artifact(artifact)

        # Log summary metrics
        for metric, stats in summary_stats['weighted'].items():
            wandb.run.summary[f"baseline/{metric}_weighted_mean"] = stats['mean']
            wandb.run.summary[f"baseline/{metric}_weighted_std"] = stats['std']

        for metric, stats in summary_stats['binaural'].items():
            wandb.run.summary[f"baseline/{metric}_mean"] = stats['mean']

        wandb.finish()

    print("\nEnhanced binaural baseline evaluation complete!")


if __name__ == "__main__":
    main()