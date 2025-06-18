import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

# -----------------------------------
# Plotting Utilities
# -----------------------------------

def plot_histogram(series: pd.Series, title: str, xlabel: str, outfile: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=30, color="skyblue", edgecolor="black")
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_discrete_bar(series: pd.Series, title: str, xlabel: str, outfile: Path, color="skyblue"):
    ELEVATIONS = np.linspace(-45, 230.625, 50)
    counts = series.value_counts().reindex(ELEVATIONS, fill_value=0)
    positions = np.arange(len(ELEVATIONS))
    show_every = 5
    tick_labels = [f"{val:.1f}" if i % show_every == 0 else "" for i, val in enumerate(ELEVATIONS)]

    plt.figure(figsize=(12, 5))
    plt.bar(positions, counts.values, width=0.8, color=color, edgecolor="black")
    plt.xticks(positions, tick_labels, rotation=45, ha='right', fontsize=9)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, outfile: Path):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=10, alpha=0.7, color="steelblue", edgecolor="k", linewidth=0.3)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_bar(counts: pd.Series, title: str, xlabel: str, ylabel: str, outfile: Path):
    plt.figure(figsize=(10, 4))
    ax = counts.sort_index().plot(kind="bar", color="cornflowerblue", edgecolor="black")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_combined_discrete_bar(series1, series2, xvals, title, xlabel, outfile: Path,
                                label1="s1", label2="s2", color1="mediumseagreen", color2="lightskyblue"):
    counts1 = pd.Series(series1).value_counts().reindex(xvals, fill_value=0)
    counts2 = pd.Series(series2).value_counts().reindex(xvals, fill_value=0)
    positions = np.arange(len(xvals))
    bar_width = 0.4

    plt.figure(figsize=(12, 5))
    plt.bar(positions - bar_width/2, counts1.values, width=bar_width, label=label1, color=color1, edgecolor="black")
    plt.bar(positions + bar_width/2, counts2.values, width=bar_width, label=label2, color=color2, edgecolor="black")

    show_every = max(1, len(xvals) // 20)
    tick_labels = [f"{x:.1f}" if i % show_every == 0 else "" for i, x in enumerate(xvals)]
    plt.xticks(positions, tick_labels, rotation=45, ha='right', fontsize=9)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

# -----------------------------------
# Main Analysis
# -----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse cocktail-party dataset")
    parser.add_argument("--datadir", type=Path, required=True, help="Directory containing CSVs (e.g. train/val/test)")
    parser.add_argument("--outdir", type=Path, default=None, help="Directory for output PNGs")
    args = parser.parse_args()

    # Setup output path
    outdir: Path = (args.outdir or Path.cwd()).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover CSVs
    csv_dir: Path = args.datadir.expanduser().resolve()
    csv_paths = list(csv_dir.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in directory: {csv_dir}")
        return

    # Load and merge
    dfs = [pd.read_csv(csv) for csv in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    split = "all"

    print(f"Loaded {len(df)} combined samples from {len(csv_paths)} CSVs")
    print(df.columns)

    for col in ["snr_measured_db", "sisnr_db"]:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found – skipping.")
            continue
        display_name = "SNR" if col == "snr_measured_db" else "SI-SNR"
        series = df[col]
        print(f"\n{col} statistics (dB):")
        print(series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
        plot_histogram(series, title=f"Distribution of {display_name}", xlabel="dB", outfile=outdir / f"{split}_{col}_hist.png")

    if {"snr_measured_db", "sisnr_db"} <= set(df.columns):
        plot_scatter(
            df["snr_measured_db"],
            df["sisnr_db"],
            title="SNR vs SI‑SNR",
            xlabel="SNR (dB)",
            ylabel="SI‑SNR (dB)",
            outfile=outdir / f"{split}_snr_vs_sisnr.png",
        )

    if "target_speaker" in df.columns:
        counts = df["target_speaker"].value_counts().sort_index()
        plot_bar(counts, title="Target‑Speaker Distribution", xlabel="Speaker ID", ylabel="Samples", outfile=outdir / f"{split}_target_speaker_counts.png")

    if "hrir_subject" in df.columns:
        counts = df["hrir_subject"].value_counts()
        plot_bar(counts, title="HRTF Subject Usage", xlabel="Subject ID", ylabel="Samples", outfile=outdir / f"{split}_hrtf_subject_usage.png")

    # Combined azimuth (s1 vs s2)
    if {"az1", "az2"} <= set(df.columns):
        AZIMUTHS = np.array(
            [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15,
             -10, -5, 0, 5, 10, 15, 20, 25, 30, 35,
             40, 45, 55, 65, 80], dtype=float
        )
        plot_combined_discrete_bar(
            df["az1"], df["az2"], AZIMUTHS,
            title="Azimuth Distribution (s1 vs s2)",
            xlabel="Azimuth (deg)",
            outfile=outdir / f"{split}_azimuth_s1_vs_s2.png",
            label1="s1", label2="s2"
        )

    # Combined elevation (s1 vs s2)
    if {"el1", "el2"} <= set(df.columns):
        ELEVATIONS = np.linspace(-45, 230.625, 50)
        plot_combined_discrete_bar(
            df["el1"], df["el2"], ELEVATIONS,
            title="Elevation Distribution (s1 vs s2)",
            xlabel="Elevation (deg)",
            outfile=outdir / f"{split}_elevation_s1_vs_s2.png",
            label1="s1", label2="s2"
        )

    print(f"\nAnalysis complete – plots saved to: {outdir}\n")

# -----------------------------------
# Run Script
# -----------------------------------

if __name__ == "__main__":
    import sys
    sys.argv = [
        "",  # Dummy script name
        "--datadir", "/home/erlend/Documents/div/cocktail-party/datasets/static",
        "--outdir", "./plots"
    ]
    main()
