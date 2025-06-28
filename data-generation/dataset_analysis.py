import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ast

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["font.family"] = "DejaVu Sans"

def plot_histogram(series, title, xlabel, outfile):
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=30, color="skyblue", edgecolor="black")
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_scatter(x, y, title, xlabel, ylabel, outfile):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=10, alpha=0.7, color="steelblue", edgecolor="k", linewidth=0.3)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_bar(counts, title, xlabel, ylabel, outfile):
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

def plot_combined_discrete_bar(series1, series2, xvals, title, xlabel, outfile,
                                label1="s1", label2="s2", color1="mediumseagreen", color2="lightskyblue"):
    counts1 = pd.Series(series1).value_counts().reindex(xvals, fill_value=0)
    counts2 = pd.Series(series2).value_counts().reindex(xvals, fill_value=0)
    positions = np.arange(len(xvals))
    bar_width = 0.4

    plt.figure(figsize=(12, 5))
    plt.bar(positions - bar_width/2, counts1.values, width=bar_width, label=label1, color=color1, edgecolor="black")
    plt.bar(positions + bar_width/2, counts2.values, width=bar_width, label=label2, color=color2, edgecolor="black")

    tick_labels = [f"{x:.0f}" for x in xvals]
    plt.xticks(positions, tick_labels, rotation=0, ha='center', fontsize=10)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def parse_float_array(s):
    try:
        arr = ast.literal_eval(s)
        return [float(x) for x in arr]
    except Exception:
        return [np.nan, np.nan, np.nan]

def analyze_dataframe(df, split, outdir):
    df["s1_azimuth"] = pd.to_numeric(df["s1_azimuth"], errors="coerce")
    df["s2_azimuth"] = pd.to_numeric(df["s2_azimuth"], errors="coerce")
    df["room_dim"] = df["room_dim"].apply(parse_float_array)

    if "snr_measured_db" in df.columns:
        plot_histogram(
            df["snr_measured_db"],
            title=f"SNR Distribution ({split})",
            xlabel="SNR (dB)",
            outfile=outdir / f"{split}_snr_hist.png"
        )

    if {"s1_azimuth", "s2_azimuth"} <= set(df.columns):
        az_min = int(np.floor(min(df["s1_azimuth"].min(), df["s2_azimuth"].min())))
        az_max = int(np.ceil(max(df["s1_azimuth"].max(), df["s2_azimuth"].max())))
        az_limit = max(abs(az_min), abs(az_max))
        az_limit = int(np.ceil(az_limit / 10.0)) * 10
        bin_edges = np.arange(-az_limit - 5, az_limit + 10, 10)
        bin_centers = bin_edges[:-1] + 5

        df["az1_binned"] = pd.cut(df["s1_azimuth"], bins=bin_edges, labels=bin_centers, include_lowest=True)
        df["az2_binned"] = pd.cut(df["s2_azimuth"], bins=bin_edges, labels=bin_centers, include_lowest=True)

        az1_vals = df["az1_binned"].dropna().astype(float)
        az2_vals = df["az2_binned"].dropna().astype(float)

        plot_combined_discrete_bar(
            az1_vals, az2_vals, bin_centers,
            title=f"Azimuth Distribution (s1 vs s2) — {split}",
            xlabel="Azimuth (deg)",
            outfile=outdir / f"{split}_azimuth_s1_vs_s2.png"
        )

        gap = (df["s1_azimuth"] - df["s2_azimuth"]).abs()
        plot_histogram(
            gap,
            title=f"Azimuth Gap (|s1 - s2|) — {split}",
            xlabel="|Az1 - Az2| (degrees)",
            outfile=outdir / f"{split}_azimuth_gap_hist.png"
        )

        if "snr_measured_db" in df.columns:
            plot_scatter(
                gap,
                df["snr_measured_db"],
                title=f"Azimuth Gap vs SNR — {split}",
                xlabel="Azimuth Gap (deg)",
                ylabel="SNR (dB)",
                outfile=outdir / f"{split}_azimuth_vs_snr.png"
            )

    if "s1_pos" in df.columns:
        df["s1_pos"] = df["s1_pos"].apply(parse_float_array)
        df["s2_pos"] = df["s2_pos"].apply(parse_float_array)

    if "s1_brir_r" in df.columns:
        usage = df["s1_brir_r"].value_counts().head(20)
        plot_bar(usage, title=f"Top 20 s1 BRIR Files — {split}", xlabel="BRIR filename", ylabel="Count",
                  outfile=outdir / f"{split}_s1_brir_usage.png")

    if "room_dim" in df.columns and "snr_measured_db" in df.columns:
        df["room_id"] = df["room_dim"].apply(lambda x: f"{x[0]:.3f}×{x[1]:.3f}×{x[2]:.3f}" if isinstance(x, (list, tuple)) and len(x) == 3 else "unknown")
        room_snr = df.groupby("room_id")["snr_measured_db"].mean().sort_values()
        plot_bar(
            room_snr,
            title=f"Average SNR per Room — {split}",
            xlabel="Room Dimensions (L×W×H)",
            ylabel="Avg SNR (dB)",
            outfile=outdir / f"{split}_room_snr.png"
        )

def main():
    parser = argparse.ArgumentParser(description="Analyse BRIR-based cocktail-party dataset")
    parser.add_argument("--datadir", type=Path, default=("datasets/static"), help="Directory containing train/val/test CSVs")
    parser.add_argument("--outdir", type=Path, default=Path("data-generation/plots"), help="Directory for output PNGs")
    args = parser.parse_args()

    outdir: Path = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    csv_dir: Path = args.datadir.expanduser().resolve()
    csv_paths = list(csv_dir.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in directory: {csv_dir}")
        return

    all_dfs = []
    for csv in csv_paths:
        split = csv.stem
        df = pd.read_csv(csv)
        print(f"Loaded {len(df)} samples from {csv.name}")
        all_dfs.append(df)
        analyze_dataframe(df.copy(), split, outdir)

    # Analyze combined dataset
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📊 Performing combined analysis on all splits...")
    analyze_dataframe(full_df, "all_splits", outdir)

    print(f"\n✅ Analysis complete. Plots saved to: {outdir}\n")

if __name__ == "__main__":
    main()