import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Plot helpers (matplotlib‑only)
# ---------------------------

def plot_histogram(series: pd.Series, title: str, xlabel: str, outfile: Path):
    plt.figure()
    plt.hist(series.dropna(), bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, outfile: Path):
    plt.figure()
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_bar(counts: pd.Series, title: str, xlabel: str, ylabel: str, outfile: Path):
    plt.figure(figsize=(10, 4))
    counts.sort_index().plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_az(series, title, outfile):
    cipic_angles = np.array(
        [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15,
         -10, -5, 0, 5, 10, 15, 20, 25, 30, 35,
         40, 45, 55, 65, 80], dtype=float
    )
    # Half-way boundaries between successive angles
    edges = np.r_[cipic_angles[:-1] + np.diff(cipic_angles) / 2,
                  cipic_angles[-1] + 7.5]  # add a final edge
    edges = np.r_[-87.5, edges]  # add a left edge
    # Plot
    plt.figure(figsize=(9, 4))
    plt.hist(series, bins=edges, edgecolor="k")
    plt.xticks(cipic_angles, rotation=45)
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

# ---------------------------
# Main analysis routine
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse cocktail‑party dataset CSV")
    parser.add_argument("--csv", type=Path, required=True, help="Path to train.csv (or val/test)")
    parser.add_argument("--outdir", type=Path, default=None, help="Directory for output PNGs")
    args = parser.parse_args()

    csv_path: Path = args.csv.expanduser().resolve()
    outdir: Path = (args.outdir or csv_path.parent).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    total = len(df)
    print(df.columns)
    print("Total samples:", total)

    for col in ["snr_measured_db", "sisnr_db"]:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found – skipping.")
            continue
        series = df[col]
        print(f"\n{col} statistics (dB):")
        print(series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
        plot_histogram(series, title=f"Distribution of {col}", xlabel="dB", outfile=outdir / f"{col}_hist.png")

    # Scatter: SNR vs SI‑SNR (if both present)
    if {col for col in df.columns} >= {"snr_measured_db", "sisnr_db"}:
        plot_scatter(
            df["snr_measured_db"],
            df["sisnr_db"],
            title="SNR vs SI‑SNR",
            xlabel="SNR (dB)",
            ylabel="SI‑SNR (dB)",
            outfile=outdir / "snr_vs_sisnr.png",
        )

    if "target_speaker" in df.columns:
        counts = df["target_speaker"].value_counts().sort_index()
        print("\nTarget‑speaker counts:")
        print(counts.to_string())
        plot_bar(counts, title="Target‑Speaker Distribution", xlabel="Speaker ID", ylabel="Samples", outfile=outdir / "target_speaker_counts.png")

    if "hrir_subject" in df.columns:
        counts = df["hrir_subject"].value_counts()
        print("\nTop HRTF subjects by sample count:")
        print(counts.head(10).to_string())
        plot_bar(counts, title="HRTF Subject Usage", xlabel="Subject ID", ylabel="Samples", outfile=outdir / "hrtf_subject_usage.png")

    for az_col, el_col, tag in [("az1", "el1", "s1"), ("az2", "el2", "s2")]:
        if az_col in df.columns:
            plot_az(df[az_col], title=f"Azimuth distribution ({tag})", outfile=outdir / f"{tag}_azimuth_hist.png")
        if el_col in df.columns:
            plot_histogram(df[el_col], title=f"Elevation distribution ({tag})", xlabel="Elevation (deg)", outfile=outdir / f"{tag}_elevation_hist.png")

    print(f"\nAnalysis complete – figures saved to: {outdir}\n")

if __name__ == "__main__":
    main()
