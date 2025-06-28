#!/usr/bin/env python3
import os
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.ndimage import binary_closing, binary_opening
import matplotlib.pyplot as plt

# ------------------- Parameters ------------------- #
SAMPLE_RATE = 16000
TARGET_CLIP_COUNT = 60000
LENGTH_RANGE = (2.5, 3.0)
LIBRISPEECH_DIR = Path("data/LibriSpeech/train-clean-360/")
OUT_DIR = Path("data-generation/premix/premix_clips")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_INDEX = Path("data-generation/premix/premix_index.csv")
TRAINVAL_LIST = Path("data-generation/premix/trainval_pool.txt")
TEST_LIST = Path("data-generation/premix/test_pool.txt")
PLOT_PATH = Path("data-generation/premix/clip_duration_distribution.png")

# Pause removal settings
FRAME_MS = 10
REL_DB = -35
ABS_DB = -55
MAX_GAP_MS = 200
MIN_SPEECH_MS = 120
LEAD_PAD_MS = 50

# ------------------- Silence Removal ------------------- #
def build_keep_mask(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    hop = int(FRAME_MS * SAMPLE_RATE / 1000)
    n = len(x) // hop
    frames = x[:n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((frames**2).mean(1) + 1e-12))
    speech = (db > db.max() + REL_DB) & (db > ABS_DB)
    speech = binary_closing(speech, np.ones(int(MAX_GAP_MS / FRAME_MS)))
    speech = binary_opening(speech, np.ones(int(MIN_SPEECH_MS / FRAME_MS)))
    speech = np.convolve(speech.astype(int), np.ones(int(LEAD_PAD_MS / FRAME_MS)), mode="same") > 0
    mask = np.repeat(speech, hop)
    return np.pad(mask, (0, len(x) - len(mask)), constant_values=False)

# ------------------- File Traversal ------------------- #
def collect_librispeech_files(root: Path):
    return list(root.rglob("*.flac"))

# ------------------- Processing Logic ------------------- #
def process_file(filepath: Path, clip_id: int) -> tuple[str, float] | None:
    try:
        audio, sr = sf.read(filepath)
        if sr != SAMPLE_RATE:
            return None, None

        mask = build_keep_mask(audio)
        cleaned = audio[mask]

        # Hard lower bound in samples
        min_samples = int(LENGTH_RANGE[0] * SAMPLE_RATE)
        max_samples = int(LENGTH_RANGE[1] * SAMPLE_RATE)

        if len(cleaned) < min_samples:
            return None, len(cleaned) / SAMPLE_RATE

        if len(cleaned) > max_samples:
            target_len = random.randint(min_samples, max_samples)
            start = random.randint(0, len(cleaned) - target_len)
            cleaned = cleaned[start:start + target_len]

        # Recheck trimmed length
        if len(cleaned) < min_samples:
            return None, len(cleaned) / SAMPLE_RATE

        out_name = f"clip_{clip_id:06d}.wav"
        out_path = OUT_DIR / out_name
        sf.write(out_path, cleaned, SAMPLE_RATE)

        speaker_id = filepath.parts[-3]
        chapter_id = filepath.parts[-2]
        utt_id = filepath.stem

        duration = len(cleaned) / SAMPLE_RATE
        return (out_name, duration, speaker_id, chapter_id, utt_id), duration

    except Exception:
        return None, None

# ------------------- Main Routine ------------------- #
def main():
    files = collect_librispeech_files(LIBRISPEECH_DIR)
    print(f"Found {len(files)} LibriSpeech files.")

    results = []
    durations = []
    clip_id = 0
    CSV_INDEX.parent.mkdir(parents=True, exist_ok=True)

    with CSV_INDEX.open("w", newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["clip", "duration", "speaker_id", "chapter_id", "utt_id"])

        for file in tqdm(files, desc="Processing clips"):
            if clip_id >= TARGET_CLIP_COUNT:
                break
            result, dur = process_file(file, clip_id)

            # Only log and use clips that passed all checks
            if result and dur >= 2.5:
                durations.append(dur)
                writer.writerow(result)
                results.append(result[0])
                clip_id += 1

            


    print(f"✅ Collected {clip_id} valid clips.")

    # Shuffle and split
    random.shuffle(results)
    with TRAINVAL_LIST.open("w") as f:
        f.writelines(f"{clip}\n" for clip in results[:40000])
    with TEST_LIST.open("w") as f:
        f.writelines(f"{clip}\n" for clip in results[40000:])

    print("✅ Saved trainval and test pools.")

    # Plot histogram of all durations
    plt.figure(figsize=(10, 5))
    plt.hist(durations, bins=60, edgecolor='black')
    plt.title("Distribution of Trimmed Clip Durations")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of clips")
    plt.grid(True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"📊 Duration distribution saved to: {PLOT_PATH}")

if __name__ == "__main__":
    main()
