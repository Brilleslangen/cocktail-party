import os
import csv
import random
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm
import pyroomacoustics as pra

SAMPLE_RATE = 16000
N_MIXES = 16000
SNR_TRAINVAL_CUTOFF = 5.0
SNR_TEST_CUTOFF = 0.0
MIN_ANGLE_DIFF = 10.0

PREMIX_CLIPS = Path("data-generation/premix/premix_clips")
TRAINVAL_LIST = Path("data-generation/premix/trainval_pool.txt").resolve()
TEST_LIST = Path("data-generation/premix/test_pool.txt").resolve()
OUTPUT_DIR = Path("datasets/static")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATHS = {
    "train": OUTPUT_DIR / "train.csv",
    "val": OUTPUT_DIR / "val.csv",
    "test": OUTPUT_DIR / "test.csv",
}

def load_audio(name):
    data, sr = sf.read(PREMIX_CLIPS / name)
    assert sr == SAMPLE_RATE
    return data

def simulate_direct_only(room_dim, mic_pos, source_pos, signal):
    room = pra.ShoeBox(room_dim, fs=SAMPLE_RATE, max_order=0, absorption=1.0)
    room.add_microphone_array(pra.MicrophoneArray(np.array([mic_pos]).T, room.fs))
    room.add_source(source_pos, signal=signal)
    room.compute_rir()
    room.simulate()
    return room.mic_array.signals[0]

def simulate_reverberant(room_dim, mic_pos, source_pos, signal):
    room = pra.ShoeBox(room_dim, fs=SAMPLE_RATE, max_order=17, absorption=0.4)
    room.add_microphone_array(pra.MicrophoneArray(np.array([mic_pos]).T, room.fs))
    room.add_source(source_pos, signal=signal)
    room.compute_rir()
    room.simulate()
    return room.mic_array.signals[0]

def process_pair(f1, f2, split, writer, snr_cutoff):
    x1 = load_audio(f1)
    x2 = load_audio(f2)
    if len(x1) != len(x2):
        return

    room_dim = np.random.uniform([6, 5, 2.5], [10, 8, 4])
    mic_pos = [room_dim[0] / 2, room_dim[1] / 2, 1.5]

    while True:
        az1 = np.random.normal(0, 30)
        az2 = np.random.normal(0, 30)
        if abs(az1 - az2) >= MIN_ANGLE_DIFF:
            break

    d1, d2 = np.random.uniform(1.0, 2.5, size=2)

    def pos_from_azimuth(az, d):
        angle_rad = np.deg2rad(az)
        x = mic_pos[0] + d * np.cos(angle_rad)
        y = mic_pos[1] + d * np.sin(angle_rad)
        z = mic_pos[2]
        return [np.clip(x, 0.1, room_dim[0] - 0.1), np.clip(y, 0.1, room_dim[1] - 0.1), z]

    s1_pos = pos_from_azimuth(az1, d1)
    s2_pos = pos_from_azimuth(az2, d2)

    s1_reverb = simulate_reverberant(room_dim, mic_pos, s1_pos, x1)
    s2_reverb = simulate_reverberant(room_dim, mic_pos, s2_pos, x2)

    min_len = min(len(s1_reverb), len(s2_reverb))
    s1_reverb = s1_reverb[:min_len]
    s2_reverb = s2_reverb[:min_len]
    mix = s1_reverb + s2_reverb

    measured_snr = 10 * np.log10(np.sum(s1_reverb ** 2) / (np.sum(s2_reverb ** 2) + 1e-8))
    if (split == "test" and measured_snr > SNR_TEST_CUTOFF) or (split != "test" and measured_snr > SNR_TRAINVAL_CUTOFF):
        return

    if abs(az1) <= abs(az2):
        target_idx = 1
        y_clean = simulate_direct_only(room_dim, mic_pos, s1_pos, x1)[:min_len]
    else:
        target_idx = 2
        y_clean = simulate_direct_only(room_dim, mic_pos, s2_pos, x2)[:min_len]

    name = f"mixture_{f1[:-4]}_{f2}"
    mix_dir = OUTPUT_DIR / split / "mixture"
    clean_dir = OUTPUT_DIR / split / "clean"
    mix_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    sf.write(mix_dir / name, mix, SAMPLE_RATE)
    sf.write(clean_dir / name, y_clean, SAMPLE_RATE)

    writer.writerow([
        name, f1, s1_pos, az1,
        f2, s2_pos, az2,
        0, 1.0, "NA",
        measured_snr, target_idx,
        room_dim.tolist()
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    for split in ("train", "val", "test"):
        csvf = CSV_PATHS[split]
        csvf.parent.mkdir(parents=True, exist_ok=True)
        with csvf.open("w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "s1_file", "s1_pos", "s1_azimuth",
                "s2_file", "s2_pos", "s2_azimuth",
                "early_onset_ms", "overlap_ratio", "snr_db",
                "snr_measured_db", "target_speaker", "room_dim"
            ])

            pool_file = TRAINVAL_LIST if split in ("train", "val") else TEST_LIST
            if not pool_file.exists():
                print(f"Missing pool list: {pool_file}")
                continue

            files = [l.strip() for l in pool_file.read_text().splitlines() if l.strip()]
            random.shuffle(files)

            n = 0
            max_n = 1000 if split == "test" else N_MIXES
            for _ in tqdm(range(max_n), desc=f"Generating {split}"):
                f1, f2 = random.sample(files, 2)
                mix_dir = OUTPUT_DIR / split / "mix"
                pre_count = len(list(mix_dir.glob("*.wav"))) if mix_dir.exists() else 0
                process_pair(f1, f2, split, writer, snr_cutoff=SNR_TEST_CUTOFF if split == "test" else SNR_TRAINVAL_CUTOFF)
                post_count = len(list(mix_dir.glob("*.wav")))
                if post_count > pre_count:
                    n += 1
                if n >= max_n:
                    break

if __name__ == "__main__":
    main()
