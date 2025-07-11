import os
import csv
import random
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm
import pyroomacoustics as pra
from scipy.signal import fftconvolve
import scipy.io

SAMPLE_RATE = 16000
N_MIXES = 16000
MIN_ANGLE_DIFF = 10.0
AZIMUTH_MEAN = 0 
AZIMUTH_STD = 30

PREMIX_CLIPS = Path("data-generation/premix/premix_clips")
TRAINVAL_LIST = Path("data-generation/premix/trainval_pool.txt").resolve()
TEST_LIST = Path("data-generation/premix/test_pool.txt").resolve()
OUTPUT_DIR = Path("datasets/static")

WHAM_NOISE_DIR = Path("data/wham_noise/tr")
HRIR_DIR = Path("data/cipic-hrtf-database/standard_hrir_database")

SUBJECTS = sorted(HRIR_DIR.glob("subject_*"))
np.random.seed(42)
np.random.shuffle(SUBJECTS)
TRAIN_SUBJECTS = SUBJECTS[:20]
VAL_SUBJECTS = SUBJECTS[20:25]
TEST_SUBJECTS = SUBJECTS[25:30]

CIPIC_AZIMUTHS = np.array([-80, -65, -55, -45, -40, -35, -30, -25, -20, -15,
                           -10, -5, 0, 5, 10, 15, 20, 25, 30, 35,
                           40, 45, 55, 65, 80])

def load_audio(name):
    data, sr = sf.read(PREMIX_CLIPS / name)
    assert sr == SAMPLE_RATE
    return data

def load_hrir(azimuth, subject_path):
    mat_data = scipy.io.loadmat(subject_path / "hrir_final.mat")
    azimuths = CIPIC_AZIMUTHS
    elev_idx = 8 # Head height
    az_idx = np.argmin(np.abs(azimuths - azimuth))
    hrir_l = mat_data['hrir_l'][az_idx, elev_idx, :]
    hrir_r = mat_data['hrir_r'][az_idx, elev_idx, :]
    return hrir_l.flatten(), hrir_r.flatten()

def simulate_brir(room_dim, source_pos, mic_pos, azimuth, subject_path):
    room = pra.ShoeBox(room_dim, fs=SAMPLE_RATE, max_order=17, absorption=0.4)
    ear_distance = 0.18
    mic_positions = np.array([
        [mic_pos[0] - ear_distance / 2, mic_pos[1], mic_pos[2]],
        [mic_pos[0] + ear_distance / 2, mic_pos[1], mic_pos[2]]
    ]).T
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
    room.add_source(source_pos)
    room.compute_rir()
    rir_left, rir_right = room.rir[0][0], room.rir[1][0]
    hrir_left, hrir_right = load_hrir(azimuth, subject_path)
    brir_left = fftconvolve(rir_left, hrir_left)
    brir_right = fftconvolve(rir_right, hrir_right)
    min_len = min(len(brir_left), len(brir_right))
    return brir_left[:min_len], brir_right[:min_len]

def simulate_clean(room_dim, source_pos, mic_pos, azimuth, subject_path):
    room = pra.ShoeBox(room_dim, fs=SAMPLE_RATE, max_order=0, absorption=1.0)
    ear_distance = 0.1449 # Average from CIPIC HRTF database
    mic_positions = np.array([
        [mic_pos[0] - ear_distance / 2, mic_pos[1], mic_pos[2]],
        [mic_pos[0] + ear_distance / 2, mic_pos[1], mic_pos[2]]
    ]).T
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
    room.add_source(source_pos)
    room.compute_rir()
    rir_left, rir_right = room.rir[0][0], room.rir[1][0]
    hrir_left, hrir_right = load_hrir(azimuth, subject_path)
    brir_left = fftconvolve(rir_left, hrir_left)
    brir_right = fftconvolve(rir_right, hrir_right)
    min_len = min(len(brir_left), len(brir_right))
    return brir_left[:min_len], brir_right[:min_len]


def pos_from_azimuth(azimuth, distance, mic_pos, room_dim):
    angle_rad = np.deg2rad(azimuth)
    x = mic_pos[0] + distance * np.cos(angle_rad)
    y = mic_pos[1] + distance * np.sin(angle_rad)
    return [np.clip(x, 0.1, room_dim[0]-0.1), np.clip(y, 0.1, room_dim[1]-0.1), mic_pos[2]]


def adjust_loudness(target, interferer, snr_db=5.0):
    """
    Adjust loudness of interferer to ensure target has higher energy.
    """
    target_energy = np.sum(target ** 2)
    interferer_energy = np.sum(interferer ** 2)
    desired_interferer_energy = target_energy / (10 ** (snr_db / 10))
    scale_factor = np.sqrt(desired_interferer_energy / (interferer_energy + 1e-10))
    return interferer * scale_factor


def load_wham_noise(target_len):
    noise_files = list(WHAM_NOISE_DIR.glob("*.wav"))
    noise, sr = sf.read(random.choice(noise_files))
    assert sr == SAMPLE_RATE
    if noise.ndim > 1:
        noise = noise[:, 0]  # Ensure mono
    if len(noise) < target_len:
        noise = np.tile(noise, int(np.ceil(target_len / len(noise))))
    return noise[:target_len]

def round_to_nearest_hrir_azimuth(azimuth):
    return CIPIC_AZIMUTHS[np.argmin(np.abs(CIPIC_AZIMUTHS - azimuth))]

def process_pair(f1, f2, split, writer):
    x1, x2 = load_audio(f1), load_audio(f2)
    min_len = min(len(x1), len(x2))
    x1, x2 = x1[:min_len], x2[:min_len]

    room_dim = np.random.uniform([6, 5, 2.5], [10, 8, 4])
    mic_pos = [room_dim[0]/2, room_dim[1]/2, 1.5]

    while True:
        az1 = float(np.random.normal(AZIMUTH_MEAN, AZIMUTH_STD))
        if -80 <= az1 <= 80:
            break

    angle_diff = float(np.clip(np.random.normal(loc=35, scale=10), MIN_ANGLE_DIFF, 80))
    direction = np.random.choice([-1, 1])
    az2 = float(az1 + (direction * angle_diff))

    if az2 < -80 or az2 > 80:
        az2 = az1 - direction * angle_diff

    if not (-80 < az1 < 80 and -80 < az2 < 80):
        print(f"Exiting: pair {f1}, {f2} due to azimuth limits: az1={az1}, az2={az2}, {type(az1), type(az2)}")
        exit(0)
    

    az1 = round_to_nearest_hrir_azimuth(az1)
    az2 = round_to_nearest_hrir_azimuth(az2)


    d1, d2 = np.random.uniform(1.0, 2.5, size=2)
    s1_pos = pos_from_azimuth(az1, d1, mic_pos, room_dim)
    s2_pos = pos_from_azimuth(az2, d2, mic_pos, room_dim)

    subject = random.choice(TRAIN_SUBJECTS if split == "train" else VAL_SUBJECTS if split == "val" else TEST_SUBJECTS)

    brir1_left, brir1_right = simulate_brir(room_dim, s1_pos, mic_pos, az1, subject)
    brir2_left, brir2_right = simulate_brir(room_dim, s2_pos, mic_pos, az2, subject)

    s1_left = fftconvolve(x1, brir1_left)
    s1_right = fftconvolve(x1, brir1_right)
    s2_left = fftconvolve(x2, brir2_left)
    s2_right = fftconvolve(x2, brir2_right)

    min_binaural_len = min(len(s1_left), len(s1_right), len(s2_left), len(s2_right), min_len)

    s1_binaural = np.vstack([s1_left[:min_binaural_len], s1_right[:min_binaural_len]]).T
    s2_binaural = np.vstack([s2_left[:min_binaural_len], s2_right[:min_binaural_len]]).T

        # Identify target and interferer based on absolute azimuth values
    if abs(az1) < abs(az2):
        target_binaural, interferer_binaural = s1_binaural, s2_binaural
    else:
        target_binaural, interferer_binaural = s2_binaural, s1_binaural

    # Adjust interferer loudness explicitly to achieve desired SNR (target louder)
    interferer_binaural_adjusted = adjust_loudness(target_binaural, interferer_binaural, snr_db=5.0)

    # Create final mix
    mix_clean = target_binaural + interferer_binaural_adjusted

    noise = load_wham_noise(min_binaural_len)
    noise_binaural = np.stack([noise, noise], axis=1)

    noise = load_wham_noise(min_binaural_len)
    noise_binaural = np.vstack([noise, noise]).T  # Ensures shape (N,2)

    mix = mix_clean + noise_binaural


    target_audio, target_pos = (x1, s1_pos) if abs(az1) < abs(az2) else (x2, s2_pos)
    clean_left_sim, clean_right_sim = simulate_clean(room_dim, target_pos, mic_pos, az1 if abs(az1) < abs(az2) else az2, subject)

    clean_left = fftconvolve(target_audio, clean_left_sim)
    clean_right = fftconvolve(target_audio, clean_right_sim)

    min_clean_len = min(len(clean_left), len(clean_right), min_binaural_len)

    clean_audio = np.vstack([clean_left[:min_clean_len], clean_right[:min_clean_len]]).T
    mix = mix[:min_clean_len]

    snr_measured = 10 * np.log10(np.sum(clean_audio ** 2) / np.sum((mix - clean_audio) ** 2))

    threshold = 5 if split in ["train", "val"] else 0
    if snr_measured > threshold:
        return False  # Indicate failure to meet SNR criterion

    (OUTPUT_DIR/split/"mixture").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR/split/"clean").mkdir(parents=True, exist_ok=True)

    fname_mix = f"mix_{Path(f1).stem}_{Path(f2).stem}.wav"
    fname_clean = f"clean_{Path(f1).stem}_{Path(f2).stem}.wav"


    try:
        sf.write(OUTPUT_DIR/split/"mixture"/fname_mix, mix, SAMPLE_RATE)
        sf.write(OUTPUT_DIR/split/"clean"/fname_clean, clean_audio, SAMPLE_RATE)

        writer.writerow([fname_mix, f1, az1, f2, az2, snr_measured])
    except Exception as e:
        print(f"Error writing files for {f1}, {f2}: {e}")
        return False

    return True

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, n_mixes in [("train", 16000), ("val", 4000), ("test", 1000)]:
        csvf = OUTPUT_DIR / f"{split}.csv"
        csvf.parent.mkdir(parents=True, exist_ok=True)
        with csvf.open("w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["mixture", "s1_file", "az1", "s2_file", "az2", "snr"])

            pool_file = TRAINVAL_LIST if split in ("train", "val") else TEST_LIST
            files = [l.strip() for l in pool_file.read_text().splitlines() if l.strip()]

            generated = 0
            with tqdm(total=n_mixes, desc=f"Generating {split}") as pbar:
                while generated < n_mixes:
                    f1, f2 = random.sample(files, 2)
                    success = process_pair(f1, f2, split, writer)
                    if success:
                        generated += 1
                        pbar.update(1)


if __name__ == "__main__":
    main()
