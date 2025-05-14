import os
import numpy as np
import soundfile as sf
import scipy.io
from scipy.signal import fftconvolve
import csv
import random

# Configuration
sample_rate = 16000
train_ratio = 0.8
test_ratio = 0.1  # 10% test set
early_onset_ms = 0
early_onset_samples = int(sample_rate * early_onset_ms / 1000)
overlap_ratio = 1.0
snr_db = 10.0
narrow_target_angle = False

# Paths
base_path = "../data/SparseLibriMix/output/sparse_2_1/wav16000"
input_dir_s1 = os.path.join(base_path, "s1")
input_dir_s2 = os.path.join(base_path, "s2")
noise_dir = os.path.join(base_path, "noise")

output_root = "../datasets/static"
train_dir, val_dir, test_dir = (
    os.path.join(output_root, "train"),
    os.path.join(output_root, "val"),
    os.path.join(output_root, "test")
)
train_mix_dir, train_clean_dir = os.path.join(train_dir, "mixture"), os.path.join(train_dir, "clean")
val_mix_dir, val_clean_dir = os.path.join(val_dir, "mixture"), os.path.join(val_dir, "clean")
test_mix_dir, test_clean_dir = os.path.join(test_dir, "mixture"), os.path.join(test_dir, "clean")

os.makedirs(train_mix_dir, exist_ok=True)
os.makedirs(train_clean_dir, exist_ok=True)
os.makedirs(val_mix_dir, exist_ok=True)
os.makedirs(val_clean_dir, exist_ok=True)
os.makedirs(test_mix_dir, exist_ok=True)
os.makedirs(test_clean_dir, exist_ok=True)

train_csv = os.path.join(output_root, "train.csv")
val_csv = os.path.join(output_root, "val.csv")
test_csv = os.path.join(output_root, "test.csv")

# HRIR Subject setup
hrtf_base = "../data/cipic-hrtf-database/standard_hrir_database"

train_subjects = ['003', '008', '011', '015', '020', '033', '048', '051', '060', '126']
test_subjects = ['124', '135', '147', '165']  # Taken from original val pool
val_subjects = ['009', '010', '012', '044', '059', '154']  # Remaining val subjects (exclusive from test)

azimuths = np.array([-80, -65, -55, -45, -40, -35, -30, -25, -20, -15,
                     -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80])
elevations = np.linspace(-45, 230.625, 50)

# File list and split
files = sorted([f for f in os.listdir(input_dir_s1) if f.endswith('.wav')])
num_total = len(files)
num_train = int(train_ratio * num_total)
num_test = int(test_ratio * num_total)
num_val = num_total - num_train - num_test

train_indices = range(0, num_train)
val_indices = range(num_train, num_train + num_val)
test_indices = range(num_train + num_val, num_total)


def load_hrir(subject_id):
    path = os.path.join(hrtf_base, f"subject_{subject_id}/hrir_final.mat")
    data = scipy.io.loadmat(path)
    return data['hrir_l'], data['hrir_r']


def pad_to(arr, length):
    return np.pad(arr, (0, length - len(arr)), mode='constant')


def select_angles(is_target):
    if is_target:
        az_idx = np.random.randint(10, 15)
        el_idx = 25
    else:
        az_idx = np.random.randint(0, len(azimuths))
        el_idx = np.random.randint(0, len(elevations))
    return az_idx, el_idx


def apply_snr(clean_signal, noise_signal, snr_db):
    clean_signal = pad_to(clean_signal, len(noise_signal))
    noise_signal = pad_to(noise_signal, len(clean_signal))
    power_clean = np.mean(clean_signal ** 2)
    power_noise = np.mean(noise_signal ** 2)
    desired_noise_power = power_clean / (10 ** (snr_db / 10))
    scale = np.sqrt(desired_noise_power / (power_noise + 1e-9))
    return noise_signal * scale


def process_sample(i, idx, hrir_subjects, out_dir, csv_writer):
    file = files[idx]
    s1_path = os.path.join(input_dir_s1, file)
    s2_path = os.path.join(input_dir_s2, file)
    noise_path = os.path.join(noise_dir, file)

    audio1, _ = sf.read(s1_path)
    audio2, _ = sf.read(s2_path)
    noise, _ = sf.read(noise_path)

    subject_id = random.choice(hrir_subjects)
    hrir_left, hrir_right = load_hrir(subject_id)

    az_idx1, el_idx1 = select_angles(is_target=narrow_target_angle)
    az_idx2, el_idx2 = select_angles(is_target=False)
    az1, el1 = azimuths[az_idx1], elevations[el_idx1]
    az2, el2 = azimuths[az_idx2], elevations[el_idx2]

    target_speaker = 1 if abs(az1) < abs(az2) else 2

    offset_samples = int(len(audio1) * (1 - overlap_ratio))
    if target_speaker == 1:
        audio2 = np.pad(audio2, (offset_samples + early_onset_samples, 0), mode='constant')
        audio1 = np.pad(audio1, (early_onset_samples, 0), mode='constant')
        noise = np.pad(noise, (early_onset_samples, 0), mode='constant')
    else:
        audio1 = np.pad(audio1, (offset_samples + early_onset_samples, 0), mode='constant')
        audio2 = np.pad(audio2, (early_onset_samples, 0), mode='constant')
        noise = np.pad(noise, (early_onset_samples, 0), mode='constant')

    left1 = fftconvolve(audio1, hrir_left[az_idx1, el_idx1], mode='full')
    right1 = fftconvolve(audio1, hrir_right[az_idx1, el_idx1], mode='full')
    left2 = fftconvolve(audio2, hrir_left[az_idx2, el_idx2], mode='full')
    right2 = fftconvolve(audio2, hrir_right[az_idx2, el_idx2], mode='full')

    max_len = max(len(left1), len(left2), len(noise))
    left1, right1 = pad_to(left1, max_len), pad_to(right1, max_len)
    left2, right2 = pad_to(left2, max_len), pad_to(right2, max_len)
    noise = pad_to(noise, max_len)

    mix_left = left1 + left2
    mix_right = right1 + right2

    noise_scaled = apply_snr(mix_left, noise, snr_db)
    mix_left += noise_scaled
    mix_right += noise_scaled

    stereo = np.stack([mix_left, mix_right], axis=0)

    out_filename = f"mixture_{idx:07d}.wav"
    sf.write(os.path.join(out_dir, "mixture", out_filename), stereo.T, sample_rate)

    clean = np.stack([left1, right1], axis=0) if target_speaker == 1 else np.stack([left2, right2], axis=0)
    sf.write(os.path.join(out_dir, "clean", out_filename), clean.T, sample_rate)

    csv_writer.writerow([
        out_filename,
        file, az1, el1,
        file, az2, el2,
        early_onset_ms,
        overlap_ratio,
        snr_db,
        target_speaker,
        subject_id
    ])


# Train set
with open(train_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename", "s1_file", "az1", "el1", "s2_file", "az2", "el2",
        "early_onset_ms", "overlap_ratio", "snr_db", "target_speaker", "hrir_subject"
    ])
    for i, idx in enumerate(train_indices):
        process_sample(i, idx, train_subjects, train_dir, writer)

# Val set
with open(val_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename", "s1_file", "az1", "el1", "s2_file", "az2", "el2",
        "early_onset_ms", "overlap_ratio", "snr_db", "target_speaker", "hrir_subject"
    ])
    for i, idx in enumerate(val_indices):
        process_sample(i, idx, val_subjects, val_dir, writer)

# Test set
with open(test_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename", "s1_file", "az1", "el1", "s2_file", "az2", "el2",
        "early_onset_ms", "overlap_ratio", "snr_db", "target_speaker", "hrir_subject"
    ])
    for i, idx in enumerate(test_indices):
        process_sample(i, idx, test_subjects, test_dir, writer)

print("Dataset generation complete.")
