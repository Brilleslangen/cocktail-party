import os
import numpy as np
import soundfile as sf
import scipy.io
import csv
from scipy.signal import fftconvolve

# Generates binaural audio
# TODO: 2-speaker overlapping binaural audio
# TODO: animate movement of the speakers

hrir_path = "data/cipic-hrtf-database/standard_hrir_database/subject_003/hrir_final.mat"
input_dir_s1 = "data/sparse_2_1/wav48000/s1"
input_dir_s2 = "data/sparse_2_1/wav48000/s2"
output_dir = "data/binaural/s1s2_binaural_overlap"
csv_path = os.path.join(output_dir, "binaural_labels.csv")

overlap_ratio = 0.5  # 1.0 = full overlap, 0.0 = no overlap
sample_rate = 48000

hrir_data = scipy.io.loadmat(hrir_path)
hrir_left = hrir_data['hrir_l']  # (azimuths, elevations, samples)
hrir_right = hrir_data['hrir_r']

azimuths = np.array([
    -80, -65, -55, -45, -40, -35, -30, -25, -20, -15,
    -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80
])

elevations = np.linspace(-45, 230.625, 50)

os.makedirs(output_dir, exist_ok=True)
files = sorted([f for f in os.listdir(input_dir_s1) if f.endswith('.wav')])
num_mixtures = len(files)

with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "filename",
        "s1_file", "az1", "el1",
        "s2_file", "az2", "el2",
        "overlap_ratio", "target_speaker"
    ])

    for i, file in enumerate(files):
        s1_path = os.path.join(input_dir_s1, file)
        s2_path = os.path.join(input_dir_s2, file)

        audio1, sr1 = sf.read(s1_path)
        audio2, sr2 = sf.read(s2_path)
        assert sr1 == sr2 == sample_rate

        # Random az/el per speaker
        az_idx1, el_idx1 = np.random.randint(0, 25), np.random.randint(0, 50)
        az_idx2, el_idx2 = np.random.randint(0, 25), np.random.randint(0, 50)
        az1, el1 = azimuths[az_idx1], elevations[el_idx1]
        az2, el2 = azimuths[az_idx2], elevations[el_idx2]

        # Overlap control
        offset_samples = int(len(audio1) * (1 - overlap_ratio))
        audio2 = np.pad(audio2, (offset_samples, 0), mode="constant")

        # Convolve both speakers
        left1 = fftconvolve(audio1, hrir_left[az_idx1, el_idx1], mode="full")
        right1 = fftconvolve(audio1, hrir_right[az_idx1, el_idx1], mode="full")

        left2 = fftconvolve(audio2, hrir_left[az_idx2, el_idx2], mode="full")
        right2 = fftconvolve(audio2, hrir_right[az_idx2, el_idx2], mode="full")

        # Match lengths
        max_len = max(len(left1), len(left2), len(right1), len(right2))
        def pad_to(x, length):
            return np.pad(x, (0, length - len(x)), mode="constant")

        left1 = pad_to(left1, max_len)
        right1 = pad_to(right1, max_len)
        left2 = pad_to(left2, max_len)
        right2 = pad_to(right2, max_len)

        mix_left = left1 + left2
        mix_right = right1 + right2
        stereo = np.stack([mix_left, mix_right], axis=1)

        # Determine target speaker
        target_speaker = 1 if abs(az1) < abs(az2) else 2

        # Save
        out_filename = f"mix_{i+1:07d}.wav"
        out_path = os.path.join(output_dir, out_filename)
        sf.write(out_path, stereo, sample_rate)

        # Log
        writer.writerow([
            out_filename,
            file, az1, el1,
            file, az2, el2,
            overlap_ratio,
            target_speaker
        ])

        print(f"[{i+1}/{num_mixtures}] Saved {out_filename} | Target: Speaker {target_speaker}")

print("✅ All mixtures generated and labeled.")