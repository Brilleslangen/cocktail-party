import os
import numpy as np
import soundfile as sf
import scipy.io
from scipy.signal import fftconvolve

# Generates binaural audio
# TODO: animate movement of the speakers

hrir_path = "cipic-hrtf-database/standard_hrir_database/subject_003/hrir_final.mat"
hrir_data = scipy.io.loadmat(hrir_path)

hrir_left = hrir_data['hrir_l']  # (azimuths, elevations, samples)
hrir_right = hrir_data['hrir_r']

azimuths = np.array([
    -80, -65, -55, -45, -40, -35, -30, -25, -20, -15,
    -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80
])

elevations = np.linspace(-45, 230.625, 50)

input_dir = 'data/sparse_2_0_monaural/wav8000/s1'
output_dir = 'data/binaural/s1_binaural_random'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith('.wav'):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    audio, sr = sf.read(input_path)

    # Randomly pick azimuth and elevation
    az_idx = np.random.randint(0, len(azimuths))
    el_idx = np.random.randint(0, len(elevations))

    # Extract HRIRs
    left_hrir = hrir_left[az_idx, el_idx, :]
    right_hrir = hrir_right[az_idx, el_idx, :]

    # Convolve
    left_channel = fftconvolve(audio, left_hrir, mode='full')
    right_channel = fftconvolve(audio, right_hrir, mode='full')

    # Trim
    min_len = min(len(left_channel), len(right_channel))
    left_channel = left_channel[:min_len]
    right_channel = right_channel[:min_len]

    # Stack to stereo
    stereo = np.stack([left_channel, right_channel], axis=1)
    sf.write(output_path, stereo, sr)

    print(f"Processed {filename} with random azimuth {azimuths[az_idx]}°, elevation {elevations[el_idx]}°")

print("All files processed with randomized spatial positions")
