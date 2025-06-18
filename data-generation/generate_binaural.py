#!/usr/bin/env python3
# ===========================================================================
#  Static 2-Speaker Binaural Dataset Generator (v2025-06-13)
# ---------------------------------------------------------------------------
#  • Target speaker  = talker whose |azimuth| is smaller (≈ “in front”)
#  • Interferer is **time-shifted** (±3 s search) to maximise VAD overlap
#    with the target before any trimming
#  • HRIR convolution (CIPIC), background noise at +100 dB SNR (≈ muted)
#  • Long pauses inside the target window are removed (controllable params)
#  • Logs SNR and SI-SNR (mono) to CSV
# ===========================================================================

from __future__ import annotations
import os, csv, random
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.io
from scipy.signal import fftconvolve

import torch
from src.evaluate.metrics import batch_snr, batch_si_snr

# ------------------------------- Hyper-params ------------------------------ #
SAMPLE_RATE        = 16_000
TRAIN_RATIO        = 0.80
TEST_RATIO         = 0.10             # rest → validation
OVERLAP_RATIO      = 1.0              # kept for legacy column in CSV
SNR_DB             = 10.0            # background ≈ muted
NARROW_TARGET_ANG  = False            # True → pick target az around 0°

# Pause-removal parameters
FRAME_MS       = 10
REL_DB         = -35
ABS_DB         = -55
MAX_GAP_MS     = 300
MIN_SPEECH_MS  = 120
LEAD_PAD_MS    = 50

# ------------------------------ File layout -------------------------------- #
BASE_PATH = Path("../data/output/10k/wav16000")            # <-- updated base
S1_DIR, S2_DIR, NOISE_DIR = BASE_PATH / "s1", BASE_PATH / "s2", BASE_PATH / "noise"

OUT_ROOT = Path("../datasets/static")
for s in ("train", "val", "test"):
    (OUT_ROOT / s / "mixture").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / s / "clean").mkdir(parents=True, exist_ok=True)

TRAIN_CSV, VAL_CSV, TEST_CSV = OUT_ROOT / "train.csv", OUT_ROOT / "val.csv", OUT_ROOT / "test.csv"

# ------------------------------ HRIR setup --------------------------------- #
HRIR_BASE = Path("../data/cipic-hrtf-database/standard_hrir_database")
TRAIN_SUBJECTS = ['003','008','011','015','020','033','048','051','060','126']
VAL_SUBJECTS   = ['009','010','012','044','059','154']
TEST_SUBJECTS  = ['124','135','147','165']

AZIMUTHS   = np.array([-80,-65,-55,-45,-40,-35,-30,-25,-20,-15,
                       -10,-5,0,5,10,15,20,25,30,35,40,45,55,65,80])
ELEVATIONS = np.linspace(-45, 230.625, 50)

# ------------------------------ Utilities ---------------------------------- #
def load_hrir(subj: str):
    data = scipy.io.loadmat(HRIR_BASE / f"subject_{subj}" / "hrir_final.mat")
    return data["hrir_l"], data["hrir_r"]

def pad_to(x: np.ndarray, L: int):
    return np.pad(x, (0, max(0, L - len(x))), mode="constant")

def select_angles():
    if NARROW_TARGET_ANG:
        return np.random.randint(10, 15), 25       # ±5° on horizontal plane
    return np.random.randint(len(AZIMUTHS)), np.random.randint(len(ELEVATIONS))

def apply_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float):
    clean, noise = pad_to(clean, len(noise)), pad_to(noise, len(clean))
    return noise * np.sqrt(clean.var() / (10**(snr_db/10) * noise.var() + 1e-9))

# ---------- VAD helpers ---------------------------------------------------- #
def vad_frames(x: np.ndarray,
               sr: int = SAMPLE_RATE,
               frame_ms: int = FRAME_MS,
               rel_db: float = REL_DB,
               abs_db: float = ABS_DB) -> np.ndarray:
    """Boolean VAD per frame (shape [N_frames])."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    hop = int(frame_ms * sr / 1000)
    n   = len(x) // hop
    if n == 0:
        return np.zeros(1, dtype=bool)
    frames = x[:n*hop].reshape(n, hop)
    db = 20*np.log10(np.sqrt((frames**2).mean(1)+1e-12))
    return (db > db.max() + rel_db) & (db > abs_db)

def build_keep_mask(x: np.ndarray,
                    sr: int = SAMPLE_RATE,
                    frame_ms: int = FRAME_MS,
                    rel_db: float = REL_DB,
                    abs_db: float = ABS_DB,
                    max_gap_ms: int = MAX_GAP_MS,
                    min_speech_ms: int = MIN_SPEECH_MS,
                    lead_pad_ms: int = LEAD_PAD_MS) -> np.ndarray:
    """Boolean per-sample mask that removes long pauses but keeps context."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    hop = int(frame_ms * sr / 1000)
    n   = len(x) // hop
    if n == 0:
        return np.ones(len(x), dtype=bool)
    frames = x[:n*hop].reshape(n, hop)
    db = 20*np.log10(np.sqrt((frames**2).mean(1)+1e-12))
    speech = (db > db.max() + rel_db) & (db > abs_db)

    from scipy.ndimage import binary_closing, binary_opening
    speech = binary_closing(speech, np.ones(int(max_gap_ms/frame_ms)))
    speech = binary_opening(speech, np.ones(int(min_speech_ms/frame_ms)))

    speech = np.convolve(speech.astype(int), np.ones(int(lead_pad_ms/frame_ms)), mode="same") > 0
    mask = np.repeat(speech, hop)
    return np.pad(mask, (0, len(x)-len(mask)), constant_values=False)

# --------------------------- Core sample builder --------------------------- #
def process_sample(file: str, split: str,
                   subject_pool: list[str], writer: csv.writer):
    # --- load mono sources -------------------------------------------------- #
    a1,_ = sf.read(S1_DIR / file)
    a2,_ = sf.read(S2_DIR / file)
    nz,_ = sf.read(NOISE_DIR / file)

    # --- HRIR + random spatial positions ----------------------------------- #
    subj = random.choice(subject_pool)
    hrir_l, hrir_r = load_hrir(subj)
    az1_idx, el1_idx = select_angles()
    az2_idx, el2_idx = select_angles()

    # --- choose target (smaller |azimuth|) ---------------------------------- #
    az1, az2 = AZIMUTHS[az1_idx], AZIMUTHS[az2_idx]
    if abs(az1) < abs(az2):
        tgt_audio, int_audio = a1, a2
        tgt_lidx, tgt_ridx   = az1_idx, el1_idx
        int_lidx, int_ridx   = az2_idx, el2_idx
        tgt_spk              = 1
    else:
        tgt_audio, int_audio = a2, a1
        tgt_lidx, tgt_ridx   = az2_idx, el2_idx
        int_lidx, int_ridx   = az1_idx, el1_idx
        tgt_spk              = 2

    # --- tile interferer if shorter than target ---------------------------- #
    if len(int_audio) < len(tgt_audio):
        reps = int(np.ceil(len(tgt_audio) / len(int_audio)))
        int_audio = np.tile(int_audio, reps)[:len(tgt_audio)]

    # --- best shift (±3 s search, 10 ms steps) ------------------------------ #
    hop = int(FRAME_MS * SAMPLE_RATE / 1000)
    m_t = vad_frames(tgt_audio)
    m_i = vad_frames(int_audio)
    max_shift_frames = int(3000 / FRAME_MS)
    best_shift, best_overlap = 0, -1
    for s in range(-max_shift_frames, max_shift_frames+1):
        if s < 0:
            ov = (m_t[:s] & m_i[-s:]).sum()
        elif s > 0:
            ov = (m_t[s:] & m_i[:-s]).sum()
        else:
            ov = (m_t & m_i).sum()
        if ov > best_overlap:
            best_overlap, best_shift = ov, s
    shift_samples = best_shift * hop
    if shift_samples > 0:
        int_audio = np.pad(int_audio, (shift_samples, 0))
    elif shift_samples < 0:
        int_audio = int_audio[-shift_samples:]
    if len(int_audio) < len(tgt_audio):
        int_audio = np.pad(int_audio, (0, len(tgt_audio)-len(int_audio)))

    # --- HRIR convolution --------------------------------------------------- #
    def convolve(x, lidx, ridx):
        return (fftconvolve(x, hrir_l[lidx,ridx], mode='full'),
                fftconvolve(x, hrir_r[lidx,ridx], mode='full'))

    tl, tr = convolve(tgt_audio, tgt_lidx, tgt_ridx)
    il, ir = convolve(int_audio, int_lidx, int_ridx)

    L = max(len(tl), len(il), len(nz))
    tl, tr, il, ir, nz = (pad_to(x, L) for x in (tl, tr, il, ir, nz))

    mix_l, mix_r = tl + il, tr + ir
    mix_l += apply_snr(mix_l, nz, SNR_DB)
    mix_r += apply_snr(mix_r, nz, SNR_DB)
    stereo_mix = np.stack([mix_l, mix_r], axis=1)
    clean_tgt  = np.stack([tl, tr], axis=1)

    # --- pause removal on TARGET ------------------------------------------- #
    keep = build_keep_mask(clean_tgt)
    stereo_mix = stereo_mix[keep]
    clean_tgt  = clean_tgt [keep]

    # --- metrics ----------------------------------------------------------- #
    with torch.no_grad():
        est = torch.from_numpy(stereo_mix.mean(1)).float().unsqueeze(0)
        ref = torch.from_numpy(clean_tgt.mean(1)).float().unsqueeze(0)
        m   = torch.ones_like(ref)
        snr_val   = batch_snr(est, ref, m).item()
        sisnr_val = batch_si_snr(est, ref, m).item()

    # --- save audio -------------------------------------------------------- #
    out_dir   = OUT_ROOT / split
    mix_path  = out_dir / "mixture" / f"mixture_{file}"
    clean_path= out_dir / "clean"   / f"mixture_{file}"
    sf.write(mix_path,  stereo_mix, SAMPLE_RATE)
    sf.write(clean_path, clean_tgt, SAMPLE_RATE)

    # --- CSV row ----------------------------------------------------------- #
    writer.writerow([
        mix_path.name,
        file, AZIMUTHS[az1_idx], ELEVATIONS[el1_idx],
        file, AZIMUTHS[az2_idx], ELEVATIONS[el2_idx],
        0, 1.0, SNR_DB,
        snr_val, sisnr_val,
        tgt_spk, subj,
    ])

# ------------------------------- Main loop --------------------------------- #
def run_generation():
    files = sorted(f for f in os.listdir(S1_DIR) if f.endswith(".wav"))
    n_total = len(files)
    n_train = int(n_total * TRAIN_RATIO)
    n_test  = int(n_total * TEST_RATIO)
    n_val   = n_total - n_train - n_test

    splits = [
        ("train", files[:n_train],              TRAIN_SUBJECTS, TRAIN_CSV),
        ("val",   files[n_train:n_train+n_val], VAL_SUBJECTS,   VAL_CSV),
        ("test",  files[n_train+n_val:],        TEST_SUBJECTS,  TEST_CSV),
    ]

    for split_name, file_list, pool, csv_path in splits:
        with csv_path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "filename",
                "s1_file","az1","el1",
                "s2_file","az2","el2",
                "early_onset_ms","overlap_ratio","snr_db",
                "snr_measured_db","sisnr_db",
                "target_speaker","hrir_subject",
            ])
            for f in file_list:
                process_sample(f, split_name, pool, writer)

    print("✅ Dataset generation complete.")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_generation()
