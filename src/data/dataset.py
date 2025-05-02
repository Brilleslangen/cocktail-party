import glob
import os

import torchaudio
from omegaconf import DictConfig
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """
    Stereo mixture / stereo-clean dataset.

    Expects under `mix_dir/`:
      *.wav          ← stereo mixtures [2, T]
    and under `clean_dir/`:
      *.wav          ← stereo clean references [2, T]
    """

    def __init__(self, dataset_dir: DictConfig, sample_rate: int):
        self.mix_files = sorted(glob.glob(os.path.join(dataset_dir.mixture, "*.wav")))
        self.clean_files = sorted(glob.glob(os.path.join(dataset_dir.clean, "*.wav")))
        assert len(self.mix_files) == len(self.clean_files), (
            f"Found {len(self.mix_files)} mixtures but "
            f"{len(self.clean_files)} clean files."
        )
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mixture, sr1 = torchaudio.load(self.mix_files[idx])  # [2, T]
        clean, sr2 = torchaudio.load(self.clean_files[idx])  # [2, T]
        assert sr1 == sr2 == self.sample_rate, (
            f"Sample‐rate mismatch: mix {sr1}, clean {sr2}, expected {self.sample_rate}"
        )

        # split channels
        left_ref = clean[0]  # → [T]
        right_ref = clean[1]  # → [T]

        return mixture, left_ref, right_ref
