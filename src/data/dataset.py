import glob
import os

import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, split_dir, sample_rate: int):
        self.mix_files = sorted(glob.glob(os.path.join(split_dir, "mixture", "*.wav")))
        self.clean_files = sorted(glob.glob(os.path.join(split_dir, "clean", "*.wav")))

        assert len(self.mix_files) == len(self.clean_files), (
            f"Found {len(self.mix_files)} mixtures but {len(self.clean_files)} clean files."
        )

        self.sample_rate = sample_rate

    def __getitem__(self, idx):
        mixture, sr1 = torchaudio.load(self.mix_files[idx])
        clean, sr2 = torchaudio.load(self.clean_files[idx])

        assert sr1 == sr2 == self.sample_rate, (
            f"Sample-rate mismatch: mix {sr1}, clean {sr2}, expected {self.sample_rate}"
        )

        return mixture, clean



