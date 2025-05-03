# src/data/bucket_sampler.py
import math
import random
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[list[int]]):
    """
    Groups dataset indices into buckets by length, then emits
    randomized batches within each bucket.

    Args:
        lengths (List[int]): list of sequence lengths (e.g. audio-frame counts).
        batch_size (int): how many samples per batch.
        n_buckets (int): number of equal-width length‐based buckets.
        shuffle (bool): whether to shuffle order of buckets and samples.
    """
    def __init__(self, lengths, batch_size, n_buckets, shuffle=True):
        super().__init__()
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

        # assign each idx into one of n_buckets based on length
        min_len, max_len = min(lengths), max(lengths)
        if max_len == min_len:
            # everyone the same length → single bucket
            self.buckets = [list(range(len(lengths)))]
        else:
            bucket_width = (max_len - min_len) / n_buckets
            self.buckets = [[] for _ in range(n_buckets)]
            for idx, L in enumerate(lengths):
                # compute bucket index
                b = int((L - min_len) / bucket_width)
                b = min(b, n_buckets - 1)
                self.buckets[b].append(idx)

        # precompute batches for each bucket
        self.batched_idxs = []
        for bucket in self.buckets:
            if shuffle:
                random.shuffle(bucket)
            # slice out batch_size chunks
            for i in range(0, len(bucket), batch_size):
                self.batched_idxs.append(bucket[i: i + batch_size])

        if shuffle:
            random.shuffle(self.batched_idxs)

    def __iter__(self):
        # yield each precomputed batch
        for batch in self.batched_idxs:
            yield batch

    def __len__(self):
        return len(self.batched_idxs)
