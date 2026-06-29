import numpy as np


class LiDARModel:
    def __init__(self, range_noise_std: float, seed: int | None = None):
        self.range_noise_std = range_noise_std
        self.rng = np.random.default_rng(seed)

    def measure_ranges(self, true_ranges: np.ndarray) -> np.ndarray:
        noise = self.range_noise_std * self.rng.standard_normal(true_ranges.shape)
        return true_ranges + noise
