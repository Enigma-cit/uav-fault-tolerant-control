import numpy as np


class AdversarialAttack:
    """
    Abstract adversarial attack on LiDAR-based perception.

    For this project, we model the attack as:
      - injecting spurious points,
      - removing points in an adversary-chosen region, or
      - perturbing perceived range.

    The actual LiDAR processing pipeline is abstracted; we focus on how attacks
    degrade reliability scores and pose estimates.
    """

    def __init__(self, strength: float = 0.5, seed: int | None = None):
        self.strength = strength
        self.rng = np.random.default_rng(seed)

    def apply_to_ranges(self, ranges: np.ndarray) -> np.ndarray:
        # Simple model: additive noise with structured bias
        noise = self.strength * self.rng.standard_normal(ranges.shape)
        bias = self.strength * 0.5
        return ranges + noise + bias
