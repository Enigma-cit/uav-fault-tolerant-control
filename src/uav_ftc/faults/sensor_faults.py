import numpy as np


class SensorFaultModel:
    """
    Simple stochastic sensor fault model supporting:

    - Dropout (missing measurements)
    - Bias (constant offset)
    - Stuck-at (frozen output)
    """

    def __init__(self,
                 dropout_prob: float = 0.0,
                 bias_mean: float = 0.0,
                 bias_std: float = 0.0,
                 stuck_prob: float = 0.0,
                 seed: int | None = None):
        self.dropout_prob = dropout_prob
        self.bias_mean = bias_mean
        self.bias_std = bias_std
        self.stuck_prob = stuck_prob
        self.rng = np.random.default_rng(seed)
        self._stuck_value = None

    def apply(self, true_measurement: np.ndarray) -> np.ndarray | None:
        # Dropout
        if self.rng.random() < self.dropout_prob:
            return None

        # Stuck-at fault
        if self.rng.random() < self.stuck_prob:
            if self._stuck_value is None:
                self._stuck_value = true_measurement.copy()
            return self._stuck_value

        # Bias
        bias = self.bias_mean + self.bias_std * self.rng.standard_normal(true_measurement.shape)
        return true_measurement + bias
