import numpy as np


class IMUModel:
    def __init__(self, accel_bias: np.ndarray, gyro_bias: np.ndarray,
                 accel_noise_std: float, gyro_noise_std: float, seed: int | None = None):
        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.rng = np.random.default_rng(seed)

    def measure(self, true_accel: np.ndarray, true_gyro: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        accel_noise = self.accel_noise_std * self.rng.standard_normal(true_accel.shape)
        gyro_noise = self.gyro_noise_std * self.rng.standard_normal(true_gyro.shape)
        return true_accel + self.accel_bias + accel_noise, \
               true_gyro + self.gyro_bias + gyro_noise
