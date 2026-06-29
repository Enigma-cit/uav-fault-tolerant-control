from dataclasses import dataclass
import numpy as np


@dataclass
class QuadrotorParams:
    mass: float = 1.2          # kg
    arm_length: float = 0.23   # m
    inertia: np.ndarray = np.diag([0.02, 0.02, 0.04])  # kg m^2
    gravity: float = 9.81      # m/s^2
    k_thrust: float = 1.9e-6   # N/(rad/s)^2
    k_drag: float = 2.5e-7     # Nm/(rad/s)^2


@dataclass
class SimulationParams:
    dt: float = 0.002          # s
    t_final: float = 10.0      # s
    process_noise_std: float = 0.0
    measurement_noise_std: float = 0.0
    seed: int = 0


@dataclass
class PIDGains:
    # Outer position loop
    kp_xyz: np.ndarray = np.array([1.0, 1.0, 3.0])
    kd_xyz: np.ndarray = np.array([0.7, 0.7, 2.0])
    # Inner attitude loop
    kp_att: np.ndarray = np.array([4.0, 4.0, 3.0])
    kd_att: np.ndarray = np.array([1.5, 1.5, 1.0])
    ki_att: np.ndarray = np.array([0.1, 0.1, 0.05])
