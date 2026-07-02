from dataclasses import dataclass, field
import numpy as np


@dataclass
class QuadrotorParams:
    mass: float = 1.2
    arm_length: float = 0.23
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.02, 0.02, 0.04]))
    gravity: float = 9.81
    k_thrust: float = 1.9e-6
    k_drag: float = 2.5e-7


@dataclass
class SimulationParams:
    dt: float = 0.002
    t_final: float = 10.0
    process_noise_std: float = 0.0
    measurement_noise_std: float = 0.0
    seed: int = 0


@dataclass
class PIDGains:
    kp_xyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 3.0], dtype=float))
    kd_xyz: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.7, 2.0], dtype=float))
    kp_att: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 3.0], dtype=float))
    kd_att: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.5, 1.0], dtype=float))
    ki_att: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.05], dtype=float))
