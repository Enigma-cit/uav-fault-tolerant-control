import numpy as np


class ActuatorFaultModel:
    """
    Actuator fault model with:

    - Loss of effectiveness per rotor (multiplicative scaling)
    - Additive disturbance torque

    Fault parameters can be time-varying or constant.
    """

    def __init__(self,
                 effectiveness: np.ndarray | None = None,
                 disturbance_std: float = 0.0,
                 seed: int | None = None):
        self.effectiveness = effectiveness  # shape (4,), scale in [0, 1]
        self.disturbance_std = disturbance_std
        self.rng = np.random.default_rng(seed)

    def apply(self, rotor_speeds_sq: np.ndarray) -> np.ndarray:
        if self.effectiveness is None:
            eff = np.ones_like(rotor_speeds_sq)
        else:
            eff = self.effectiveness

        faulty = eff * rotor_speeds_sq

        # Add zero-mean disturbance in torque space approx via rotor speed perturbation
        disturbance = self.disturbance_std * self.rng.standard_normal(rotor_speeds_sq.shape)
        return faulty + disturbance
