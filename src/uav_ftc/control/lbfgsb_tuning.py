from dataclasses import dataclass
from typing import Callable
 
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from uav_ftc.config import PIDGains
from uav_ftc.experiments.runner import run_fault_sweep


@dataclass
class TuningResult:
    gains: PIDGains
    cost: float
    success: bool
    message: str


class PIDTunerLBFGSB:
    """
    L-BFGS-B-based PID tuning:

    Minimize a robustness cost J(g) over a set of fault scenarios, where g collects
    PID gains. J can penalize overshoot, settling time, and control effort under
    sensor/actuator faults and adversarial attacks.
    """

    def __init__(self, cost_fn: Callable[[PIDGains], float]):
        self.cost_fn = cost_fn

    def tune(self, initial_gains: PIDGains,
             bounds: list[tuple[float, float]] | None = None) -> TuningResult:
        # Flatten gains to a vector
        x0 = np.concatenate([
            initial_gains.kp_xyz,
            initial_gains.kd_xyz,
            initial_gains.kp_att,
            initial_gains.kd_att,
            initial_gains.ki_att,
        ])

        def func(x: np.ndarray) -> float:
            gains = self._vector_to_gains(x)
            return self.cost_fn(gains)

        result_x, result_f, info = fmin_l_bfgs_b(
            func,
            x0,
            approx_grad=True,
            bounds=bounds,
            maxiter=50,
        )

        tuned_gains = self._vector_to_gains(result_x)
        return TuningResult(
            gains=tuned_gains,
            cost=result_f,
            success=info["warnflag"] == 0,
            message=info["task"],
        )

    @staticmethod
    def _vector_to_gains(x: np.ndarray) -> PIDGains:
        idx = 0

        kp_xyz = x[idx:idx + 3]; idx += 3
        kd_xyz = x[idx:idx + 3]; idx += 3
        kp_att = x[idx:idx + 3]; idx += 3
        kd_att = x[idx:idx + 3]; idx += 3
        ki_att = x[idx:idx + 3]; idx += 3

        return PIDGains(
            kp_xyz=kp_xyz,
            kd_xyz=kd_xyz,
            kp_att=kp_att,
            kd_att=kd_att,
            ki_att=ki_att,
        )
