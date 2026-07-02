import numpy as np


def compute_metrics(time: np.ndarray, p_traj: np.ndarray, p_ref: np.ndarray,
                    u_traj: np.ndarray) -> dict:
    """
    Compute time-domain resilience metrics: 

    - RMS tracking error
    - max overshoot
    - settling time (epsilon-band)
    - control effort
    """
    e = p_traj - p_ref
    rms_error = float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
    overshoot = float(np.max(p_traj[:, 2] - p_ref[:, 2]))  # z-axis overshoot

    # Settling time: first time when |error| < eps for all subsequent times
    eps = 0.05
    idx = len(time) - 1
    for k in range(len(time)):
        if np.all(np.linalg.norm(e[k:], axis=1) < eps):
            idx = k
            break
    settling_time = float(time[idx])

    control_effort = float(np.mean(np.linalg.norm(u_traj, axis=1)))

    return {
        "rms_error": rms_error,
        "overshoot": overshoot,
        "settling_time": settling_time,
        "control_effort": control_effort,
    }
