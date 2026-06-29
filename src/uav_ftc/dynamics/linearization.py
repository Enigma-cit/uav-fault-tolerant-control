import numpy as np
from uav_ftc.config import QuadrotorParams


def linearize_hover(params: QuadrotorParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearized dynamics around hover (small angles, hover thrust):

    Returns (A, B) for state x = [p, v, eta, omega] and inputs u = [T, τ_phi, τ_theta, τ_psi].
    This is a simplified small-angle model suitable for sanity checks and LTI analysis.
    """
    m = params.mass
    g = params.gravity
    J = params.inertia

    n_state = 12
    n_input = 4

    A = np.zeros((n_state, n_state))
    B = np.zeros((n_state, n_input))

    # p_dot = v
    A[0:3, 3:6] = np.eye(3)

    # v_dot ≈ [0, 0, (T/m - g)] with small-angle approximation, linearize around T0 = m g.
    # So perturbation in thrust δT yields δv_z_dot = δT / m.
    B[5, 0] = 1.0 / m

    # eta_dot ≈ omega
    A[6:9, 9:12] = np.eye(3)

    # omega_dot = J^{-1} τ (neglect cross terms at hover)
    B[9:12, 1:4] = np.linalg.inv(J)

    return A, B