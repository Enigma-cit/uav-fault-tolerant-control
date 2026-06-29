import numpy as np
from uav_ftc.config import PIDGains, QuadrotorParams


class CascadedPIDController:
    """
    Cascaded PID control for quadrotor:

    - Outer loop: position (x, y, z) → desired roll/pitch and thrust
    - Inner loop: attitude (φ, θ, ψ) → body torque commands
    """

    def __init__(self, gains: PIDGains, params: QuadrotorParams):
        self.gains = gains
        self.params = params
        self.att_integral = np.zeros(3)

    def compute_control(self, p_ref: np.ndarray, v_ref: np.ndarray,
                        eta_ref: np.ndarray, p: np.ndarray, v: np.ndarray,
                        eta: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute [T, τ_φ, τ_θ, τ_ψ] given reference + current state.
        """
        # Position errors
        e_p = p_ref - p
        e_v = v_ref - v

        # Desired acceleration in world frame (ignoring constraints)
        a_des = self.gains.kp_xyz * e_p + self.gains.kd_xyz * e_v

        # Map desired acceleration to thrust and attitude commands
        g = self.params.gravity
        T_des = self.params.mass * (g + a_des[2])

        # Small-angle approximation for desired roll/pitch from lateral accelerations
        phi_des = -(a_des[1] / g)
        theta_des = (a_des[0] / g)
        psi_des = eta_ref[2]

        eta_des = np.array([phi_des, theta_des, psi_des])

        # Attitude errors
        e_eta = eta_des - eta
        e_omega = -omega

        # Integral update for attitude
        self.att_integral += e_eta * dt

        τ = (
            self.gains.kp_att * e_eta
            + self.gains.kd_att * e_omega
            + self.gains.ki_att * self.att_integral
        )

        return np.concatenate([[T_des], τ])
