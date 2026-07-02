import numpy as np
from uav_ftc.config import PIDGains, QuadrotorParams
from uav_ftc.control.pid_cascade import CascadedPIDController

 
class FaultTolerantController:
    """
    Fault-tolerant controller for quadrotor:

    - Uses cascaded PID as nominal controller.
    - Performs simple control allocation accounting for rotor loss of effectiveness.
    """

    def __init__(self, gains: PIDGains, params: QuadrotorParams,
                 effectiveness: np.ndarray | None = None):
        self.nominal = CascadedPIDController(gains, params)
        if effectiveness is None:
            effectiveness = np.ones(4)
        self.effectiveness = effectiveness
        self.params = params

    def compute_fault_tolerant_control(self, p_ref, v_ref, eta_ref,
                                       p, v, eta, omega, dt) -> np.ndarray:
        """
        Returns rotor speed squared commands under actuator effectiveness vector.

        effectiveness[i] ∈ [0, 1] scales rotor i's contribution.
        """
        # Nominal desired [T, τ_phi, τ_theta, τ_psi]
        u_nom = self.nominal.compute_control(
            p_ref=p_ref, v_ref=v_ref,
            eta_ref=eta_ref,
            p=p, v=v,
            eta=eta, omega=omega,
            dt=dt,
        )

        T_des = u_nom[0]
        tau_des = u_nom[1:4]

        # Simple control allocation:
        # Solve for rotor speeds squared that achieve approximate thrust and torques,
        # given effectiveness scaling.
        kT = self.params.k_thrust
        L = self.params.arm_length
        kD = self.params.k_drag

        eff = self.effectiveness  # shape (4,)

        # Allocation matrix mapping rotor thrusts to [T, τ_phi, τ_theta, τ_psi]
        # Use per-rotor thrust variables f_i = kT * eff_i * ω_i^2.
        A = np.array([
            [eff[0], eff[1], eff[2], eff[3]],
            [-eff[0], eff[1], eff[2], -eff[3]],        # τ_phi / (L*kT)
            [eff[0], eff[1], -eff[2], -eff[3]],       # τ_theta / (L*kT)
            [eff[0], -eff[1], eff[2], -eff[3]],       # τ_psi / kD
        ])

        b = np.array([
            T_des,
            tau_des[0] / (L * kT),
            tau_des[1] / (L * kT),
            tau_des[2] / kD,
        ])

        # Least-squares allocation with non-negativity clipping.
        f, *_ = np.linalg.lstsq(A, b, rcond=None)
        f = np.clip(f, 0.0, np.inf)

        # Back out rotor speeds squared
        rotor_sq = f / (kT * eff + 1e-9)

        return rotor_sq
