import numpy as np
from uav_ftc.config import QuadrotorParams, SimulationParams

 
class QuadrotorDynamics:
    """
    Newton–Euler quadrotor model in x-configuration.

    State x = [p, v, eta, omega] with:
      p    ∈ ℝ^3 : position in world frame
      v    ∈ ℝ^3 : linear velocity in world frame
      eta  ∈ ℝ^3 : roll-pitch-yaw (φ, θ, ψ)
      omega∈ ℝ^3 : body angular velocity

    Control input u can be either:
      - rotor speeds squared: u = [ω_1^2, ..., ω_4^2]
      - or equivalent thrust/torque vector: u = [T, τ_φ, τ_θ, τ_ψ].
    """

    def __init__(self, quad_params: QuadrotorParams, sim_params: SimulationParams):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.eta = np.zeros(3)    # [phi, theta, psi]
        self.omega = np.zeros(3)  # body rates
        self.params = quad_params
        self.sim = sim_params

    @staticmethod
    def _rotation_matrix(eta: np.ndarray) -> np.ndarray:
        phi, theta, psi = eta
        cφ, sφ = np.cos(phi), np.sin(phi)
        cθ, sθ = np.cos(theta), np.sin(theta)
        cψ, sψ = np.cos(psi), np.sin(psi)

        # ZYX convention
        Rz = np.array([[cψ, -sψ, 0],
                       [sψ,  cψ, 0],
                       [0,    0, 1]])
        Ry = np.array([[cθ, 0, sθ],
                       [0,  1, 0],
                       [-sθ, 0, cθ]])
        Rx = np.array([[1,  0,   0],
                       [0, cφ, -sφ],
                       [0, sφ,  cφ]])

        return Rz @ Ry @ Rx

    def _forces_and_moments_from_rotors(self, rotor_speeds_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Map rotor speeds squared to collective thrust and body torques.

        Assumes x-configuration:
          rotors at (+L, +L), (-L, +L), (-L, -L), (+L, -L)
        and standard spin directions.
        """
        kT = self.params.k_thrust
        kD = self.params.k_drag
        L = self.params.arm_length

        ω1_sq, ω2_sq, ω3_sq, ω4_sq = rotor_speeds_sq

        T = kT * (ω1_sq + ω2_sq + ω3_sq + ω4_sq)

        τφ = L * kT * (ω2_sq + ω3_sq - ω1_sq - ω4_sq)
        θτ = L * kT * (ω1_sq + ω2_sq - ω3_sq - ω4_sq)
        τψ = kD * (ω1_sq - ω2_sq + ω3_sq - ω4_sq)

        tau = np.array([τφ, θτ, τψ])
        return T, tau

    def step(self, u: np.ndarray, rotor_input: bool = True):
        """
        One integration step with forward Euler.

        Parameters
        ----------
        u : ndarray
            Control input. If rotor_input=True, u is rotor speeds squared (ω_i^2).
            Otherwise, u is [T, τ_φ, τ_θ, τ_ψ].
        """
        dt = self.sim.dt
        m = self.params.mass
        g = self.params.gravity
        J = self.params.inertia

        if rotor_input:
            T, tau = self._forces_and_moments_from_rotors(u)
        else:
            T = u[0]
            tau = u[1:4]

        R = self._rotation_matrix(self.eta)
        z_body = np.array([0.0, 0.0, 1.0])

        # Translational dynamics
        gravity = np.array([0.0, 0.0, -g])
        thrust_world = (T / m) * (R @ z_body)
        a = gravity + thrust_world

        # Rotational dynamics
        omega = self.omega
        omega_dot = np.linalg.solve(J, tau - np.cross(omega, J @ omega))

        # Kinematics
        self.p = self.p + dt * self.v
        self.v = self.v + dt * a
        self.eta = self.eta + dt * self._body_rates_to_euler_dot(omega, self.eta)
        self.omega = self.omega + dt * omega_dot

    @staticmethod
    def _body_rates_to_euler_dot(omega: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """
        Map body rates to Euler angle rates using standard kinematic relation.
        """
        p, q, r = omega
        phi, theta, _ = eta

        sφ, cφ = np.sin(phi), np.cos(phi)
        tθ = np.tan(theta)
        cθ = np.cos(theta)

        T = np.array([
            [1.0, sφ * tθ, cφ * tθ],
            [0.0, cφ,      -sφ],
            [0.0, sφ / cθ, cφ / cθ],
        ])
        return T @ omega
