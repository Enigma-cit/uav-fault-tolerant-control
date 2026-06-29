import numpy as np
from uav_ftc.config import QuadrotorParams, SimulationParams
from uav_ftc.dynamics.quadrotor_model import QuadrotorDynamics


def test_hover_equilibrium():
    quad = QuadrotorDynamics(QuadrotorParams(), SimulationParams(dt=0.01, t_final=0.1))
    # Ideal hover: thrust balances weight, no torques.
    m = quad.params.mass
    g = quad.params.gravity
    T_hover = m * g
    rotor_sq = np.ones(4) * (T_hover / (4 * quad.params.k_thrust))

    for _ in range(100):
        quad.step(rotor_sq, rotor_input=True)

    assert np.linalg.norm(quad.p) < 1e-2
    assert abs(quad.v[2]) < 1e-2
