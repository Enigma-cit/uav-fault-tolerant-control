import numpy as np
from uav_ftc.config import PIDGains, QuadrotorParams
from uav_ftc.control.pid_cascade import CascadedPIDController
from uav_ftc.control.fault_tolerant_controller import FaultTolerantController
 

def test_pid_zero_error_zero_torque():
    gains = PIDGains()
    params = QuadrotorParams()
    ctrl = CascadedPIDController(gains, params)

    p_ref = np.zeros(3)
    v_ref = np.zeros(3)
    eta_ref = np.zeros(3)
    p = np.zeros(3)
    v = np.zeros(3)
    eta = np.zeros(3)
    omega = np.zeros(3)

    u = ctrl.compute_control(p_ref, v_ref, eta_ref, p, v, eta, omega, dt=0.01)
    # With zero error, torques should be near zero (but thrust balances gravity).
    assert abs(u[1]) < 1e-6
    assert abs(u[2]) < 1e-6
    assert abs(u[3]) < 1e-6


def test_fault_tolerant_allocation_respects_effectiveness():
    gains = PIDGains()
    params = QuadrotorParams()
    effectiveness = np.array([1.0, 0.5, 0.5, 1.0])

    ftc = FaultTolerantController(gains, params, effectiveness=effectiveness)

    p_ref = np.zeros(3)
    v_ref = np.zeros(3)
    eta_ref = np.zeros(3)
    p = np.zeros(3)
    v = np.zeros(3)
    eta = np.zeros(3)
    omega = np.zeros(3)

    rotor_sq = ftc.compute_fault_tolerant_control(p_ref, v_ref, eta_ref, p, v, eta, omega, dt=0.01)
    assert rotor_sq.shape == (4,)
    assert np.all(rotor_sq >= 0.0)
