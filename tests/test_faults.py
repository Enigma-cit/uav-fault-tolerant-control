import numpy as np
from uav_ftc.faults.sensor_faults import SensorFaultModel
from uav_ftc.faults.actuator_faults import ActuatorFaultModel

 
def test_sensor_dropout_returns_none():
    fault = SensorFaultModel(dropout_prob=1.0, seed=0)
    meas = np.array([1.0, 2.0, 3.0])
    assert fault.apply(meas) is None


def test_actuator_effectiveness_scales_rotors():
    eff = np.array([1.0, 0.5, 0.0, 1.0])
    fault = ActuatorFaultModel(effectiveness=eff, disturbance_std=0.0, seed=0)
    u = np.array([10.0, 10.0, 10.0, 10.0])
    faulty = fault.apply(u)
    assert faulty[0] == 10.0
    assert faulty[1] == 5.0
    assert faulty[2] == 0.0
    assert faulty[3] == 10.0
