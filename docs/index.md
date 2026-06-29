# UAV Fault-Tolerant Control Toolkit

This documentation provides an overview of the Python package `uav_ftc`, which
implements modeling and analysis tools for fault-tolerant quadrotor control.

## Modules

- `uav_ftc.dynamics`: Newton–Euler quadrotor dynamics and linearization tools.
- `uav_ftc.control`: Cascaded PID controllers, fault-tolerant control, and PID tuning.
- `uav_ftc.faults`: Sensor and actuator fault models, plus adversarial attacks.
- `uav_ftc.sensing`: IMU and LiDAR models with fusion reliability policy.
- `uav_ftc.experiments`: Scenario definitions, metrics, and experiment runners.
- `uav_ftc.viz`: Plotting and animation utilities.

See the `examples/` directory for end-to-end scripts demonstrating hover control,
fault injection experiments, and adversarial attack scenarios.