"""
uav_ftc: Fault-tolerant control toolkit for quadcopter UAVs.

This package provides:
- Newton–Euler quadrotor dynamics
- Cascaded PID control for attitude/position
- Fault models for sensors, actuators, and adversarial attacks
- L-BFGS-B-based PID tuning
- LiDAR–IMU fusion and sensor reliability policy
- Experiment runners for fault injection and resilience analysis
"""

__all__ = [
    "config",
    "dynamics",
    "control",
    "faults",
    "sensing",
    "experiments",
    "viz",
]
