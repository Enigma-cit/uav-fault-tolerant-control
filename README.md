# uav-fault-tolerant-control

Modeling and analysis of fault-tolerant controllers for quadcopter UAVs under sensor and actuator
fault scenarios, with adversarial attacks and dynamic sensor reliability policies.

This repository was originally developed in the context of the *Security of Safety-Critical Systems*
course at IIT Madras, and refactored into a research-grade Python package.

## Features

- Newton–Euler quadrotor dynamics (6-DOF rigid-body model).
- Cascaded PID control for attitude and position.
- Data-driven PID tuning via L-BFGS-B over fault scenarios.
- Explicit sensor fault models (dropouts, bias, stuck-at).
- Actuator fault models (loss of effectiveness, disturbance).
- Adversarial attacks on LiDAR perception.
- LiDAR–IMU fusion with dynamic sensor reliability policy.
- Experiment runner for time-domain simulations and resilience metrics.
- Example scripts for hover, fault injection, and adversarial scenarios.


# Repository Structure

uav-fault-tolerant-control/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── uav_ftc/
│       ├── __init__.py
│       ├── config.py
│       ├── cli.py
│       ├── dynamics/
│       │   ├── __init__.py
│       │   ├── quadrotor_model.py
│       │   └── linearization.py
│       ├── control/
│       │   ├── __init__.py
│       │   ├── pid_cascade.py
│       │   ├── lbfgsb_tuning.py
│       │   └── fault_tolerant_controller.py
│       ├── faults/
│       │   ├── __init__.py
│       │   ├── sensor_faults.py
│       │   ├── actuator_faults.py
│       │   └── adversarial_attacks.py
│       ├── sensing/
│       │   ├── __init__.py
│       │   ├── imu_model.py
│       │   ├── lidar_model.py
│       │   └── fusion_reliability.py
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── scenarios.py
│       │   ├── metrics.py
│       │   └── runner.py
│       └── viz/
│           ├── __init__.py
│           ├── plots.py
│           └── animations.py
├── tests/
│   ├── __init__.py
│   ├── test_dynamics.py
│   ├── test_controller.py
│   ├── test_faults.py
│   └── test_sensing.py
├── examples/
│   ├── simple_hover.py
│   ├── fault_injection_demo.py
│   └── adversarial_attack_demo.py
└── docs/
    └── index.md

## Installation

```bash
pip install -e .
```

## Usage

See `examples/simple_hover.py` and `examples/fault_injection_demo.py` for end-to-end scripts.
