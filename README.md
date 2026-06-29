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


## Installation

```bash
pip install -e .
```

## Usage

See `examples/simple_hover.py` and `examples/fault_injection_demo.py` for end-to-end scripts.
