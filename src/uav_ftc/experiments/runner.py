import numpy as np

from uav_ftc.config import QuadrotorParams, SimulationParams, PIDGains
from uav_ftc.dynamics.quadrotor_model import QuadrotorDynamics
from uav_ftc.control.pid_cascade import CascadedPIDController
from uav_ftc.faults.sensor_faults import SensorFaultModel
from uav_ftc.faults.actuator_faults import ActuatorFaultModel
from uav_ftc.faults.adversarial_attacks import AdversarialAttack
from uav_ftc.sensing.fusion_reliability import SensorFusionReliability
from uav_ftc.experiments.metrics import compute_metrics


def run_single_sim(scenario, gains: PIDGains, sim_params: SimulationParams) -> dict:
    quad = QuadrotorDynamics(QuadrotorParams(), sim_params)
    controller = CascadedPIDController(gains, QuadrotorParams())

    sensor_fault = SensorFaultModel(**scenario.sensor_fault_config)
    actuator_fault = ActuatorFaultModel(**scenario.actuator_fault_config)
    adversary = AdversarialAttack(**scenario.adversarial_attack_config)

    fusion = SensorFusionReliability(sensor_fault, sensor_fault)

    T = int(sim_params.t_final / sim_params.dt)
    time = np.arange(T) * sim_params.dt

    p_ref = np.stack([np.zeros(T), np.zeros(T), np.ones(T)], axis=1)  # hover at z=1
    v_ref = np.zeros_like(p_ref)

    p_traj = np.zeros_like(p_ref)
    u_traj = np.zeros((T, 4))  # thrust + torques

    for k in range(T):
        # True state from dynamics
        p = quad.p.copy()
        v = quad.v.copy()
        eta = quad.eta.copy()
        omega = quad.omega.copy()

        # "True" pose from IMU & LiDAR (abstracted)
        imu_pose = p + 0.0  # placeholder
        lidar_pose = p + 0.0

        # Apply adversarial attack to LiDAR ranges (abstracted)
        lidar_pose = lidar_pose  # could be perturbed by adversary.apply_to_ranges(...)

        fused_pose, _ = fusion.fuse(imu_pose, lidar_pose)

        # Controller using fused pose (for now, use fused position and true attitude)
        u = controller.compute_control(
            p_ref[k], v_ref[k],
            eta_ref=np.array([0.0, 0.0, 0.0]),
            p=fused_pose, v=v,
            eta=eta, omega=omega,
            dt=sim_params.dt,
        )

        # Map actuator faults on rotor speeds (we embed torque-level faults here for simplicity)
        # In a more detailed implementation, you'd convert u to rotor speeds and then apply faults.
        rotor_sq = np.ones(4) * (u[0] / 4.0)  # naive allocation
        faulty_rotor_sq = actuator_fault.apply(rotor_sq)

        quad.step(faulty_rotor_sq, rotor_input=True)

        p_traj[k] = p
        u_traj[k, :] = u

    metrics = compute_metrics(time, p_traj, p_ref, u_traj)
    return {
        "time": time,
        "p_traj": p_traj,
        "u_traj": u_traj,
        "metrics": metrics,
    }


def run_fault_sweep(scenarios, gains: PIDGains, sim_params: SimulationParams) -> dict:
    results = {}
    for scenario in scenarios:
        results[scenario.name] = run_single_sim(scenario, gains, sim_params)
    return results
