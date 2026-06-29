import numpy as np
from uav_ftc.config import PIDGains, SimulationParams
from uav_ftc.experiments.scenarios import FaultScenario
from uav_ftc.experiments.runner import run_single_sim
from uav_ftc.viz.plots import plot_trajectory
from uav_ftc.viz.animations import animate_xy_traj


def main():
    gains = PIDGains()
    sim = SimulationParams(t_final=10.0, dt=0.005)

    scenario = FaultScenario(
        name="sensor_actuator_faults",
        sensor_fault_config={"dropout_prob": 0.15, "bias_mean": 0.0, "bias_std": 0.05, "stuck_prob": 0.05},
        actuator_fault_config={"effectiveness": np.array([1.0, 0.8, 0.6, 0.9]), "disturbance_std": 0.1},
        adversarial_attack_config={"strength": 0.0},
    )

    result = run_single_sim(scenario, gains, sim)
    time = result["time"]
    p_traj = result["p_traj"]
    p_ref = np.stack([np.zeros_like(time), np.zeros_like(time), np.ones_like(time)], axis=1)

    fig = plot_trajectory(time, p_traj, p_ref)
    fig.savefig("fault_injection_trajectory.png", dpi=200)

    animate_xy_traj(time, p_traj, filename="fault_injection_xy.gif")

    print("Metrics:", result["metrics"])


if __name__ == "__main__":
    main()