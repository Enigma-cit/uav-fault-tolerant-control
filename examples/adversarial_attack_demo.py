import numpy as np
from uav_ftc.config import PIDGains, SimulationParams
from uav_ftc.experiments.scenarios import FaultScenario
from uav_ftc.experiments.runner import run_single_sim
from uav_ftc.viz.plots import plot_trajectory


def main():
    gains = PIDGains()
    sim = SimulationParams(t_final=10.0, dt=0.005)

    scenario = FaultScenario(
        name="adversarial_lidar_attack",
        sensor_fault_config={"dropout_prob": 0.0, "bias_mean": 0.0, "bias_std": 0.02, "stuck_prob": 0.0},
        actuator_fault_config={"effectiveness": None, "disturbance_std": 0.0},
        adversarial_attack_config={"strength": 0.7},
    )

    result = run_single_sim(scenario, gains, sim)
    time = result["time"]
    p_traj = result["p_traj"]
    p_ref = np.stack([np.zeros_like(time), np.zeros_like(time), np.ones_like(time)], axis=1)

    fig = plot_trajectory(time, p_traj, p_ref)
    fig.savefig("adversarial_attack_trajectory.png", dpi=200)

    print("Metrics:", result["metrics"])


if __name__ == "__main__":
    main()