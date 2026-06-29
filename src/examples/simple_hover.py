from uav_ftc.config import PIDGains, SimulationParams
from uav_ftc.experiments.scenarios import FaultScenario
from uav_ftc.experiments.runner import run_single_sim
from uav_ftc.viz.plots import plot_trajectory


def main():
    gains = PIDGains()
    sim = SimulationParams(t_final=8.0, dt=0.005)

    scenario = FaultScenario(
        name="no_fault_hover",
        sensor_fault_config={"dropout_prob": 0.0, "bias_mean": 0.0, "bias_std": 0.0, "stuck_prob": 0.0},
        actuator_fault_config={"effectiveness": None, "disturbance_std": 0.0},
        adversarial_attack_config={"strength": 0.0},
    )

    result = run_single_sim(scenario, gains, sim)
    fig = plot_trajectory(result["time"], result["p_traj"], result["time"][:, None] * 0.0 + [0.0, 0.0, 1.0])
    fig.savefig("hover_trajectory.png", dpi=200)


if __name__ == "__main__":
    main()
