"""
Command-line interface for the uav_ftc package.

Examples
--------
# Run a single predefined scenario
python -m uav_ftc.cli run --scenario no_fault_hover

# Run a sweep over multiple scenarios
python -m uav_ftc.cli sweep

# Run L-BFGS-B-based PID tuning
python -m uav_ftc.cli tune
"""

import argparse
from uav_ftc.config import PIDGains, SimulationParams
from uav_ftc.experiments.scenarios import FaultScenario
from uav_ftc.experiments.runner import run_single_sim, run_fault_sweep
from uav_ftc.control.lbfgsb_tuning import PIDTunerLBFGSB
from uav_ftc.experiments.metrics import compute_metrics


def _default_scenarios():
    return [
        FaultScenario(
            name="no_fault_hover",
            sensor_fault_config={"dropout_prob": 0.0, "bias_mean": 0.0, "bias_std": 0.0, "stuck_prob": 0.0},
            actuator_fault_config={"effectiveness": None, "disturbance_std": 0.0},
            adversarial_attack_config={"strength": 0.0},
        ),
        FaultScenario(
            name="sensor_dropout",
            sensor_fault_config={"dropout_prob": 0.1, "bias_mean": 0.0, "bias_std": 0.0, "stuck_prob": 0.0},
            actuator_fault_config={"effectiveness": None, "disturbance_std": 0.0},
            adversarial_attack_config={"strength": 0.0},
        ),
        FaultScenario(
            name="actuator_loss",
            sensor_fault_config={"dropout_prob": 0.0, "bias_mean": 0.0, "bias_std": 0.0, "stuck_prob": 0.0},
            actuator_fault_config={"effectiveness": 0.7, "disturbance_std": 0.05},
            adversarial_attack_config={"strength": 0.0},
        ),
    ]


def cmd_run(args: argparse.Namespace) -> None:
    gains = PIDGains()
    sim = SimulationParams(t_final=args.t_final, dt=args.dt)

    scenarios = {s.name: s for s in _default_scenarios()}
    scenario = scenarios.get(args.scenario)
    if scenario is None:
        raise ValueError(f"Unknown scenario {args.scenario}")

    result = run_single_sim(scenario, gains, sim)
    m = result["metrics"]
    print(f"Scenario: {scenario.name}")
    print(f"  RMS error      : {m['rms_error']:.3f}")
    print(f"  Overshoot (z)  : {m['overshoot']:.3f}")
    print(f"  Settling time  : {m['settling_time']:.3f}")
    print(f"  Control effort : {m['control_effort']:.3f}")


def cmd_sweep(args: argparse.Namespace) -> None:
    gains = PIDGains()
    sim = SimulationParams(t_final=args.t_final, dt=args.dt)

    scenarios = _default_scenarios()
    results = run_fault_sweep(scenarios, gains, sim)

    for name, res in results.items():
        m = res["metrics"]
        print(f"[{name}] rms={m['rms_error']:.3f}, overshoot={m['overshoot']:.3f}, "
              f"settling={m['settling_time']:.3f}, effort={m['control_effort']:.3f}")


def cmd_tune(args: argparse.Namespace) -> None:
    sim = SimulationParams(t_final=args.t_final, dt=args.dt)
    scenarios = _default_scenarios()

    def cost_fn(gains: PIDGains) -> float:
        # Average RMS tracking error over scenarios
        results = run_fault_sweep(scenarios, gains, sim)
        rms_errors = [res["metrics"]["rms_error"] for res in results.values()]
        return float(sum(rms_errors) / len(rms_errors))

    tuner = PIDTunerLBFGSB(cost_fn)
    initial = PIDGains()
    result = tuner.tune(initial_gains=initial, bounds=None)

    print("Tuning finished")
    print(f"  success: {result.success}, message: {result.message}")
    print(f"  cost   : {result.cost:.3f}")
    print("  tuned gains:")
    print("    kp_xyz:", result.gains.kp_xyz)
    print("    kd_xyz:", result.gains.kd_xyz)
    print("    kp_att:", result.gains.kp_att)
    print("    kd_att:", result.gains.kd_att)
    print("    ki_att:", result.gains.ki_att)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fault-tolerant quadrotor control experiment runner."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Run a single scenario.")
    p_run.add_argument("--scenario", type=str, default="no_fault_hover")
    p_run.add_argument("--t_final", type=float, default=8.0)
    p_run.add_argument("--dt", type=float, default=0.005)
    p_run.set_defaults(func=cmd_run)

    p_sweep = subparsers.add_parser("sweep", help="Run a sweep over scenarios.")
    p_sweep.add_argument("--t_final", type=float, default=8.0)
    p_sweep.add_argument("--dt", type=float, default=0.005)
    p_sweep.set_defaults(func=cmd_sweep)

    p_tune = subparsers.add_parser("tune", help="Tune PID via L-BFGS-B.")
    p_tune.add_argument("--t_final", type=float, default=6.0)
    p_tune.add_argument("--dt", type=float, default=0.005)
    p_tune.set_defaults(func=cmd_tune)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()