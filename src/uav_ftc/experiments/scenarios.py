from dataclasses import dataclass


@dataclass
class FaultScenario:
    name: str
    sensor_fault_config: dict
    actuator_fault_config: dict
    adversarial_attack_config: dict
