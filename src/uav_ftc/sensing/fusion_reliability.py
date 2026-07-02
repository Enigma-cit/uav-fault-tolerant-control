import numpy as np
 

class SensorFusionReliability:
    """
    LiDAR–IMU fusion reliability policy:

    - Computes pose estimates from IMU integration and LiDAR (e.g., scan-matching).
      Here we abstract these as two candidate poses.
    - Computes a reliability score per sensor based on innovation magnitude,
      consistency checks, and known fault statistics.
    - Outputs a fused pose and sensor weights for downstream control.
    """

    def __init__(self, imu_fault_model, lidar_fault_model):
        self.imu_fault_model = imu_fault_model
        self.lidar_fault_model = lidar_fault_model

    def fuse(self, imu_pose: np.ndarray, lidar_pose: np.ndarray) -> tuple[np.ndarray, dict]:
        innovation = lidar_pose - imu_pose
        innovation_norm = np.linalg.norm(innovation)

        # Simple heuristic reliability scores
        imu_rel = np.exp(-0.5 * innovation_norm)
        lidar_rel = np.exp(-0.5 * innovation_norm)  # symmetric here; can be asymmetric if needed

        # Adjust reliability downwards if sensor fault models indicate compromise
        imu_rel *= (1.0 - self.imu_fault_model.dropout_prob)
        lidar_rel *= (1.0 - self.lidar_fault_model.dropout_prob)

        total = imu_rel + lidar_rel + 1e-9
        w_imu = imu_rel / total
        w_lidar = lidar_rel / total

        fused_pose = w_imu * imu_pose + w_lidar * lidar_pose
        weights = {"imu": w_imu, "lidar": w_lidar}
        return fused_pose, weights
