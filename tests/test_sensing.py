import numpy as np
from uav_ftc.faults.sensor_faults import SensorFaultModel
from uav_ftc.sensing.fusion_reliability import SensorFusionReliability
  

def test_fusion_weights_sum_to_one():
    imu_fault = SensorFaultModel(dropout_prob=0.1)
    lidar_fault = SensorFaultModel(dropout_prob=0.0)
    fusion = SensorFusionReliability(imu_fault, lidar_fault)

    imu_pose = np.array([0.0, 0.0, 0.0])
    lidar_pose = np.array([1.0, 0.0, 0.0])
    fused_pose, weights = fusion.fuse(imu_pose, lidar_pose)

    assert fused_pose.shape == (3,)
    w_sum = weights["imu"] + weights["lidar"]
    assert abs(w_sum - 1.0) < 1e-6
