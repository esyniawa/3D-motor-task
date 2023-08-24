import numpy as np

params = {
    'lower_waist_joint_limits' : np.radians([-22, -39, -59]),
    'upper_waist_joint_limits' : np.radians([84, 39, 59]),

    'lower_arm_joint_limits' : np.radians([-95, 0, -37, 5.5, -50]),
    'upper_arm_joint_limits' : np.radians([5, 160.8, 100, 106, 50]),

    'lower_head_joint_limits': np.radians([-20, -50, -30, -15, -30]),
    'upper_head_joint_limits': np.radians([20, 50, 30, 15, 30]),
    # initial waist position
    'waist_position': np.radians([0, 0, 0]),
    # all points are depending on the waist position
    'starting_angles': np.radians([5, 20, 0, 90, 20]),
    'end_point': np.array([-61.0, 171.0, -162.0])
}

