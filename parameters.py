import numpy as np

params = {
    'lower_waist_joint_limits' : np.radians([-22, -39, -59]),
    'upper_waist_joint_limits' : np.radians([84, 39, 59]),
    'lower_arm_joint_limits' : np.radians([-95, 0, -37, 5.5, -50]),
    'upper_arm_joint_limits' : np.radians([5, 160.8, 100, 106, 50]),

    'waist_position': np.radians([0, 0, 0]),
    # all points are depending on the waist position
    'starting_angles' : np.radians([5, 120, 90, 90, 20]),
    'end_point' : np.array([-154.2, 10.0, 78.0])
}

