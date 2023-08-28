import numpy as np
import sys, os

from pybads.bads import BADS
from contextlib import contextmanager

from forward_kinematics import forward_kinematics_arm
from parameters import params

# supress standard output
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def bads_inverse_kinematic(end_attractor,
                           starting_joint_angles,
                           waist_angles,
                           moving_arm='right',
                           rad = True):
    if not rad:
        starting_joint_angles = np.radians(starting_joint_angles)
        waist_angles = np.radians(waist_angles)
    # Error function for optimization
    def error_function_kinematic(atheta,
                                 end_point=end_attractor,
                                 wtheta=waist_angles,
                                 arm=moving_arm):

        # distance between current and end position
        error = np.linalg.norm(end_point - forward_kinematics_arm(wtheta, atheta, arm))
        # minimal movement -> minimal change in joint angles
        error += np.linalg.norm(starting_joint_angles-atheta)
        return error

    target = error_function_kinematic

    bads = BADS(target, starting_joint_angles,
                plausible_lower_bounds=params['lower_arm_joint_limits'],
                plausible_upper_bounds=params['upper_arm_joint_limits'])

    optimize_result = bads.optimize()
    joint_angles = optimize_result['x']

    return joint_angles


if __name__ == '__main__':
    with suppress_stdout():
        res = bads_inverse_kinematic(starting_joint_angles=params['starting_angles'],
                                     end_attractor=params['end_point'],
                                     waist_angles=params['waist_position'])

    print(np.degrees(res))
    print(forward_kinematics_arm(params['waist_position'], res))