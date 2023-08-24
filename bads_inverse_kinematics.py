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

def error_function_kinematic(atheta,
                             end_point = params['end_point'],
                             wtheta = params['waist_position'],
                             arm='right'):

    error = end_point - forward_kinematics_arm(wtheta, atheta, arm)
    return np.linalg.norm(error)


def bads_inverse_kinematic(starting_angles, rad = True):
    if not rad:
        starting_angles = np.radians(starting_angles)

    target = error_function_kinematic

    print(starting_angles)
    bads = BADS(target, starting_angles,
                plausible_lower_bounds=params['lower_arm_joint_limits'],
                plausible_upper_bounds=params['upper_arm_joint_limits'])

    optimize_result = bads.optimize()
    joint_angles = optimize_result['x']

    return joint_angles


if __name__ == '__main__':
    with suppress_stdout():
        res = bads_inverse_kinematic(starting_angles=params['starting_angles'])

    print(np.degrees(res))