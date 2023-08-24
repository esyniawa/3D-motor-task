import numpy as np

from parameters import params

# Calculate the Transformation Matrix A according to the DH convention
def DH_matrix(a, d, alpha, theta, rad=True):

    if not rad:
        alpha = np.radians(alpha)
        theta = np.radians(theta)

    A = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                  [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                  [0, np.sin(alpha), np.cos(alpha), d],
                  [0, 0, 0, 1]])

    return A


def check_theta_limits_arm(wtheta, atheta, rad=True):
    if not rad:
        wtheta = np.radians(wtheta)
        atheta = np.radians(atheta)

    # waist check
    waist_ok = all(wtheta >= params['lower_waist_joint_limits']) & all(wtheta <= params['upper_waist_joint_limits'])
    # arm check
    arm_ok = all(atheta >= params['lower_arm_joint_limits']) & all(atheta <= params['upper_arm_joint_limits'])

    return waist_ok & arm_ok


def check_theta_limits_head(wtheta, htheta, rad=True):
    if not rad:
        wtheta = np.radians(wtheta)
        htheta = np.radians(htheta)

    # waist check
    # TODO: waist limits are slightly different with the head
    waist_ok = all(wtheta >= params['lower_waist_joint_limits']) & all(wtheta <= params['upper_waist_joint_limits'])
    # head check
    head_ok = all(htheta >= params['lower_head_joint_limits']) & all(htheta <= params['upper_head_joint_limits'])

    return waist_ok & head_ok


# iCub forward kinematic for the arms. All DH-parameters originate from
# https://icub-tech-iit.github.io/documentation/icub_kinematics/icub-forward-kinematics/icub-forward-kinematics-arms/.
def forward_kinematics_arm(wtheta,
                           atheta,
                           arm = 'right',
                           radians = True,
                           return_joint_coordinates = False):

    if not radians:
        wtheta = np.radians(wtheta)
        atheta = np.radians(atheta)

    if not check_theta_limits_arm(wtheta, atheta):
        pass
    else:
        if arm == 'right':
            const = -1
        elif arm == 'left':
            const = 1
        else:
            raise ValueError('Arm should be either the right or the left arm')

        # ---------------------- WAIST ----------------------------
        # wtheta = [torso_pitch, torso_roll, torso_yaw]
        A_0 = DH_matrix(a=32, d=0, alpha=np.pi/2, theta=wtheta[0])
        A_1 = DH_matrix(a=0, d=-5.5, alpha=np.pi/2, theta=wtheta[1] - np.pi/2)
        A_2 = DH_matrix(a=const * 23.3647, d=-143.3, alpha=const * -np.pi/2, theta=wtheta[2] + const * 105 * np.pi/180)

        A_01 = A_0 @ A_1
        A_12 = A_01 @ A_2

        # ---------------------- ARM ----------------------------
        # atheta = [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, forearm]
        # shoulder
        A_3 = DH_matrix(a=0, d=const * 107.74, alpha=const * -np.pi/2, theta=atheta[0] + const * np.pi/2)
        A_4 = DH_matrix(a=0, d=0, alpha=const * np.pi/2, theta=atheta[1] - np.pi/2)
        A_5 = DH_matrix(a=const * 15, d=const * 152.28, alpha=-np.pi/2, theta=atheta[2] - 15*np.pi/180 + const*np.pi/2)
        # elbow
        A_6 = DH_matrix(a=const * -15, d=0, alpha=np.pi/2, theta=atheta[3])
        #forearm
        A_7 = DH_matrix(a=0, d=const * 137.3, alpha=np.pi/2, theta=atheta[4] - np.pi/2)

        A_23 = A_12 @ A_3
        A_34 = A_23 @ A_4
        A_45 = A_34 @ A_5
        A_56 = A_45 @ A_6
        A_67 = A_56 @ A_7

        if return_joint_coordinates:
            return np.column_stack((A_0[:3, 3], A_12[:3, 3], A_23[:3, 3], A_56[:3, 3], A_67[:3, 3]))
        else:
            return A_67[:3, 3]


def forward_kinematics_head(wtheta,
                            htheta,
                            radians = True):

    if not radians:
        wtheta = np.radians(wtheta)
        htheta = np.radians(htheta)

    if not check_theta_limits_head(wtheta, htheta):
        raise ValueError('Check joint constraints')

    else:
        # ---------------------- WAIST ----------------------------
        # wtheta = [torso_pitch, torso_roll, torso_yaw]
        A_0 = DH_matrix(a=32, d=0, alpha=np.pi/2, theta=wtheta[0])
        A_1 = DH_matrix(a=0, d=-5.5, alpha=np.pi/2, theta=wtheta[1] - np.pi/2)
        A_2 = DH_matrix(a=2.31, d=-193.3, alpha=-np.pi/2, theta=wtheta[2] - np.pi/2)

        A_01 = A_0 @ A_1
        A_12 = A_01 @ A_2
        # ---------------------- HEAD ----------------------------
        # neck + head
        A_3 = DH_matrix(a=33, d=0, alpha=np.pi/2, theta=htheta[0] + np.pi/2)
        A_4 = DH_matrix(a=0, d=1, alpha=-np.pi/2, theta=htheta[1] - np.pi/2)
        A_5 = DH_matrix(a=-54, d=82.5, alpha=-np.pi/2, theta=htheta[2] + np.pi/2)

        A_23 = A_12 @ A_3
        A_34 = A_23 @ A_4
        A_45 = A_34 @ A_5

        # eyes
        A_6_left = DH_matrix(a=0, d=34, alpha=-np.pi/2, theta=htheta[3])
        A_7_left = DH_matrix(a=0, d=0, alpha=np.pi/2, theta=htheta[4] - np.pi/2)

        A_6_right = DH_matrix(a=0, d=-34, alpha=-np.pi/2, theta=htheta[3])
        A_7_right = DH_matrix(a=0, d=0, alpha=np.pi/2, theta=htheta[4] - np.pi/2)

        # left eye
        A_56_left = A_45 @ A_6_left
        A_67_left = A_56_left @ A_7_left
        # right eye
        A_56_right = A_45 @ A_6_right
        A_67_right = A_56_right @ A_7_right

        return np.column_stack((A_67_left[:3, 3], A_67_right[:3, 3]))


def plot_arms(wtheta, atheta_left, atheta_right, rad=True):
    if not rad:
        wtheta = np.radians(wtheta)
        atheta_left = np.radians(atheta_left)
        atheta_right = np.radians(atheta_right)

    pos_left_arm = forward_kinematics_arm(wtheta, atheta_left, arm='left', return_joint_coordinates=True)
    pos_right_arm = forward_kinematics_arm(wtheta, atheta_right, arm='right', return_joint_coordinates=True)

    # 3D plot
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(pos_left_arm[0,:], pos_left_arm[1,:], pos_left_arm[2,:])
    ax.plot(pos_right_arm[0,:], pos_right_arm[1,:], pos_right_arm[2,:])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def plot_upper_body(wtheta, atheta_left, atheta_right, htheta, rad=True):
    if not rad:
        wtheta = np.radians(wtheta)
        atheta_left = np.radians(atheta_left)
        atheta_right = np.radians(atheta_right)

    pos_left_arm = forward_kinematics_arm(wtheta, atheta_left, arm='left', return_joint_coordinates=True)
    pos_right_arm = forward_kinematics_arm(wtheta, atheta_right, arm='right', return_joint_coordinates=True)
    pos_eyes = forward_kinematics_head(wtheta, htheta)

    # 3D plot
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection='3d')
    # arm positions
    ax.plot(pos_left_arm[0,:],pos_left_arm[1,:], pos_left_arm[2,:])
    ax.plot(pos_right_arm[0,:],pos_right_arm[1,:], pos_right_arm[2,:])
    # eye positions
    ax.scatter(pos_eyes[0, 0], pos_eyes[1, 0], pos_eyes[2, 0])
    ax.scatter(pos_eyes[0, 1], pos_eyes[1, 1], pos_eyes[2, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':
    print(forward_kinematics_arm(wtheta=[0, 0, 0], atheta=np.radians([-25, 20, 30, 12, 30])))

    plot_upper_body(wtheta=[0, 0, 0],
                    atheta_left=np.radians([-25, 20, 30, 12, 30]),
                    atheta_right=np.radians([5, 20, 0, 90, 20]),
                    htheta=np.radians([0, 0, 20, 0, 0]))

