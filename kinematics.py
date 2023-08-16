import numpy as np


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


def theta_limits(wtheta, atheta):
    # waist check

    # arm check
    pass

# iCub
def forward_kinematics(wtheta,
                       atheta,
                       arm = 'right',
                       radians = True,
                       return_joint_coordinates = False):

    if not radians:
        wtheta = np.radians(wtheta)
        atheta = np.radians(atheta)

    if arm == 'right':
        const = -1
    elif arm == 'left':
        const = 1
    else:
        raise ValueError('Arm should be either the right arm or the left arm')

    # ---------------------- WAIST ----------------------------
    # wtheta = [torso_pitch, torso_roll, torso_yaw]
    A_0 = DH_matrix(a=32, d=0, alpha=np.pi/2, theta=wtheta[0])
    A_1 = DH_matrix(a=0, d=-5.5, alpha=-np.pi/2, theta=wtheta[1] - np.pi/2)
    A_2 = DH_matrix(a=const * 23.3647, d=-143.3, alpha=const * -np.pi/2, theta=wtheta[2] + const * 105 * np.pi/180)

    A_01 = A_0 @ A_1
    A_12 = A_01 @ A_2

    # ---------------------- ARM ----------------------------
    # atheta = [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, forearm]
    # shoulder
    A_3 = DH_matrix(a=0, d=const * 107.74, alpha=const * -np.pi/2, theta=atheta[0] + const * np.pi/2)
    A_4 = DH_matrix(a=0, d=0, alpha=np.pi/2, theta=atheta[1] - np.pi/2)
    A_5 = DH_matrix(a=0, d=const*152.28, alpha=-np.pi/2, theta=atheta[2] - 15*np.pi/180 + const*np.pi/2)
    # elbow
    A_6 = DH_matrix(a=-15, d=0, alpha=np.pi/2, theta=atheta[3])
    #forearm
    A_7 = DH_matrix(a=0, d=137.3, alpha=np.pi/2, theta=atheta[4] - np.pi/2)

    A_23 = A_12 @ A_3
    A_34 = A_23 @ A_4
    A_45 = A_34 @ A_5
    A_56 = A_45 @ A_6
    A_67 = A_56 @ A_7

    if return_joint_coordinates:
        return A_12[:3, 3], A_23[:3, 3], A_56[:3, 3], A_67[:3, 3]
    else:
        return A_67[:3, 3]

if __name__ == '__main__':
    print(forward_kinematics(wtheta = [0,0,0], atheta = np.radians([5, 120, 90, 90, 20])))