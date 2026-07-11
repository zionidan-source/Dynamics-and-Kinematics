"""
ur5_project/kinematics.py
=========================
Forward kinematics for the UR5 robotic arm using Denavit-Hartenberg parameters.

This module is the foundation for the rest of the project. Everything else
(workspace, IK, Jacobian, statics, trajectory) builds on the FK defined here.

Author: Daniel, Itai & Ido
Course: Kinematics and Dynamics of Robots, Ben-Gurion University, 2026
"""

import numpy as np


# ============================================================================
# UR5 DH parameters (standard convention)
# ============================================================================
# Each row corresponds to one joint, with columns: [a, alpha, d, theta_offset]
# theta_offset is added to the joint variable theta_i. For UR5 this is 0
# for all joints in the standard DH convention.
#
# Source: Universal Robots official UR5 specifications
#         https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
#
# Distances are in meters, angles in radians.

UR5_DH_PARAMS = np.array([
    # a            alpha       d           theta_offset
    [0.0,         np.pi/2,    0.089159,   0.0],   # Joint 1: shoulder_pan
    [-0.425,      0.0,        0.0,        0.0],   # Joint 2: shoulder_lift
    [-0.39225,    0.0,        0.0,        0.0],   # Joint 3: elbow
    [0.0,         np.pi/2,    0.10915,    0.0],   # Joint 4: wrist_1
    [0.0,        -np.pi/2,    0.09465,    0.0],   # Joint 5: wrist_2
    [0.0,         0.0,        0.08230,    0.0],   # Joint 6: wrist_3
])

# Joint names in physical order (base to end-effector).
# Note: /joint_states publishes them in alphabetical order, so we must
# reorder when reading from ROS2.
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


# ============================================================================
# Single-link DH transformation matrix
# ============================================================================
def dh_transform(a, alpha, d, theta):
    """
    Build the 4x4 homogeneous transformation matrix for one DH link.

    This is the matrix that takes you from coordinate frame {i-1}
    to coordinate frame {i}, given the four DH parameters of link i.

    The matrix is the product of four elementary transformations:
        T = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)

    Parameters
    ----------
    a : float
        Link length (distance along x_i between z_{i-1} and z_i).
    alpha : float
        Link twist (angle around x_i from z_{i-1} to z_i), in radians.
    d : float
        Link offset (distance along z_{i-1} from x_{i-1} to x_i).
    theta : float
        Joint angle (angle around z_{i-1} from x_{i-1} to x_i), in radians.

    Returns
    -------
    T : (4, 4) np.ndarray
        Homogeneous transformation matrix.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    T = np.array([
        [ct,    -st * ca,    st * sa,    a * ct],
        [st,     ct * ca,   -ct * sa,    a * st],
        [0.0,    sa,         ca,         d     ],
        [0.0,    0.0,        0.0,        1.0   ],
    ])
    return T


# ============================================================================
# Forward kinematics: full chain
# ============================================================================
# ============================================================================
# Base frame correction
# ============================================================================
# The UR5's URDF defines `base_link` with a 180-degree rotation around z
# compared to the conventional DH frame {0}. To match Gazebo/RViz output,
# we pre-multiply by this correction. With T_BASE_CORRECTION = I (identity),
# our FK matches the "DH-only" derivation used in textbooks. With the
# correction applied, our FK matches the URDF/ROS frame "base_link".
T_BASE_CORRECTION = np.array([
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
])


def forward_kinematics(joint_angles, dh_params=UR5_DH_PARAMS,
                       return_all=False, apply_base_correction=True):
    """
    Compute the end-effector pose given the 6 joint angles.

    Multiplies the 6 individual DH transformation matrices to get the
    homogeneous transform from the base frame {0} to the end-effector
    frame {6}.

    Parameters
    ----------
    joint_angles : array-like of length 6
        Joint angles [theta_1, ..., theta_6] in radians, in physical order
        (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3).
    dh_params : (6, 4) np.ndarray, optional
        DH parameter table. Defaults to UR5_DH_PARAMS.
    return_all : bool, optional
        If True, also return the list of cumulative transforms
        [T_0_1, T_0_2, ..., T_0_6]. Useful for plotting the stick diagram
        and computing the Jacobian later.
    apply_base_correction : bool, optional
        If True (default), pre-multiply by T_BASE_CORRECTION so the result
        matches ROS frame `base_link` (matches Gazebo/RViz). If False, the
        result is in the pure DH frame {0} (useful for textbook comparison).

    Returns
    -------
    T_base_ee : (4, 4) np.ndarray
        Homogeneous transform from base_link frame to wrist_3_link frame.
    transforms : list of (4, 4) np.ndarray, optional
        Only returned if return_all=True. Cumulative transforms from
        base frame to each joint frame.
    """
    if len(joint_angles) != 6:
        raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")

    T_cumulative = T_BASE_CORRECTION.copy() if apply_base_correction else np.eye(4)
    transforms = []

    for i in range(6):
        a, alpha, d, theta_offset = dh_params[i]
        theta = joint_angles[i] + theta_offset
        T_i = dh_transform(a, alpha, d, theta)
        T_cumulative = T_cumulative @ T_i
        transforms.append(T_cumulative.copy())

    if return_all:
        return T_cumulative, transforms
    return T_cumulative


# ============================================================================
# Helpers: extract position and orientation from a homogeneous transform
# ============================================================================
def position_from_transform(T):
    """Return the 3x1 position vector from a 4x4 homogeneous transform."""
    return T[:3, 3]


def rotation_from_transform(T):
    """Return the 3x3 rotation matrix from a 4x4 homogeneous transform."""
    return T[:3, :3]


def rotation_to_rpy(R):
    """
    Convert a 3x3 rotation matrix to Roll-Pitch-Yaw Euler angles (XYZ convention).

    Returns
    -------
    rpy : (3,) np.ndarray
        [roll, pitch, yaw] in radians, where:
        - roll is rotation about the world X axis
        - pitch is rotation about the world Y axis
        - yaw is rotation about the world Z axis

    Note: Singular at pitch = +/- 90 degrees.
    """
    pitch = np.arcsin(-R[2, 0])

    # Check for gimbal lock
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: roll and yaw are coupled, set roll = 0
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    return np.array([roll, pitch, yaw])


# ============================================================================
# Sanity check when run as a script
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("UR5 Forward Kinematics Sanity Check")
    print("=" * 60)

    # Test 1: All zeros (the "zero" configuration in DH)
    print("\nTest 1: All joint angles = 0")
    angles = [0, 0, 0, 0, 0, 0]
    T = forward_kinematics(angles)
    pos = position_from_transform(T)
    rpy = rotation_to_rpy(rotation_from_transform(T))
    print(f"  Position [x, y, z] = [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m")
    print(f"  Orientation [R, P, Y] = [{np.degrees(rpy[0]):+.2f}, "
          f"{np.degrees(rpy[1]):+.2f}, {np.degrees(rpy[2]):+.2f}] deg")

    # Test 2: The "home" pose we saw in /joint_states
    print("\nTest 2: Home pose (shoulder_lift = -90, wrist_1 = -90, others = 0)")
    angles = [0, -np.pi/2, 0, -np.pi/2, 0, 0]
    T = forward_kinematics(angles)
    pos = position_from_transform(T)
    rpy = rotation_to_rpy(rotation_from_transform(T))
    print(f"  Position [x, y, z] = [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m")
    print(f"  Orientation [R, P, Y] = [{np.degrees(rpy[0]):+.2f}, "
          f"{np.degrees(rpy[1]):+.2f}, {np.degrees(rpy[2]):+.2f}] deg")

    # Test 3: Show all intermediate frames
    print("\nTest 3: Intermediate joint positions for the home pose")
    _, all_T = forward_kinematics(angles, return_all=True)
    for i, Ti in enumerate(all_T):
        p = position_from_transform(Ti)
        print(f"  Frame {i+1}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")
