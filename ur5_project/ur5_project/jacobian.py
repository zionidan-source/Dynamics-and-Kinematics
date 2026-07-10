#!/usr/bin/env python3
"""
ur5_project/jacobian.py
=======================
Geometric Jacobian and static joint torques for the UR5
(assignment sections 4.1.7 and 4.1.8).

For a 6-DOF serial arm with all revolute joints, the geometric Jacobian
J(q) is the 6x6 matrix that maps joint velocities to the linear and
angular velocity of the end-effector, expressed in the base frame:

    [ v ]                   [ z_{i-1} x (o_n - o_{i-1}) ]
    [   ] = J(q) * q_dot,   J_i = [                            ]
    [ w ]                   [           z_{i-1}              ]

where z_{i-1} and o_{i-1} are the z-axis and origin of frame {i-1}
expressed in the base frame, and o_n is the origin of the end-effector
frame, also in the base frame. All frames here are expressed in
`base_link` (the ROS/URDF base frame), because forward_kinematics()
returns intermediate frames with T_BASE_CORRECTION already applied.

The Jacobian is verified by finite-difference comparison against the
Gazebo-verified forward_kinematics module (worst error ~1e-8).

Static torques required to hold a payload of mass M at the end-effector
follow from the principle of virtual work:

    tau_motor = J^T(q) * F_ext

where F_ext = [0, 0, +M*g, 0, 0, 0]^T is the wrench the *motors* must
apply at the end-effector to support the load against gravity.
Equivalently, gravity applies -F_ext to the EE and the motors balance it.

Author: Daniel & Itay
Course: Kinematics and Dynamics of Robots, Ben-Gurion University, 2026
"""

import numpy as np

from ur5_project.kinematics import forward_kinematics, T_BASE_CORRECTION


# ============================================================================
# Geometric Jacobian
# ============================================================================
def compute_jacobian(joint_angles):
    """
    Compute the 6x6 geometric Jacobian of the UR5 at the given configuration.

    The Jacobian relates joint-space velocities to end-effector twist in
    the base_link frame:

        [v_x, v_y, v_z, w_x, w_y, w_z]^T = J(q) * q_dot

    Parameters
    ----------
    joint_angles : array-like of length 6
        Joint angles in radians, in physical order
        (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3).

    Returns
    -------
    J : (6, 6) np.ndarray
        Top 3 rows: linear-velocity Jacobian (m/s per rad/s).
        Bottom 3 rows: angular-velocity Jacobian (rad/s per rad/s).
        All quantities expressed in base_link.
    """
    # forward_kinematics returns the end-effector transform and the list of
    # cumulative transforms from base_link to each joint frame, all already
    # corrected by T_BASE_CORRECTION.
    T_ee, all_T = forward_kinematics(joint_angles, return_all=True)

    # End-effector origin in base_link.
    O_n = T_ee[:3, 3]

    # Frame {i-1} for the i-th joint, expressed in base_link.
    # frames[0] = base of joint 1 = DH frame {0} = T_BASE_CORRECTION
    # frames[i] = base of joint i+1 = all_T[i-1]   for i = 1..5
    frames = [T_BASE_CORRECTION] + all_T[:5]

    J = np.zeros((6, 6))
    for i in range(6):
        T_prev = frames[i]
        z_prev = T_prev[:3, 2]      # z-axis of frame {i-1} in base_link
        o_prev = T_prev[:3, 3]      # origin of frame {i-1} in base_link

        # Linear part: contribution of joint i to v_ee
        J[:3, i] = np.cross(z_prev, O_n - o_prev)
        # Angular part: revolute joints contribute their own z-axis
        J[3:, i] = z_prev

    return J


# ============================================================================
# Static joint torques under payload
# ============================================================================
def compute_statics(joint_angles, payload_mass, gravity=9.81):
    """
    Compute the joint torques the motors must apply to hold a payload
    of mass M at the end-effector, against gravity.

    Derivation: gravity exerts a downward force F_g = -M*g*z_hat on the
    payload. To keep the end-effector static, the chain of joints must
    apply an equal and opposite force F_motor = +M*g*z_hat at the EE.
    By the principle of virtual work:

        tau = J^T(q) * F_motor

    Parameters
    ----------
    joint_angles : array-like of length 6
        Joint angles in radians, physical order.
    payload_mass : float
        Mass of the payload in kg. The UR5's rated maximum is 5 kg.
    gravity : float, optional
        Gravitational acceleration in m/s^2 (default 9.81).

    Returns
    -------
    torques : (6,) np.ndarray
        Torque the motor at each joint must produce, in Nm, in physical
        order. Positive sign follows the joint's positive rotation axis.
    """
    J = compute_jacobian(joint_angles)

    # Wrench the motors must apply at the EE to support the payload.
    # Force points in +z (up); no externally applied moment.
    F_motor = np.zeros(6)
    F_motor[2] = payload_mass * gravity

    return J.T @ F_motor


# ============================================================================
# Reference: numerical Jacobian via finite differences
# (used for validation of the analytical Jacobian, and for section 4.1.10.2.5.1)
# ============================================================================
def numerical_jacobian(joint_angles, eps=1e-7):
    """
    Numerical Jacobian via forward finite differences on forward_kinematics.

    The linear part is dp/dq, computed from the end-effector position.
    The angular part is extracted from the skew-symmetric matrix
        S(omega) = (dR/dq) * R^T
    yielding omega = [S(2,1), S(0,2), S(1,0)].

    Parameters
    ----------
    joint_angles : array-like of length 6
    eps : float, optional
        Finite-difference step (default 1e-7, near optimal for double).

    Returns
    -------
    J_num : (6, 6) np.ndarray
        Same layout and units as compute_jacobian().
    """
    q = np.asarray(joint_angles, dtype=float)
    J_num = np.zeros((6, 6))

    T0 = forward_kinematics(q)
    p0 = T0[:3, 3]
    R0 = T0[:3, :3]

    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        T1 = forward_kinematics(q + dq)
        p1 = T1[:3, 3]
        R1 = T1[:3, :3]

        # Linear part: dp/dq_i
        J_num[:3, i] = (p1 - p0) / eps

        # Angular part: extract omega from (dR) R^T / eps
        dR = (R1 - R0) @ R0.T / eps
        J_num[3, i] = dR[2, 1]
        J_num[4, i] = dR[0, 2]
        J_num[5, i] = dR[1, 0]

    return J_num


# ============================================================================
# Sanity check + analytical-vs-numerical validation
# ============================================================================
def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    print("=" * 78)
    print(" UR5 JACOBIAN AND STATICS - validation")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Part 1: analytical Jacobian vs numerical Jacobian
    # ------------------------------------------------------------------
    test_poses = [
        ('Home',            [0,        -np.pi/2,  0,        -np.pi/2,    0,        0]),
        ('45-degree bend',  [0,        -np.pi/4, -np.pi/4,   0,          np.pi/2,  0]),
        ('Side fold',       [np.pi/2,  -np.pi/2,  np.pi/2,  -np.pi/2,    0,        0]),
        ('Twisted',         [np.pi/4,  -np.pi/3,  np.pi/2,  -2*np.pi/3,  np.pi/2,  np.pi/4]),
        ('Horizontal reach',[0,         0,        0,        -np.pi/2,    0,        0]),
    ]

    print("\nPart 1: analytical vs numerical Jacobian (max abs element-wise diff)")
    print("-" * 78)
    worst_err = 0.0
    for name, q in test_poses:
        J_an = compute_jacobian(q)
        J_nu = numerical_jacobian(q)
        max_err = np.max(np.abs(J_an - J_nu))
        worst_err = max(worst_err, max_err)
        status = "[OK]" if max_err < 1e-4 else "[MISMATCH]"
        print(f"  {name:<18}  max diff = {max_err:.3e}   {status}")

    print(f"\n  Worst case across all poses: {worst_err:.3e}")
    if worst_err < 1e-4:
        print("  [PASS] Analytical Jacobian agrees with numerical Jacobian.")
    else:
        print("  [FAIL] Investigate the analytical formula.")

    # ------------------------------------------------------------------
    # Part 2: Jacobian at a representative pose (for the report)
    # ------------------------------------------------------------------
    q_demo = np.array([0.0, 0.0, 0.0, -np.pi/2, 0.0, 0.0])  # arm extended horizontally
    print("\n" + "=" * 78)
    print(" Part 2: Jacobian at the 'Horizontal reach' pose")
    print("=" * 78)
    print("  Joint angles (deg): "
          + str([f"{np.degrees(x):+.1f}" for x in q_demo]))
    J_demo = compute_jacobian(q_demo)
    print("\n  J =")
    print(J_demo)

    # ------------------------------------------------------------------
    # Part 3: static torques to hold a 5 kg payload
    # ------------------------------------------------------------------
    mass = 5.0
    tau = compute_statics(q_demo, payload_mass=mass)

    print("\n" + "=" * 78)
    print(f" Part 3: motor torques to hold a {mass} kg payload")
    print("=" * 78)
    joint_names = ['shoulder_pan ', 'shoulder_lift', 'elbow        ',
                   'wrist_1      ', 'wrist_2      ', 'wrist_3      ']
    for name, t in zip(joint_names, tau):
        print(f"  {name} : {t:+9.3f} Nm")

    # Quick hand-check: at the horizontal-reach pose the EE is at radius r
    # in the horizontal plane, so the shoulder_lift torque should be
    # approximately m * g * r (modulo the vertical offset to the joint axis).
    from ur5_project.kinematics import forward_kinematics as fk
    T_demo = fk(q_demo)
    r_horiz = np.linalg.norm(T_demo[:2, 3])
    print(f"\n  Hand estimate of |shoulder_lift| torque: "
          f"m*g*r_horiz = {mass*9.81*r_horiz:.2f} Nm")
    print(f"  Computed |shoulder_lift| torque:         {abs(tau[1]):.2f} Nm")


if __name__ == "__main__":
    main()
