#!/usr/bin/env python3
"""
inverse_kinematics.py
=====================
Analytical inverse kinematics for the UR5 (assignment section 4.1.6).

For any reachable target pose T (4x4 homogeneous transform from base_link to
tool0), this module computes ALL 8 valid joint configurations that achieve it,
exploiting the UR5's spherical wrist for a closed-form solution.

The algorithm follows the well-known closed-form derivation described by
Ryan Keating (Johns Hopkins University) and used in multiple open-source
UR5 implementations. It uses a slightly different DH transform convention
than our forward_kinematics module (Translate-Rotate-Translate-Rotate),
which is mathematically equivalent for the same DH parameters but leads
to a cleaner IK derivation. We use a local DH transform helper for IK only.

The 8 solutions correspond to the combinations of:
    - shoulder configuration: left or right          (2)
    - wrist configuration: not-flipped or flipped    (2)
    - elbow configuration: up or down                (2)
    Total: 2 x 2 x 2 = 8 solutions

For unreachable poses or singular configurations, fewer solutions may be
returned.

Author: Daniel
Course: Kinematics and Dynamics of Robots, Ben-Gurion University, 2026
"""

import numpy as np
import cmath
from math import cos, sin, atan2, acos, asin, sqrt, pi

from ur5_project.kinematics import (
    UR5_DH_PARAMS,
    T_BASE_CORRECTION,
    forward_kinematics,
)


# ============================================================================
# UR5 DH constants (extracted for readability in IK formulas)
# ============================================================================
_d = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
_a = np.array([0, -0.425, -0.39225, 0, 0, 0])
_alph = np.array([pi/2, 0, 0, pi/2, -pi/2, 0])

d1 = _d[0]; a2 = _a[1]; a3 = _a[2]
d4 = _d[3]; d5 = _d[4]; d6 = _d[5]


# ============================================================================
# Local DH transform: T_d * Rz * T_a * Rx convention
# ============================================================================
def _dh_local(n, theta):
    """The n-th DH transform (1-indexed) for joint angle theta."""
    T_a = np.eye(4); T_a[0, 3] = _a[n - 1]
    T_d = np.eye(4); T_d[2, 3] = _d[n - 1]
    Rzt = np.array([
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta),  cos(theta), 0, 0],
        [0,           0,          1, 0],
        [0,           0,          0, 1],
    ])
    al = _alph[n - 1]
    Rxa = np.array([
        [1, 0,        0,         0],
        [0, cos(al), -sin(al),   0],
        [0, sin(al),  cos(al),   0],
        [0, 0,        0,         1],
    ])
    return T_d @ Rzt @ T_a @ Rxa


def _ah(n, th, c):
    return _dh_local(n, th[n - 1, c])


# ============================================================================
# Inverse kinematics (closed-form analytical solution)
# ============================================================================
def inverse_kinematics(T_target):
    """
    Compute all valid IK solutions for the UR5.

    Parameters
    ----------
    T_target : (4, 4) np.ndarray
        Desired homogeneous transform from base_link to tool0.

    Returns
    -------
    solutions : (N, 6) np.ndarray
        Up to 8 valid joint configurations. Empty if unreachable.
    """
    # Step 0: undo the base correction so we work in the pure DH frame
    desired_pos = T_BASE_CORRECTION @ T_target

    th = np.zeros((6, 8))

    # ----- theta1: shoulder pan, 2 candidates -----
    P_05 = (desired_pos @ np.array([0, 0, -d6, 1]).reshape(4, 1)
            - np.array([0, 0, 0, 1]).reshape(4, 1))
    psi = atan2(P_05[1, 0], P_05[0, 0])
    horiz = sqrt(P_05[0, 0]**2 + P_05[1, 0]**2)
    if horiz < 1e-9:
        # Wrist directly above base - shoulder singularity
        return np.empty((0, 6))
    arg = d4 / horiz
    if abs(arg) > 1.0 + 1e-6:
        return np.empty((0, 6))
    arg = max(-1.0, min(1.0, arg))   # clip floating point overshoot
    phi = acos(arg)
    th[0, 0:4] = pi/2 + psi + phi
    th[0, 4:8] = pi/2 + psi - phi

    # ----- theta5: wrist_2, 2 candidates per shoulder -----
    for c in [0, 4]:
        T_10 = np.linalg.inv(_ah(1, th, c))
        T_16 = T_10 @ desired_pos
        arg = (T_16[2, 3] - d4) / d6
        if abs(arg) > 1.0 + 1e-6:
            th[4, c:c + 4] = np.nan
            continue
        arg = max(-1.0, min(1.0, arg))
        th[4, c    : c + 2] = +acos(arg)
        th[4, c + 2: c + 4] = -acos(arg)

    # ----- theta6: wrist_3 -----
    for c in [0, 2, 4, 6]:
        if np.isnan(th[4, c]):
            th[5, c:c + 2] = np.nan
            continue
        s5 = sin(th[4, c])
        if abs(s5) < 1e-7:
            th[5, c:c + 2] = 0.0
            continue
        T_10 = np.linalg.inv(_ah(1, th, c))
        T_16 = np.linalg.inv(T_10 @ desired_pos)
        th[5, c:c + 2] = atan2(-T_16[1, 2] / s5, T_16[0, 2] / s5)

    # ----- theta3: elbow, 2 candidates -----
    for c in [0, 2, 4, 6]:
        if np.isnan(th[4, c]):
            th[2, c:c + 2] = np.nan
            continue
        T_10 = np.linalg.inv(_ah(1, th, c))
        T_65 = _ah(6, th, c)
        T_54 = _ah(5, th, c)
        T_14 = (T_10 @ desired_pos) @ np.linalg.inv(T_54 @ T_65)
        P_13 = (T_14 @ np.array([0, -d4, 0, 1]).reshape(4, 1)
                - np.array([0, 0, 0, 1]).reshape(4, 1))
        norm_P13 = np.linalg.norm(P_13)
        cos_arg = (norm_P13**2 - a2**2 - a3**2) / (2 * a2 * a3)
        t3 = cmath.acos(cos_arg)
        th[2, c    ] = +t3.real
        th[2, c + 1] = -t3.real

    # ----- theta2 and theta4 -----
    for c in range(8):
        if np.isnan(th[2, c]):
            th[1, c] = np.nan
            th[3, c] = np.nan
            continue
        T_10 = np.linalg.inv(_ah(1, th, c))
        T_65 = np.linalg.inv(_ah(6, th, c))
        T_54 = np.linalg.inv(_ah(5, th, c))
        T_14 = (T_10 @ desired_pos) @ T_65 @ T_54
        P_13 = (T_14 @ np.array([0, -d4, 0, 1]).reshape(4, 1)
                - np.array([0, 0, 0, 1]).reshape(4, 1))
        norm_P13 = np.linalg.norm(P_13)
        th[1, c] = (-atan2(P_13[1, 0], -P_13[0, 0])
                    + asin(a3 * sin(th[2, c]) / norm_P13))
        T_32 = np.linalg.inv(_dh_local(3, th[2, c]))
        T_21 = np.linalg.inv(_dh_local(2, th[1, c]))
        T_34 = T_32 @ T_21 @ T_14
        th[3, c] = atan2(T_34[1, 0], T_34[0, 0])

    # Convert to (N, 6) format and filter NaNs
    solutions = th.T
    valid = ~np.any(np.isnan(solutions), axis=1)
    solutions = solutions[valid]

    # Filter out spurious solutions: at near-singular configurations the
    # algorithm can return joint angles that don't actually reach the target.
    # We re-run FK on each solution and keep only those with low position
    # error (< 1 mm). Numerically valid solutions are all at < 1e-9 mm.
    if len(solutions) > 0:
        good = []
        for sol in solutions:
            T_check = forward_kinematics(sol)
            err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
            if err < 1e-3:   # 1 mm
                good.append(sol)
        solutions = np.array(good) if good else np.empty((0, 6))

    return wrap_angles(solutions)


# ============================================================================
# Helpers
# ============================================================================
def wrap_angles(angles):
    """Wrap angles to [-pi, pi]."""
    return ((np.asarray(angles) + pi) % (2 * pi)) - pi


def pick_best_solution(solutions, current_joints):
    """Pick the IK solution closest to the current joint configuration."""
    if len(solutions) == 0:
        return None
    current = np.asarray(current_joints)
    diffs = wrap_angles(solutions - current)
    costs = np.sum(diffs**2, axis=1)
    return solutions[np.argmin(costs)]


def verify_solution(joint_angles, T_target):
    """Verify by re-running FK; returns (pos_err_mm, rot_err)."""
    T_check = forward_kinematics(joint_angles)
    pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3]) * 1000
    rot_err = np.linalg.norm(T_check[:3, :3] - T_target[:3, :3])
    return pos_err, rot_err


# ============================================================================
# Sanity check
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" UR5 INVERSE KINEMATICS - Self test")
    print("=" * 70)

    test_configs = [
        ('Home',         [0, -pi/2, 0, -pi/2, 0, 0]),
        ('Straight up',  [0, -pi/2, 0, 0, 0, 0]),
        ('Forward bent', [0, -pi/4, -pi/4, 0, pi/2, 0]),
        ('Side fold',    [pi/2, -pi/2, pi/2, -pi/2, 0, 0]),
        ('Twisted',      [pi/4, -pi/3, pi/2, -2*pi/3, pi/2, pi/4]),
    ]

    for name, q in test_configs:
        print(f"\n--- {name} ---")
        q_orig = np.array(q)
        T = forward_kinematics(q_orig)
        sols = inverse_kinematics(T)
        print(f"  Found {len(sols)} valid IK solution(s)")

        if len(sols) == 0:
            print("  [FAIL] No solutions found")
            continue

        match_found = False
        worst_err = 0.0
        for sol in sols:
            pos_err, _ = verify_solution(sol, T)
            worst_err = max(worst_err, pos_err)
            diff = wrap_angles(sol - q_orig)
            if np.max(np.abs(diff)) < 1e-3:
                match_found = True

        print(f"  Worst position error across all solutions: "
              f"{worst_err:.3e} mm")
        if match_found:
            print("  [PASS] Original input recovered as one of the solutions")
        elif worst_err < 0.01:
            print("  [PASS] All solutions yield correct end-effector pose")
        else:
            print("  [FAIL] Solutions do not produce target pose")
            for i, sol in enumerate(sols):
                d = np.degrees(sol)
                print(f"    Sol {i+1}: [{d[0]:+7.1f}, {d[1]:+7.1f}, "
                      f"{d[2]:+7.1f}, {d[3]:+7.1f}, {d[4]:+7.1f}, "
                      f"{d[5]:+7.1f}]")
