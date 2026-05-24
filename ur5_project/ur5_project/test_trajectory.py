#!/usr/bin/env python3
"""
test_trajectory.py
==================
Wrapper for testing the full UR5 simulation pipeline with arbitrary
Cartesian start and end poses (X, Y, Z, Roll, Pitch, Yaw), matching the
input format required by assignment section 4.1.9.1.

Workflow
--------
1. Edit START_POSE and END_POSE in the EDIT_THIS section below.
2. Run:    ros2 run ur5_project test_trajectory
3. The script first validates that:
     a. Both endpoints are reachable (IK returns at least one solution).
     b. Neither endpoint is near a wrist-2 singularity (|sin θ_5| > threshold).
     c. The straight-line Cartesian path between them is reachable
        at every sampled waypoint.
   If any check fails, it explains what went wrong instead of crashing
   deep inside the simulation.

4. If all checks pass, it runs the full plan_trajectory + simulation
   pipeline and writes the same set of plots and the MP4 video into a
   timestamped subdirectory of docs/report_assets/ (so multiple test
   runs don't overwrite each other).

Author: Daniel
Course: Kinematics and Dynamics of Robots, Ben-Gurion University, 2026
"""

import os
import sys
import time
import argparse
import numpy as np

from ur5_project.kinematics import forward_kinematics
from ur5_project.inverse_kinematics import inverse_kinematics
from ur5_project.trajectory import pose_to_transform, plan_trajectory
from pathlib import Path

from ur5_project.simulation import (
    plot_stick_diagrams, plot_ee_pose, plot_joint_angles,
    plot_velocity_comparison, plot_torques, animate_motion,
    compute_velocities, compute_torques,
)

# This file lives at <REPO>/ur5_project/ur5_project/test_trajectory.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "report_assets"


# ============================================================================
# === EDIT THIS: Cartesian start and end poses ===============================
# ============================================================================
# Format: [x, y, z, roll, pitch, yaw]
#   - x, y, z in meters (base_link frame)
#   - roll, pitch, yaw in DEGREES (converted to radians automatically)
#
# UR5 reach envelope (approximate):
#   - Sphere of radius ~0.85 m centered above the base
#   - Stay 5-10 cm inside the boundary for numerical safety
#   - Keep z above 0 to avoid ground collisions in Gazebo
#
# To convert a known joint configuration to a Cartesian pose, you can
# uncomment the helper at the bottom of this file.
# ============================================================================

START_POSE = [-0.07,  0.11,  0.89,  -90.0,   0.0,  90.0]   # default reachable
END_POSE   = [ 0.21,  0.25,  0.91,  -40.9,  20.7, 142.2]   # default reachable

# Other tested examples (uncomment one pair to try):
#
# Side-to-side at mid-height
# START_POSE = [-0.30, 0.20, 0.50,  0.0, 90.0,  0.0]
# END_POSE   = [ 0.30, 0.20, 0.50,  0.0, 90.0,  0.0]
#
# Reaching down and forward
# START_POSE = [ 0.20, 0.10, 0.70,  0.0,  45.0, 0.0]
# END_POSE   = [ 0.50, 0.10, 0.30,  0.0,  90.0, 0.0]


# ============================================================================
# Configuration parameters
# ============================================================================
PAYLOAD_KG    = 3.0
V_MAX         = 0.25
A_MAX         = 0.5
DT            = 0.01
WRIST_SING_TOL= 0.05    # |sin(theta_5)| must be > this to avoid wrist singularity
N_PATH_PROBES = 21      # how many points to check along the path before planning


# ============================================================================
# Validation helpers
# ============================================================================
def pose_deg_to_rad(p):
    """Convert [x, y, z, roll, pitch, yaw(deg)] to [..., rpy(rad)]."""
    p = np.asarray(p, dtype=float).copy()
    p[3:] = np.radians(p[3:])
    return p


def check_endpoint(label, pose_rad):
    """
    Check whether a single end-effector pose is reachable and non-singular.

    Returns
    -------
    ok : bool
    q  : np.ndarray or None
        A representative IK solution (used later as a seed).
    msg : str
        Diagnostic message.
    """
    T = pose_to_transform(pose_rad)
    sols = inverse_kinematics(T)

    if len(sols) == 0:
        return False, None, (
            f"  [FAIL] {label}: no IK solution. "
            f"Pose likely outside UR5 reach envelope or in deep singularity."
        )

    # Pick the solution with the largest |sin theta_5| - the "least singular"
    sin5 = np.abs(np.sin(sols[:, 4]))
    best = np.argmax(sin5)
    q = sols[best]

    if sin5[best] < WRIST_SING_TOL:
        return False, q, (
            f"  [FAIL] {label}: wrist-2 singularity. "
            f"|sin(theta_5)| = {sin5[best]:.4f} < {WRIST_SING_TOL} "
            f"in best of {len(sols)} solutions."
        )

    return True, q, (
        f"  [OK]   {label}: {len(sols)} IK solution(s), "
        f"|sin(theta_5)| up to {sin5[best]:.3f}."
    )


def check_path(start_rad, end_rad, n_probes=N_PATH_PROBES):
    """
    Probe IK at n_probes evenly spaced points along the linear path. Returns
    list of failure indices and total count (so we can warn or abort
    before running the full simulation).
    """
    failures = []
    for i in range(n_probes):
        u = i / (n_probes - 1)
        pose = start_rad + u * (end_rad - start_rad)
        T = pose_to_transform(pose)
        sols = inverse_kinematics(T)
        if len(sols) == 0:
            failures.append((i, u, pose))
    return failures


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run UR5 simulation between arbitrary Cartesian poses"
    )
    parser.add_argument('--no_video', action='store_true',
                        help='Skip MP4 rendering')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plots interactively')
    parser.add_argument('--tag', type=str, default=None,
                        help="Subfolder name suffix (default: timestamp)")
    args = parser.parse_args()

    print("=" * 72)
    print(" UR5 Cartesian trajectory test")
    print("=" * 72)

    # ----- Convert to radians and report -----
    p0 = pose_deg_to_rad(START_POSE)
    p1 = pose_deg_to_rad(END_POSE)

    print(f"  START: xyz = {p0[:3].round(4)} m, "
          f"rpy = {np.degrees(p0[3:]).round(2)} deg")
    print(f"  END:   xyz = {p1[:3].round(4)} m, "
          f"rpy = {np.degrees(p1[3:]).round(2)} deg")
    L = np.linalg.norm(p1[:3] - p0[:3])
    print(f"  Straight-line distance: {L:.4f} m")

    # ----- Endpoint validation -----
    print("\nValidating endpoints...")
    ok_s, q_seed_start, msg_s = check_endpoint("start", p0)
    print(msg_s)
    ok_e, q_seed_end,   msg_e = check_endpoint("end",   p1)
    print(msg_e)

    if not (ok_s and ok_e):
        print("\nABORT: at least one endpoint failed validation. "
              "Edit START_POSE / END_POSE in this file and try again.")
        sys.exit(1)

    # ----- Path probing -----
    print(f"\nProbing the linear path at {N_PATH_PROBES} sample points...")
    failures = check_path(p0, p1)
    if failures:
        print(f"  [FAIL] {len(failures)}/{N_PATH_PROBES} intermediate "
              f"waypoints have no IK solution. First failure:")
        i, u, pose = failures[0]
        print(f"    u = {u:.3f}: xyz = {pose[:3].round(4)}, "
              f"rpy = {np.degrees(pose[3:]).round(2)} deg")
        print("\n  This usually means the straight-line path leaves the "
              "reachable workspace at some intermediate point, even though "
              "both endpoints are individually reachable. Try different "
              "endpoints, or split the motion into two segments.")
        sys.exit(2)
    print(f"  [OK]   all {N_PATH_PROBES} probe points have valid IK.")

    # ----- All checks passed, run the simulation -----
    # Use a timestamped or tagged subdir so multiple runs don't overwrite
    if args.tag:
        tag = args.tag
    else:
        tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = str(DEFAULT_REPORT_DIR / f"test_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutputs will be written to: {out_dir}/")

    print("\nPlanning trajectory...")
    traj = plan_trajectory(p0, p1, v_max=V_MAX, a_max=A_MAX, dt=DT,
                           initial_joints=q_seed_start)
    print(f"  Generated {len(traj['t'])} waypoints, "
          f"duration {traj['t'][-1]:.3f} s.")

    # ----- Plots -----
    print("\nGenerating plots...")
    plot_stick_diagrams(traj,
        save_path=os.path.join(out_dir, 'sim_stick_diagrams.png'))
    plot_ee_pose(traj,
        save_path=os.path.join(out_dir, 'sim_ee_pose.png'))
    plot_joint_angles(traj,
        save_path=os.path.join(out_dir, 'sim_joint_angles.png'))
    plot_velocity_comparison(traj,
        save_path=os.path.join(out_dir, 'sim_velocity_compare.png'))
    plot_torques(traj, PAYLOAD_KG,
        save_path=os.path.join(out_dir, 'sim_torques.png'))

    # Numerical summary
    v_num, v_an = compute_velocities(traj)
    tau = compute_torques(traj, PAYLOAD_KG)
    print(f"  Peak |v_num - v_analytic|:    {np.max(np.abs(v_num - v_an)):.3e} m/s")
    print(f"  Peak |shoulder_lift| torque:  {np.max(np.abs(tau[:, 1])):.2f} Nm")
    print(f"  Peak |elbow|         torque:  {np.max(np.abs(tau[:, 2])):.2f} Nm")
    print(f"  Plots saved.")

    # ----- MP4 -----
    if not args.no_video:
        print("\nRendering MP4 animation...")
        try:
            animate_motion(traj,
                           save_path=os.path.join(out_dir, 'sim_motion.mp4'),
                           fps=30, slowdown=3.0)
        except FileNotFoundError as e:
            print(f"  [WARN] MP4 rendering skipped: {e}")
            print("  Install with: sudo apt install ffmpeg")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()

    print(f"\nDone. See {out_dir}/")


# ============================================================================
# Helper: convert a known joint config to a Cartesian pose for editing
# ============================================================================
# Uncomment and run this section if you want to know the Cartesian pose
# corresponding to a given joint configuration -- useful to find safe,
# non-singular start/end candidates.
#
# def joints_to_pose():
#     from kinematics import rotation_to_rpy
#     q = np.array([0.0, -np.pi/3, -np.pi/3, -np.pi/3, np.pi/2, 0.0])
#     T = forward_kinematics(q)
#     xyz = T[:3, 3]
#     rpy = np.degrees(rotation_to_rpy(T[:3, :3]))
#     print(f"Pose: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}, "
#           f"{rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}]")
#
# if __name__ == "__main__":
#     joints_to_pose()


if __name__ == '__main__':
    main()
