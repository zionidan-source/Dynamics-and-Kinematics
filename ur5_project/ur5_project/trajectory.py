#!/usr/bin/env python3
"""
ur5_project/trajectory.py
=========================
Cartesian linear trajectory planning with a trapezoidal velocity profile
(assignment section 4.1.9).

Given a start pose and an end pose, each specified by (X, Y, Z) position and
(Roll, Pitch, Yaw) Euler angles in the world (base_link) frame, this module
generates a time-sampled trajectory that:

  * moves the end-effector along a straight line in Cartesian space
  * interpolates orientation linearly in Roll-Pitch-Yaw
  * follows a trapezoidal velocity profile (constant acceleration to v_max,
    constant cruise at v_max, constant deceleration to a stop)
  * is sampled at uniform time steps dt

If the path is too short to reach v_max, the profile degenerates to a
triangle (accelerate-then-decelerate, no cruise phase).

For each Cartesian waypoint we also solve inverse kinematics to obtain the
corresponding joint-space trajectory, choosing the IK branch closest to the
previous waypoint's joints for continuity.

Author: Daniel
Course: Kinematics and Dynamics of Robots, Ben-Gurion University, 2026
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pathlib import Path

from ur5_project.kinematics import forward_kinematics
from ur5_project.inverse_kinematics import inverse_kinematics, pick_best_solution

# This file lives at <REPO>/ur5_project/ur5_project/trajectory.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = str(REPO_ROOT / "docs" / "report_assets")


# ============================================================================
# Pose <-> homogeneous transform conversions
# ============================================================================
def rpy_to_rotation(roll, pitch, yaw):
    """
    Build a 3x3 rotation matrix from Roll-Pitch-Yaw angles (world XYZ
    convention, intrinsic Z-Y-X is the same matrix). The full rotation is:

        R = Rz(yaw) * Ry(pitch) * Rx(roll)

    which rotates first about world X by roll, then about world Y by pitch,
    then about world Z by yaw.
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1, 0,  0 ], [0, cr, -sr], [0, sr,  cr]])
    Ry = np.array([[cp, 0, sp], [0,  1, 0  ], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0,  0,  1]])
    return Rz @ Ry @ Rx


def pose_to_transform(pose):
    """
    Convert a 6-element pose [x, y, z, roll, pitch, yaw] to a 4x4
    homogeneous transform (base_link to tool0).
    """
    x, y, z, roll, pitch, yaw = pose
    T = np.eye(4)
    T[:3, :3] = rpy_to_rotation(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


# ============================================================================
# Trapezoidal velocity profile
# ============================================================================
def trapezoidal_profile(L, v_max, a_max, dt):
    """
    Generate the time-sampled arc-length s(t) along a path of total length L
    using a trapezoidal velocity profile.

    Phases (in time):
      0    .. t_acc    : v = a_max * t                  (accelerate)
      t_acc .. t_acc+t_cruise : v = v_max               (cruise)
      t_acc+t_cruise .. T : v = v_max - a_max*(t - t_acc - t_cruise) (decel)

    If the path is too short to ever reach v_max, the profile is a triangle:
      0    .. T/2 : accelerate
      T/2  .. T   : decelerate

    Parameters
    ----------
    L : float
        Total path length (e.g. straight-line distance in Cartesian space).
    v_max : float
        Maximum allowed velocity (m/s).
    a_max : float
        Constant acceleration during ramp-up and ramp-down (m/s^2).
    dt : float
        Sampling time step (s).

    Returns
    -------
    t  : (N,) np.ndarray  -- time samples, starting at 0
    s  : (N,) np.ndarray  -- arc length along the path, 0 to L
    sd : (N,) np.ndarray  -- |v(t)|, the speed
    """
    if L <= 0:
        # Degenerate case: start == end
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    # Distance covered during a full accel-from-0-to-vmax ramp:
    L_ramp = v_max**2 / a_max  # = 2 * (1/2 * a * t_acc^2),  t_acc = v_max/a_max

    if L >= L_ramp:
        # Trapezoidal: we DO reach v_max
        t_acc    = v_max / a_max
        L_cruise = L - L_ramp
        t_cruise = L_cruise / v_max
        T_total  = 2 * t_acc + t_cruise
    else:
        # Triangular: we never reach v_max
        t_acc    = np.sqrt(L / a_max)
        t_cruise = 0.0
        T_total  = 2 * t_acc

    # Sample. Force the final point to land exactly on T_total so the path
    # ends precisely at L (avoid round-off leaving the last sample short).
    N = int(np.ceil(T_total / dt)) + 1
    t = np.linspace(0.0, T_total, N)

    s  = np.zeros(N)
    sd = np.zeros(N)
    for i, ti in enumerate(t):
        if ti <= t_acc:
            # Acceleration phase
            sd[i] = a_max * ti
            s[i]  = 0.5 * a_max * ti**2
        elif ti <= t_acc + t_cruise:
            # Cruise phase
            sd[i] = v_max
            s[i]  = 0.5 * a_max * t_acc**2 + v_max * (ti - t_acc)
        else:
            # Deceleration phase
            t_dec = ti - t_acc - t_cruise
            sd[i] = v_max - a_max * t_dec if t_cruise > 0 else a_max * t_acc - a_max * t_dec
            v_top = v_max if t_cruise > 0 else a_max * t_acc
            s_at_dec_start = 0.5 * a_max * t_acc**2 + v_top * t_cruise
            s[i] = s_at_dec_start + v_top * t_dec - 0.5 * a_max * t_dec**2

    # Clip floating-point overshoot at the end
    s[-1] = L
    sd[-1] = 0.0
    return t, s, sd


# ============================================================================
# Full Cartesian trajectory + joint-space trajectory
# ============================================================================
def plan_trajectory(start_pose, end_pose,
                    v_max=0.25, a_max=0.5, dt=0.01,
                    initial_joints=None):
    """
    Plan a Cartesian-linear trajectory between two 6-DOF poses with a
    trapezoidal velocity profile, and compute the joint-space trajectory
    by per-waypoint inverse kinematics.

    Parameters
    ----------
    start_pose, end_pose : array-like of length 6
        [x, y, z, roll, pitch, yaw] in meters and radians.
    v_max : float, optional
        Peak Cartesian linear speed (m/s). Default 0.25.
    a_max : float, optional
        Cartesian linear acceleration (m/s^2). Default 0.5.
    dt : float, optional
        Time step between samples (s). Default 0.01 -> 100 Hz.
    initial_joints : array-like of length 6, optional
        Joint configuration to bias the first IK call. If None, an arbitrary
        well-behaved seed is used.

    Returns
    -------
    traj : dict with keys
        't'         : (N,) time samples
        'cartesian' : (N, 6) [x, y, z, roll, pitch, yaw] along the path
        'joints'    : (N, 6) joint configurations from IK
        'speed'     : (N,) end-effector linear speed magnitude |v(t)|
        's'         : (N,) arc length along the path
    """
    p0 = np.asarray(start_pose, dtype=float)
    p1 = np.asarray(end_pose,   dtype=float)

    # Cartesian straight-line direction (positional component only)
    delta_pos = p1[:3] - p0[:3]
    L = np.linalg.norm(delta_pos)

    if L < 1e-9:
        raise ValueError("Start and end positions are identical; "
                         "no meaningful Cartesian path.")

    # Trapezoidal profile along the straight-line arc
    t, s, sd = trapezoidal_profile(L, v_max, a_max, dt)
    N = len(t)

    # Parametric position along the line: r(u) = p0 + u * (p1 - p0), u in [0,1]
    u = s / L  # (N,)

    # Cartesian waypoints (position + RPY)
    cart = np.zeros((N, 6))
    for i in range(6):
        cart[:, i] = p0[i] + u * (p1[i] - p0[i])

    # Joint-space via per-waypoint IK with continuity
    joints = np.zeros((N, 6))
    if initial_joints is None:
        # Reasonable seed for the UR5: the 'Home' pose
        q_seed = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
    else:
        q_seed = np.asarray(initial_joints, dtype=float)

    q_prev = q_seed
    for i in range(N):
        T_target = pose_to_transform(cart[i])
        sols = inverse_kinematics(T_target)
        if len(sols) == 0:
            raise RuntimeError(
                f"IK failed at waypoint {i}/{N-1}: target pose "
                f"{cart[i]} is unreachable or singular."
            )
        q = pick_best_solution(sols, q_prev)
        joints[i] = q
        q_prev = q

    return {
        't':         t,
        'cartesian': cart,
        'joints':    joints,
        'speed':     sd,
        's':         s,
    }


# ============================================================================
# Acceleration (joint-space angular + Cartesian magnitude)
# ============================================================================
def compute_acceleration(traj):
    """
    Compute the acceleration of the motion in both spaces, purely from the
    time-sampled trajectory, using numerical central differences.

    The trajectory dict returned by plan_trajectory() stores positions but
    not their derivatives, so we differentiate twice:

    Joint space: q_dot is the central-difference derivative of the joint
                 trajectory; q_ddot is the central-difference derivative of
                 q_dot -> joint angular acceleration in rad/s^2.
    Work space:  the numerical end-effector velocity is the central-difference
                 derivative of the Cartesian position; differentiating it once
                 more gives the linear acceleration, reported as |a(t)|.

    We deliberately differentiate the numerical velocity (rather than reading
    the analytic trapezoidal profile) so the plot exposes the real
    finite-difference noise, exactly as plot_velocity_comparison() does for
    the velocity.

    Parameters
    ----------
    traj : dict
        The output of plan_trajectory().

    Returns
    -------
    a_joint    : (N, 6) np.ndarray  -- joint angular acceleration (rad/s^2)
    a_cart_mag : (N,)   np.ndarray  -- end-effector linear |a(t)| (m/s^2)
    """
    t   = traj['t']
    pos = traj['cartesian'][:, :3]   # (N, 3)
    q   = traj['joints']             # (N, 6)

    def central_diff(y):
        """Central differences in time, forward/backward at the boundaries."""
        d = np.zeros_like(y)
        d[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])[:, None]
        d[0]    = (y[1]  - y[0])   / (t[1]  - t[0])
        d[-1]   = (y[-1] - y[-2])  / (t[-1] - t[-2])
        return d

    # Joint space: differentiate twice (q -> q_dot -> q_ddot)
    qdot  = central_diff(q)
    qddot = central_diff(qdot)

    # Work space: differentiate the numerical velocity of the EE position
    v_num = central_diff(pos)
    a_num = central_diff(v_num)

    return qddot, np.linalg.norm(a_num, axis=1)


# ============================================================================
# Plotting
# ============================================================================
def plot_trajectory(traj, save_dir=None, show=True):
    """
    Produce the four figures required for section 4.1.9:
      * 3D plot of the Cartesian path (4.1.9.3)
      * Speed |v(t)| vs time (4.1.9.4)
      * Joint angles vs time (preview for 4.1.10)
      * Joint and Cartesian acceleration vs time
    """
    t    = traj['t']
    cart = traj['cartesian']
    joints = traj['joints']
    speed = traj['speed']

    # ---------- Figure 1: 3D Cartesian path ----------
    fig1 = plt.figure(figsize=(8, 7))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(cart[:, 0], cart[:, 1], cart[:, 2],
             '-', color='steelblue', linewidth=1.5, label='path')
    ax1.scatter(cart[::max(1, len(t)//30), 0],
                cart[::max(1, len(t)//30), 1],
                cart[::max(1, len(t)//30), 2],
                color='steelblue', s=20, label='samples')
    ax1.scatter(cart[0, 0],  cart[0, 1],  cart[0, 2],
                color='green', s=80, marker='o', label='start')
    ax1.scatter(cart[-1, 0], cart[-1, 1], cart[-1, 2],
                color='red',   s=80, marker='*', label='end')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Cartesian linear path with trapezoidal-profile sampling')
    ax1.legend()

    # ---------- Figure 2: |v(t)| ----------
    fig2 = plt.figure(figsize=(9, 5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(t, speed, '-', color='darkorange', linewidth=2)
    ax2.fill_between(t, 0, speed, color='darkorange', alpha=0.15)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('|v(t)| (m/s)')
    ax2.set_title('End-effector speed (world frame) — trapezoidal profile')
    ax2.grid(True, alpha=0.3)

    # ---------- Figure 3: joint angles ----------
    fig3 = plt.figure(figsize=(9, 6))
    ax3 = fig3.add_subplot(111)
    joint_labels = ['shoulder_pan', 'shoulder_lift', 'elbow',
                    'wrist_1', 'wrist_2', 'wrist_3']
    for i in range(6):
        ax3.plot(t, np.degrees(joints[:, i]),
                 linewidth=1.5, label=joint_labels[i])
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('joint angle (deg)')
    ax3.set_title('Joint angles along the trajectory')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', ncol=2)

    # ---------- Figure 4: accelerations (joint-space + Cartesian) ----------
    a_joint, a_cart_mag = compute_acceleration(traj)
    fig4, axes4 = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for i in range(6):
        axes4[0].plot(t, a_joint[:, i], linewidth=1.5, label=joint_labels[i])
    axes4[0].set_ylabel('joint accel. (rad/s²)')
    axes4[0].set_title('Joint angular acceleration along the trajectory')
    axes4[0].grid(True, alpha=0.3)
    axes4[0].legend(loc='best', ncol=2)

    axes4[1].plot(t, a_cart_mag, '-', color='darkorange', linewidth=2)
    axes4[1].fill_between(t, 0, a_cart_mag, color='darkorange', alpha=0.15)
    axes4[1].set_ylabel('|a(t)| (m/s²)')
    axes4[1].set_xlabel('time (s)')
    axes4[1].set_title('End-effector linear acceleration (world frame)')
    axes4[1].grid(True, alpha=0.3)
    fig4.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, 'trajectory_cartesian_path.png'),
                     dpi=150, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, 'trajectory_speed_profile.png'),
                     dpi=150, bbox_inches='tight')
        fig3.savefig(os.path.join(save_dir, 'trajectory_joint_angles.png'),
                     dpi=150, bbox_inches='tight')
        fig4.savefig(os.path.join(save_dir, 'trajectory_acceleration.png'),
                     dpi=150, bbox_inches='tight')
        print(f"  Plots saved to {save_dir}/")

    if show:
        plt.show()


# ============================================================================
# Main: demo with a reachable straight-line motion
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='UR5 Cartesian-linear trajectory with trapezoidal velocity'
    )
    parser.add_argument('--v_max', type=float, default=0.25,
                        help='Peak Cartesian speed (m/s), default 0.25')
    parser.add_argument('--a_max', type=float, default=0.5,
                        help='Cartesian acceleration (m/s^2), default 0.5')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step (s), default 0.01')
    parser.add_argument('--save', type=str,
                        default=DEFAULT_SAVE_DIR,
                        help=f'Where to save the plots (default: {DEFAULT_SAVE_DIR})')
    parser.add_argument('--no_show', action='store_true',
                        help='Disable interactive plt.show()')
    args = parser.parse_args()

    # ----- Demo motion: between two non-singular reachable poses -----
    # We compute the start and end poses by forward-kinematics on two
    # joint configurations we know are reachable, with the wrist away from
    # theta5 = 0 (singular) so the analytical IK behaves well along the
    # entire interpolated path.
    q_start = np.array([0.0,      -np.pi/3, -np.pi/3, -np.pi/3, np.pi/2, 0.0    ])
    q_end   = np.array([np.pi/6,  -np.pi/4, -np.pi/4, -np.pi/4, np.pi/2, np.pi/6])

    T_start = forward_kinematics(q_start)
    T_end   = forward_kinematics(q_end)

    # Extract (x, y, z, roll, pitch, yaw). We use a helper to keep the
    # Euler-angle convention consistent with rpy_to_rotation().
    from ur5_project.kinematics import rotation_to_rpy
    rpy_start = rotation_to_rpy(T_start[:3, :3])
    rpy_end   = rotation_to_rpy(T_end[:3, :3])

    start_pose = np.concatenate([T_start[:3, 3], rpy_start])
    end_pose   = np.concatenate([T_end[:3, 3],   rpy_end])

    print("=" * 70)
    print(" UR5 TRAJECTORY PLANNING - section 4.1.9")
    print("=" * 70)
    print(f"  Start pose: [x,y,z]= {start_pose[:3].round(4)}  "
          f"[r,p,y]= {np.degrees(start_pose[3:]).round(2)} deg")
    print(f"  End   pose: [x,y,z]= {end_pose[:3].round(4)}  "
          f"[r,p,y]= {np.degrees(end_pose[3:]).round(2)} deg")
    L = np.linalg.norm(end_pose[:3] - start_pose[:3])
    print(f"  Straight-line distance: {L:.4f} m")
    print(f"  v_max = {args.v_max} m/s, a_max = {args.a_max} m/s^2, "
          f"dt = {args.dt} s")

    # Plan
    traj = plan_trajectory(start_pose, end_pose,
                           v_max=args.v_max, a_max=args.a_max, dt=args.dt,
                           initial_joints=q_start)

    print(f"\n  Generated {len(traj['t'])} waypoints "
          f"(total motion time {traj['t'][-1]:.3f} s)")
    print(f"  Peak |v(t)|: {traj['speed'].max():.4f} m/s "
          f"(commanded v_max = {args.v_max} m/s)")
    print(f"  Joints at t=0:    {np.degrees(traj['joints'][0]).round(2)} deg")
    print(f"  Joints at t=end:  {np.degrees(traj['joints'][-1]).round(2)} deg")

    # Plot
    plot_trajectory(traj, save_dir=args.save, show=not args.no_show)


if __name__ == '__main__':
    main()
