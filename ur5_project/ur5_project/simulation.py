#!/usr/bin/env python3
"""
ur5_project/simulation.py
=========================
Full demo simulation for the UR5 (assignment section 4.1.10).

This module consumes the trajectory produced by trajectory.plan_trajectory()
and generates every artifact required by section 4.1.10.2:

    .1  An MP4 animation of the robot following the trajectory
    .2  Stick diagrams at five poses along the motion
    .3  End-effector position and orientation vs time
    .4  Joint angles vs time
    .5  End-effector speed: numerical vs analytical (Jacobian)
    .6  Joint torques required to carry a 3 kg payload along the trajectory

Design note (deliberate decoupling):
    This module reads only the output dict of plan_trajectory() -- it does
    not know or care how the trajectory was produced. If we later change
    the planner (e.g., switch to joint-space planning or numerical IK
    fallback near singularities), this file requires no changes as long
    as the dict keys 't', 'cartesian', 'joints', 'speed', 's' remain
    consistent.

Author: Daniel
Course: Kinematics and Dynamics of Robots, Ben-Gurion University, 2026
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pathlib import Path

from ur5_project.kinematics import forward_kinematics, T_BASE_CORRECTION
from ur5_project.jacobian import compute_jacobian, compute_statics
from ur5_project.trajectory import plan_trajectory, compute_acceleration

# This file lives at <REPO>/ur5_project/ur5_project/simulation.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = str(REPO_ROOT / "docs" / "report_assets")


# ============================================================================
# Robot geometry helpers: extract all joint origins in base_link for plotting
# ============================================================================
def joint_positions(joint_angles):
    """
    Return the 3D positions of (base, joint_1, ..., joint_6 = tool0) in the
    base_link frame, given a joint configuration. Used to draw the stick
    diagram of the arm.

    Parameters
    ----------
    joint_angles : array-like of length 6

    Returns
    -------
    pts : (7, 3) np.ndarray
        Row 0 is the origin of base_link, rows 1..6 are the origins of
        the six DH frames (the last being tool0).
    """
    _, all_T = forward_kinematics(joint_angles, return_all=True)
    pts = np.zeros((7, 3))
    pts[0] = T_BASE_CORRECTION[:3, 3]  # base_link origin
    for i, T in enumerate(all_T):
        pts[i + 1] = T[:3, 3]
    return pts


# ============================================================================
# Numerical and analytical end-effector velocities
# ============================================================================
def compute_velocities(traj):
    """
    Compute end-effector linear velocity along the trajectory in two ways:

    Numerical: central-difference derivative of the Cartesian position
               (first row of `cartesian`, columns 0-2) with respect to time.
    Analytical: J(q) * q_dot at each waypoint, where q_dot is itself the
                numerical derivative of the joint trajectory.

    Both are reported as magnitudes |v|.

    Parameters
    ----------
    traj : dict
        The output of plan_trajectory().

    Returns
    -------
    v_num_mag : (N,) np.ndarray  -- numerical |v(t)|
    v_an_mag  : (N,) np.ndarray  -- analytical |v(t)|
    """
    t   = traj['t']
    pos = traj['cartesian'][:, :3]   # (N, 3)
    q   = traj['joints']             # (N, 6)
    N   = len(t)

    # Numerical: central differences (forward/backward at the boundaries)
    v_num = np.zeros_like(pos)
    v_num[1:-1] = (pos[2:] - pos[:-2]) / (t[2:] - t[:-2])[:, None]
    v_num[0]    = (pos[1]  - pos[0])   / (t[1]  - t[0])
    v_num[-1]   = (pos[-1] - pos[-2])  / (t[-1] - t[-2])

    # Joint velocities: central differences on q
    qdot = np.zeros_like(q)
    qdot[1:-1] = (q[2:] - q[:-2]) / (t[2:] - t[:-2])[:, None]
    qdot[0]    = (q[1]  - q[0])   / (t[1]  - t[0])
    qdot[-1]   = (q[-1] - q[-2])  / (t[-1] - t[-2])

    # Analytical: v_ee = top 3 rows of J(q) times qdot
    v_an = np.zeros_like(pos)
    for i in range(N):
        J = compute_jacobian(q[i])
        v_an[i] = J[:3] @ qdot[i]

    return np.linalg.norm(v_num, axis=1), np.linalg.norm(v_an, axis=1)


# ============================================================================
# Joint torques along the trajectory
# ============================================================================
def compute_torques(traj, payload_mass):
    """
    Compute the static motor torques required to hold a payload of mass M
    at the end-effector, evaluated at every waypoint along the trajectory.
    This is a quasi-static approximation: it neglects inertial torques
    arising from acceleration of the links themselves.

    Parameters
    ----------
    traj : dict
    payload_mass : float
        Payload mass in kg.

    Returns
    -------
    tau : (N, 6) np.ndarray
        Joint torques in Nm, columns in physical joint order.
    """
    q = traj['joints']
    N = len(q)
    tau = np.zeros((N, 6))
    for i in range(N):
        tau[i] = compute_statics(q[i], payload_mass)
    return tau


# ============================================================================
# Plots: stick diagrams (4.1.10.2.2)
# ============================================================================
def plot_stick_diagrams(traj, save_path=None):
    """
    Plot the robot as a 'stick figure' at five poses along the trajectory:
    start, 25%, 50%, 75%, and end. All five are overlaid on a single 3D
    axis so the progression is visible.
    """
    N = len(traj['t'])
    indices = [0, N // 4, N // 2, (3 * N) // 4, N - 1]
    labels  = ['t = 0 (start)', 't = 25%', 't = 50%', 't = 75%', 't = end']
    colors  = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Also plot the EE path as a faint reference line
    cart = traj['cartesian'][:, :3]
    ax.plot(cart[:, 0], cart[:, 1], cart[:, 2],
            '--', color='gray', alpha=0.5, linewidth=1, label='EE path')

    for idx, label, color in zip(indices, labels, colors):
        q = traj['joints'][idx]
        pts = joint_positions(q)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                '-o', color=color, linewidth=2.5, markersize=5, label=label)

    # Mark the base
    ax.scatter([0], [0], [0], c='black', s=80, marker='s', label='base_link')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Stick diagrams of UR5 at five points along the motion')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_box_aspect([1, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Plots: end-effector position and orientation vs time (4.1.10.2.3)
# ============================================================================
def plot_ee_pose(traj, save_path=None):
    """End-effector X/Y/Z position and Roll/Pitch/Yaw orientation vs time."""
    t = traj['t']
    cart = traj['cartesian']

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Position
    axes[0].plot(t, cart[:, 0], label='X', linewidth=2)
    axes[0].plot(t, cart[:, 1], label='Y', linewidth=2)
    axes[0].plot(t, cart[:, 2], label='Z', linewidth=2)
    axes[0].set_ylabel('position (m)')
    axes[0].set_title('End-effector position vs time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Orientation (degrees)
    axes[1].plot(t, np.degrees(cart[:, 3]), label='Roll',  linewidth=2)
    axes[1].plot(t, np.degrees(cart[:, 4]), label='Pitch', linewidth=2)
    axes[1].plot(t, np.degrees(cart[:, 5]), label='Yaw',   linewidth=2)
    axes[1].set_ylabel('orientation (deg)')
    axes[1].set_xlabel('time (s)')
    axes[1].set_title('End-effector orientation (RPY) vs time')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Plots: joint angles vs time (4.1.10.2.4)
# ============================================================================
def plot_joint_angles(traj, save_path=None):
    """All six joint angles vs time (in degrees)."""
    t = traj['t']
    q = traj['joints']
    labels = ['shoulder_pan', 'shoulder_lift', 'elbow',
              'wrist_1', 'wrist_2', 'wrist_3']

    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(6):
        ax.plot(t, np.degrees(q[:, i]), linewidth=1.5, label=labels[i])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('joint angle (deg)')
    ax.set_title('Joint angles along the trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Plots: numerical vs analytical velocity (4.1.10.2.5)
# ============================================================================
def plot_velocity_comparison(traj, save_path=None):
    """
    Plot end-effector speed vs time computed two ways:
      * Numerical: dp/dt via central differences on the Cartesian position
      * Analytical: |J(q) * q_dot|

    They should overlay almost exactly; any difference is a result of
    finite-difference noise (because q_dot is itself estimated numerically).
    """
    t = traj['t']
    v_num_mag, v_an_mag = compute_velocities(traj)
    err = np.abs(v_num_mag - v_an_mag)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

    axes[0].plot(t, v_num_mag, '-',  linewidth=2.5, label='numerical (dp/dt)')
    axes[0].plot(t, v_an_mag,  '--', linewidth=2,   label='analytical (J·q̇)')
    axes[0].set_ylabel('|v(t)| (m/s)')
    axes[0].set_title('End-effector speed: numerical vs analytical')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(t, np.maximum(err, 1e-12), color='crimson', linewidth=1.5)
    axes[1].set_ylabel('|err| (m/s)')
    axes[1].set_xlabel('time (s)')
    axes[1].set_title('Absolute difference (log scale)')
    axes[1].grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Plots: torque profiles (4.1.10.2.6)
# ============================================================================
def plot_torques(traj, payload_mass, save_path=None):
    """Joint torques required to hold a payload of mass M along the motion."""
    t = traj['t']
    tau = compute_torques(traj, payload_mass)
    labels = ['shoulder_pan', 'shoulder_lift', 'elbow',
              'wrist_1', 'wrist_2', 'wrist_3']

    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(6):
        ax.plot(t, tau[:, i], linewidth=1.5, label=labels[i])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('joint torque (Nm)')
    ax.set_title(f'Joint torques required to hold a {payload_mass} kg payload')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Plots: acceleration profiles (joint-space + Cartesian)
# ============================================================================
def plot_acceleration(traj, save_path=None):
    """
    Plot the acceleration of the motion in both spaces vs time:
      * Joint-space: angular acceleration of all six joints (rad/s^2)
      * Work-space:  end-effector linear acceleration magnitude |a(t)| (m/s^2)

    Both are numerical central-difference derivatives (see
    trajectory.compute_acceleration), so the curves expose the finite-
    difference noise inherent in twice-differentiating a sampled path,
    just like plot_velocity_comparison() does for the velocity.
    """
    t = traj['t']
    a_joint, a_cart_mag = compute_acceleration(traj)
    labels = ['shoulder_pan', 'shoulder_lift', 'elbow',
              'wrist_1', 'wrist_2', 'wrist_3']

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for i in range(6):
        axes[0].plot(t, a_joint[:, i], linewidth=1.5, label=labels[i])
    axes[0].set_ylabel('joint accel. (rad/s²)')
    axes[0].set_title('Joint angular acceleration along the trajectory')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', ncol=2)

    axes[1].plot(t, a_cart_mag, '-', color='darkorange', linewidth=2)
    axes[1].fill_between(t, 0, a_cart_mag, color='darkorange', alpha=0.15)
    axes[1].set_ylabel('|a(t)| (m/s²)')
    axes[1].set_xlabel('time (s)')
    axes[1].set_title('End-effector linear acceleration (world frame)')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Animation: MP4 of the robot moving along the trajectory (4.1.10.2.1)
# ============================================================================
def animate_motion(traj, save_path, fps=30, slowdown=3.0):
    """
    Generate an MP4 animation of the UR5 following the trajectory.

    Parameters
    ----------
    traj : dict
        The output of plan_trajectory().
    save_path : str
        Path to write the .mp4 file.
    fps : int, optional
        Output video frame rate (default 30).
    slowdown : float, optional
        Playback slowdown factor relative to real-time motion (default 3.0:
        the video plays the motion at 1/3 of its real speed for clarity).
    """
    N = len(traj['t'])
    T_motion_real = traj['t'][-1]                  # real-time motion duration
    T_video       = T_motion_real * slowdown       # desired video duration
    n_frames      = int(np.ceil(T_video * fps))

    # Pick which waypoint to render at each video frame. Each video frame
    # samples a waypoint linearly from the trajectory.
    frame_indices = np.linspace(0, N - 1, n_frames).astype(int)

    # Pre-compute all link points to keep the animation loop cheap
    print(f"  Pre-computing {N} robot poses for animation...")
    all_link_pts = np.array([joint_positions(traj['joints'][i]) for i in range(N)])

    # Fixed axis limits: union of base, EE path, and arm reach
    cart = traj['cartesian'][:, :3]
    margin = 0.2
    x_min, x_max = -0.8, 0.8
    y_min, y_max = -0.8, 0.8
    z_min, z_max = -0.2, 1.2

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Persistent elements (drawn once)
    ax.plot(cart[:, 0], cart[:, 1], cart[:, 2],
            '--', color='gray', alpha=0.4, linewidth=1, label='planned EE path')
    ax.scatter([0], [0], [0], c='black', s=80, marker='s', label='base_link')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect([1, 1, 1 * (z_max - z_min) / (x_max - x_min)])
    ax.legend(loc='upper left', fontsize=9)

    # Updateable elements
    (arm_line,) = ax.plot([], [], [], '-o', color='steelblue',
                          linewidth=3, markersize=6)
    (trace_line,) = ax.plot([], [], [], '-', color='crimson',
                            linewidth=1.5, alpha=0.7)
    title = ax.set_title('UR5 motion — t = 0.000 s')

    trace_xyz = []

    def init():
        arm_line.set_data([], [])
        arm_line.set_3d_properties([])
        trace_line.set_data([], [])
        trace_line.set_3d_properties([])
        return arm_line, trace_line, title

    def update(frame_no):
        idx = frame_indices[frame_no]
        pts = all_link_pts[idx]
        arm_line.set_data(pts[:, 0], pts[:, 1])
        arm_line.set_3d_properties(pts[:, 2])

        trace_xyz.append(pts[-1].copy())
        tr = np.array(trace_xyz)
        trace_line.set_data(tr[:, 0], tr[:, 1])
        trace_line.set_3d_properties(tr[:, 2])

        title.set_text(f'UR5 motion — t = {traj["t"][idx]:6.3f} s '
                       f'(playback {slowdown:.1f}x slower)')
        return arm_line, trace_line, title

    print(f"  Rendering {n_frames} frames at {fps} fps "
          f"({T_video:.2f} s of video for {T_motion_real:.2f} s of motion)...")
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=1000 / fps, blit=False)

    writer = FFMpegWriter(fps=fps, bitrate=2400,
                          codec='libx264',
                          extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(save_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Animation saved to {save_path}")


# ============================================================================
# Main: orchestrate everything
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='UR5 full simulation: stick diagrams, plots, MP4 video'
    )
    parser.add_argument('--payload', type=float, default=3.0,
                        help='Payload mass in kg (default 3.0)')
    parser.add_argument('--v_max', type=float, default=0.25,
                        help='Peak Cartesian speed (m/s)')
    parser.add_argument('--a_max', type=float, default=0.5,
                        help='Cartesian acceleration (m/s^2)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Trajectory sampling time step (s)')
    parser.add_argument('--save', type=str,
                        default=DEFAULT_SAVE_DIR,
                        help=f'Output directory for plots and MP4 '
                             f'(default: {DEFAULT_SAVE_DIR})')
    parser.add_argument('--fps', type=int, default=30,
                        help='MP4 frame rate (default 30)')
    parser.add_argument('--slowdown', type=float, default=3.0,
                        help='Video playback slowdown factor (default 3.0x)')
    parser.add_argument('--no_video', action='store_true',
                        help='Skip MP4 rendering (just plots)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plots interactively')
    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)

    # ----- Plan the demo trajectory (same non-singular poses as 4.1.9) -----
    q_start = np.array([0.0,      -np.pi/3, -np.pi/3, -np.pi/3, np.pi/2, 0.0    ])
    q_end   = np.array([np.pi/6,  -np.pi/4, -np.pi/4, -np.pi/4, np.pi/2, np.pi/6])

    # Convert to Cartesian start/end via FK + RPY extraction
    from ur5_project.kinematics import rotation_to_rpy
    T0 = forward_kinematics(q_start); T1 = forward_kinematics(q_end)
    start_pose = np.concatenate([T0[:3, 3], rotation_to_rpy(T0[:3, :3])])
    end_pose   = np.concatenate([T1[:3, 3], rotation_to_rpy(T1[:3, :3])])

    print("=" * 72)
    print(" UR5 FULL SIMULATION (section 4.1.10)")
    print("=" * 72)
    print(f"  Planning Cartesian linear trajectory:")
    print(f"    start xyz= {start_pose[:3].round(4)}, "
          f"rpy= {np.degrees(start_pose[3:]).round(1)} deg")
    print(f"    end   xyz= {end_pose[:3].round(4)}, "
          f"rpy= {np.degrees(end_pose[3:]).round(1)} deg")
    print(f"  v_max={args.v_max} m/s, a_max={args.a_max} m/s^2, "
          f"dt={args.dt} s, payload={args.payload} kg")

    traj = plan_trajectory(start_pose, end_pose,
                           v_max=args.v_max, a_max=args.a_max, dt=args.dt,
                           initial_joints=q_start)
    print(f"  Trajectory: {len(traj['t'])} waypoints, "
          f"duration {traj['t'][-1]:.3f} s\n")

    # ----- Generate each plot (4.1.10.2.2 through .6) -----
    print("Generating plots...")
    plot_stick_diagrams(traj,
        save_path=os.path.join(args.save, 'sim_stick_diagrams.png'))
    plot_ee_pose(traj,
        save_path=os.path.join(args.save, 'sim_ee_pose.png'))
    plot_joint_angles(traj,
        save_path=os.path.join(args.save, 'sim_joint_angles.png'))
    plot_velocity_comparison(traj,
        save_path=os.path.join(args.save, 'sim_velocity_compare.png'))
    plot_torques(traj, args.payload,
        save_path=os.path.join(args.save, 'sim_torques.png'))
    plot_acceleration(traj,
        save_path=os.path.join(args.save, 'sim_acceleration.png'))
    print(f"  Plots saved to {args.save}/")

    # ----- Quick numerical summary -----
    v_num, v_an = compute_velocities(traj)
    tau = compute_torques(traj, args.payload)
    print(f"\n  Peak |v_num - v_analytic|:    {np.max(np.abs(v_num - v_an)):.3e} m/s")
    print(f"  Peak |shoulder_lift| torque:  {np.max(np.abs(tau[:, 1])):.2f} Nm")
    print(f"  Peak |elbow|         torque:  {np.max(np.abs(tau[:, 2])):.2f} Nm")

    # ----- MP4 animation (4.1.10.2.1) -----
    if not args.no_video:
        print("\nRendering MP4 animation...")
        animate_motion(traj,
                       save_path=os.path.join(args.save, 'sim_motion.mp4'),
                       fps=args.fps, slowdown=args.slowdown)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
