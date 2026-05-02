#!/usr/bin/env python3
"""
workspace.py
============
Computes and visualizes the reachable workspace of the UR5 (assignment 4.1.5).

The workspace is the set of all 3D points the end-effector can reach by
varying the joint angles. We compute it by:
  1. Discretizing each joint's range into a grid
  2. Running forward kinematics for every combination (vectorized in NumPy)
  3. Collecting and plotting the (x, y, z) end-effector positions

Output: 4 figures
  - 3D isometric view
  - Top view (X-Y projection)
  - Front view (X-Z projection)
  - Side view (Y-Z projection)

Usage:
  python3 workspace.py                  # default density (~125k points)
  python3 workspace.py --density coarse # fast but rough (~10k points)
  python3 workspace.py --density fine   # slow but smooth (~1M points)

Author: Daniel
Course: Kinematics and Dynamics of Robots, Ben-Gurion University, 2026
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  needed for 3D projection

from kinematics import UR5_DH_PARAMS, T_BASE_CORRECTION


# ============================================================================
# Joint limits (UR5 official: each joint can rotate +/- 360 degrees)
# ============================================================================
JOINT_LIMITS = np.array([
    [-2 * np.pi, 2 * np.pi],   # shoulder_pan
    [-2 * np.pi, 2 * np.pi],   # shoulder_lift
    [-2 * np.pi, 2 * np.pi],   # elbow
    [-2 * np.pi, 2 * np.pi],   # wrist_1
    [-2 * np.pi, 2 * np.pi],   # wrist_2
    [-2 * np.pi, 2 * np.pi],   # wrist_3
])


# ============================================================================
# Density presets
# ============================================================================
# We use a "smart" sampling: dense grid on joints 1-3 (which dominate the
# end-effector position), coarse grid on joints 4-6 (the wrist, which mostly
# affects orientation, not position).
DENSITY_PRESETS = {
    'coarse':  {'n123': 8,  'n456': 3},   # ~14k points
    'medium':  {'n123': 12, 'n456': 4},   # ~110k points
    'fine':    {'n123': 18, 'n456': 5},   # ~730k points
    'ultra':   {'n123': 25, 'n456': 6},   # ~3.4M points - slow!
}


# ============================================================================
# Vectorized DH transformation
# ============================================================================
def dh_transform_batch(a, alpha, d, thetas):
    """
    Compute DH transformation matrices for many theta values at once.

    Parameters
    ----------
    a, alpha, d : float
        DH parameters (constants for a given joint).
    thetas : (N,) np.ndarray
        Array of joint angles to evaluate.

    Returns
    -------
    T : (N, 4, 4) np.ndarray
        Stack of N homogeneous transformation matrices.
    """
    N = len(thetas)
    ct = np.cos(thetas)
    st = np.sin(thetas)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.zeros((N, 4, 4))
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = a * ct
    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = a * st
    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = d
    T[:, 3, 3] = 1.0
    return T


# ============================================================================
# Vectorized forward kinematics for many joint configurations
# ============================================================================
def fk_batch(joint_grids):
    """
    Compute end-effector positions for a batch of joint configurations.

    Parameters
    ----------
    joint_grids : list of 6 arrays
        joint_grids[i] is a 1D array of angles to try for joint i.

    Returns
    -------
    positions : (N, 3) np.ndarray
        End-effector positions for all N = prod(len(g) for g in joint_grids)
        configurations.
    """
    # Build the meshgrid: every combination of joint angles
    grids = np.meshgrid(*joint_grids, indexing='ij')
    flat = [g.ravel() for g in grids]
    N = len(flat[0])
    print(f"  Evaluating {N:,} joint configurations...")

    # Start with the base correction matrix replicated N times
    T = np.broadcast_to(T_BASE_CORRECTION, (N, 4, 4)).copy()

    # Multiply through the 6 DH transforms
    for i in range(6):
        a, alpha, d, theta_offset = UR5_DH_PARAMS[i]
        thetas = flat[i] + theta_offset
        T_i = dh_transform_batch(a, alpha, d, thetas)
        T = T @ T_i  # batch matrix multiplication

    # Extract positions (last column, first 3 rows)
    positions = T[:, :3, 3]
    return positions


# ============================================================================
# Smart sampling: dense for j1-j3, coarse for j4-j6
# ============================================================================
def compute_workspace(n123=12, n456=4):
    """
    Compute the workspace point cloud.

    Parameters
    ----------
    n123 : int
        Number of samples per joint for joints 1, 2, 3.
    n456 : int
        Number of samples per joint for joints 4, 5, 6.

    Returns
    -------
    positions : (N, 3) np.ndarray
    """
    grids = []
    for i in range(6):
        n = n123 if i < 3 else n456
        grid = np.linspace(JOINT_LIMITS[i, 0], JOINT_LIMITS[i, 1], n)
        grids.append(grid)

    t_start = time.time()
    positions = fk_batch(grids)
    t_end = time.time()
    print(f"  Done in {t_end - t_start:.2f} seconds.")
    return positions


# ============================================================================
# Plotting
# ============================================================================
def plot_workspace(positions, save_dir=None):
    """Generate the 4 required workspace views."""
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Subsample for plotting (matplotlib chokes on millions of points)
    max_plot_points = 200_000
    if len(x) > max_plot_points:
        idx = np.random.choice(len(x), max_plot_points, replace=False)
        xp, yp, zp = x[idx], y[idx], z[idx]
        print(f"  Plotting random subsample of {max_plot_points:,} points "
              f"(of {len(x):,} total)")
    else:
        xp, yp, zp = x, y, z

    # Use small dots, slight transparency, blue color
    plot_kwargs = dict(s=0.5, c='steelblue', alpha=0.05)

    # ---------- Figure 1: 3D isometric ----------
    fig1 = plt.figure(figsize=(9, 7))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(xp, yp, zp, **plot_kwargs)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('UR5 Reachable Workspace - 3D Isometric View')
    ax1.set_box_aspect([1, 1, 1])
    # Mark the base
    ax1.scatter([0], [0], [0], c='red', s=100, marker='*', label='base_link')
    ax1.legend()

    # ---------- Figure 2: top view X-Y ----------
    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(xp, yp, **plot_kwargs)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Workspace - Top View (X-Y plane)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.scatter([0], [0], c='red', s=100, marker='*', label='base_link')
    ax2.legend()

    # ---------- Figure 3: front view X-Z ----------
    fig3 = plt.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(111)
    ax3.scatter(xp, zp, **plot_kwargs)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Workspace - Front View (X-Z plane)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.scatter([0], [0], c='red', s=100, marker='*', label='base_link')
    ax3.legend()

    # ---------- Figure 4: side view Y-Z ----------
    fig4 = plt.figure(figsize=(8, 8))
    ax4 = fig4.add_subplot(111)
    ax4.scatter(yp, zp, **plot_kwargs)
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Workspace - Side View (Y-Z plane)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.scatter([0], [0], c='red', s=100, marker='*', label='base_link')
    ax4.legend()

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(f"{save_dir}/workspace_3d.png", dpi=150, bbox_inches='tight')
        fig2.savefig(f"{save_dir}/workspace_top.png", dpi=150, bbox_inches='tight')
        fig3.savefig(f"{save_dir}/workspace_front.png", dpi=150, bbox_inches='tight')
        fig4.savefig(f"{save_dir}/workspace_side.png", dpi=150, bbox_inches='tight')
        print(f"  Figures saved to {save_dir}/")

    plt.show()


# ============================================================================
# Reachability statistics
# ============================================================================
def print_workspace_stats(positions):
    """Print summary statistics about the workspace."""
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    radius = np.linalg.norm(positions, axis=1)

    print("\n  WORKSPACE STATISTICS")
    print("  " + "-" * 40)
    print(f"  X range:       [{x.min():+.3f}, {x.max():+.3f}] m")
    print(f"  Y range:       [{y.min():+.3f}, {y.max():+.3f}] m")
    print(f"  Z range:       [{z.min():+.3f}, {z.max():+.3f}] m")
    print(f"  Max reach:     {radius.max():.3f} m")
    print(f"  Min reach:     {radius.min():.3f} m")
    print()


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='UR5 workspace visualization')
    parser.add_argument('--density', choices=list(DENSITY_PRESETS.keys()),
                        default='medium',
                        help='Sampling density (default: medium)')
    parser.add_argument('--save', type=str, default=None,
                        help='Directory to save figures (e.g., ../report_assets)')
    args = parser.parse_args()

    preset = DENSITY_PRESETS[args.density]
    print(f"\nUR5 WORKSPACE COMPUTATION")
    print(f"  Density preset: {args.density} "
          f"(joints 1-3: {preset['n123']} samples each, "
          f"joints 4-6: {preset['n456']} samples each)")
    print()

    positions = compute_workspace(**preset)
    print_workspace_stats(positions)
    plot_workspace(positions, save_dir=args.save)


if __name__ == '__main__':
    main()
