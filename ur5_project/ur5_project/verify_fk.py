#!/usr/bin/env python3
"""
verify_fk.py
============
Verification of the forward kinematics module against Gazebo simulation.

Procedure for each test pose:
  1. Publish a JointTrajectory message commanding the UR5 to that pose
  2. Wait for the robot to settle
  3. Read the actual tool0 pose from the tf2 tree
  4. Compute the predicted pose using our Python FK
  5. Print a comparison

Usage:
  # Terminal 1: launch the Gazebo simulation
  ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5

  # Terminal 2: run this script
  ros2 run ur5_project verify_fk
"""

import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException

# Add the parent directory so we can import kinematics
sys.path.insert(0, '.')
from ur5_project.kinematics import (
    forward_kinematics,
    position_from_transform,
    rotation_from_transform,
    rotation_to_rpy,
    JOINT_NAMES,
)


# ============================================================================
# The 5 test configurations (joint angles in degrees, will convert to rad)
# ============================================================================
TEST_POSES_DEG = [
    {
        'name': 'Home',
        'angles': [0,    -90,   0,    -90,   0,    0],
        'description': 'Default Gazebo home pose, arm forward',
    },
    {
        'name': 'Twisted Wrist',
        'angles': [0,    -90,   0,     0,    0,    0],
        'description': 'Wrist rotated 90 deg',
    },
    {
    'name': '45 Degree Bend',
    'angles': [0,   -45,  -45,    0,   90,    0],
    'description': 'Arm bent up at 45 deg, wrist rotated 90 deg',
    },
    {
        'name': 'Side fold',
        'angles': [90,  -90,   90,   -90,    0,    0],
        'description': 'Base rotated 90 deg, elbow bent',
    },
    {
        'name': 'Twisted',
        'angles': [45,  -60,   90,  -120,   90,   45],
        'description': 'Mixed angles, exercises all joints',
    },
]


class FKVerifier(Node):
    """ROS2 node for sending poses and reading back tf transforms."""

    def __init__(self):
        super().__init__('fk_verifier')

        # Publisher for joint trajectory commands.
        # The UR ROS2 driver uses scaled_joint_trajectory_controller.
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        # tf2 buffer and listener for reading current pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Wait briefly for publisher and tf to be ready
        time.sleep(2.0)
        self.get_logger().info('FK Verifier ready')

    def send_joint_command(self, joint_angles_rad, duration_sec=4.0):
        """Publish a joint trajectory commanding the given pose."""
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES  # physical order

        point = JointTrajectoryPoint()
        point.positions = list(joint_angles_rad)
        point.time_from_start = DurationMsg(
            sec=int(duration_sec),
            nanosec=int((duration_sec - int(duration_sec)) * 1e9)
        )
        msg.points = [point]

        self.traj_pub.publish(msg)

    def get_current_tool0_pose(self, timeout_sec=2.0):
        """Look up the current pose of tool0 relative to base_link via tf2."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'tool0',
                rclpy.time.Time(),
                Duration(seconds=timeout_sec)
            )
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().error(f'tf lookup failed: {e}')
            return None, None

        t = transform.transform.translation
        q = transform.transform.rotation
        position = np.array([t.x, t.y, t.z])

        # Convert quaternion to rotation matrix
        # qx, qy, qz, qw -> R
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ])
        return position, R


def format_pose(position, rpy_deg):
    """Format a pose for table printing."""
    return (f"[{position[0]:+7.4f}, {position[1]:+7.4f}, {position[2]:+7.4f}] m  |  "
            f"RPY [{rpy_deg[0]:+7.2f}, {rpy_deg[1]:+7.2f}, {rpy_deg[2]:+7.2f}] deg")


def main():
    rclpy.init()
    node = FKVerifier()

    # Process tf for a couple of seconds to fill the buffer
    end_time = time.time() + 2.0
    while time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.1)

    # Spin in a separate thread so tf updates while we work
    import threading
    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(node), daemon=True
    )
    spin_thread.start()

    print("\n" + "=" * 100)
    print(" FORWARD KINEMATICS VERIFICATION: Python FK vs Gazebo simulation")
    print("=" * 100)

    results = []

    for test in TEST_POSES_DEG:
        print(f"\n--- Test: {test['name']} ---")
        print(f"    {test['description']}")
        angles_deg = test['angles']
        angles_rad = [np.radians(a) for a in angles_deg]
        print(f"    Joint angles (deg): {angles_deg}")

        # Send the command
        node.send_joint_command(angles_rad, duration_sec=4.0)

        # Wait for the robot to reach the pose. 4 seconds command + 1 second slack.
        time.sleep(5.0)

        # Read the actual pose from Gazebo
        gz_pos, gz_R = node.get_current_tool0_pose()
        if gz_pos is None:
            print("    [ERROR] Could not read tf transform. Skipping.")
            continue

        # Compute the predicted pose with our FK
        T_pred = forward_kinematics(angles_rad)
        py_pos = position_from_transform(T_pred)
        py_R = rotation_from_transform(T_pred)

        gz_rpy = np.degrees(rotation_to_rpy(gz_R))
        py_rpy = np.degrees(rotation_to_rpy(py_R))

        # Compute error
        pos_err = np.linalg.norm(gz_pos - py_pos) * 1000  # in mm

        print(f"    Python FK:  {format_pose(py_pos, py_rpy)}")
        print(f"    Gazebo tf:  {format_pose(gz_pos, gz_rpy)}")
        print(f"    Position error: {pos_err:.2f} mm")

        results.append({
            'name': test['name'],
            'pos_err_mm': pos_err,
        })

    # Final summary
    print("\n" + "=" * 100)
    print(" SUMMARY")
    print("=" * 100)
    print(f"{'Test':<20} | {'Position error (mm)':>20}")
    print("-" * 45)
    for r in results:
        print(f"{r['name']:<20} | {r['pos_err_mm']:>20.2f}")

    if all(r['pos_err_mm'] < 5.0 for r in results):
        print("\n[PASS] All errors below 5 mm. Forward kinematics verified.")
    else:
        print("\n[WARN] Some errors above 5 mm. Check DH parameters.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
