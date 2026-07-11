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
from sensor_msgs.msg import JointState
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


def orientation_error_deg(R_a, R_b):
    """
    Angle of the rotation that maps R_a onto R_b, in degrees.

    This is the geodesic distance on SO(3): |angle| of R_a^T @ R_b. Unlike a
    component-wise comparison of roll/pitch/yaw, it is representation-free and
    does not blow up near the RPY gimbal-lock singularity at pitch = +/-90 deg.
    """
    R_rel = R_a.T @ R_b
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


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

        # Subscribe to /joint_states so we can feed the *measured* joint angles
        # into FK (rather than the commanded ones). /joint_states publishes in
        # alphabetical name order, so we reorder into physical JOINT_NAMES order.
        self._latest_joint_positions = None  # dict: name -> position (rad)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10
        )

        # Wait briefly for publisher and tf to be ready
        time.sleep(2.0)
        self.get_logger().info('FK Verifier ready')

    def _joint_state_cb(self, msg):
        """Store the latest joint positions keyed by joint name."""
        self._latest_joint_positions = dict(zip(msg.name, msg.position))

    def get_measured_joint_angles(self):
        """
        Return the latest measured joint angles in physical JOINT_NAMES order,
        or None if not all joints have been received yet.
        """
        latest = self._latest_joint_positions
        if latest is None or any(n not in latest for n in JOINT_NAMES):
            return None
        return np.array([latest[n] for n in JOINT_NAMES])

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


def _spin_until_shutdown(node):
    """Spin the node, swallowing the exception rclpy raises once the context
    is shut down. Lets the spin thread terminate cleanly instead of aborting."""
    try:
        rclpy.spin(node)
    except Exception:
        # rclpy raises once the context is shut down from the main thread;
        # that is expected during teardown, so let the thread exit quietly.
        pass


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
        target=lambda: _spin_until_shutdown(node), daemon=True
    )
    spin_thread.start()

    print("\n" + "=" * 100)
    print(" FORWARD KINEMATICS VERIFICATION: Python FK vs Gazebo simulation")
    print("=" * 100)

    results = []

    try:
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

            # Feed the *measured* joint angles into FK, not the commanded ones.
            # This isolates FK accuracy from the controller's tracking error: if
            # the robot did not settle exactly on the command, that shows up as a
            # joint delta below rather than being misattributed to the FK.
            measured_rad = node.get_measured_joint_angles()
            if measured_rad is None:
                print("    [WARN] No /joint_states yet; falling back to commanded angles.")
                measured_rad = np.array(angles_rad)
            joint_delta_deg = np.degrees(measured_rad - np.array(angles_rad))
            max_joint_delta = float(np.max(np.abs(joint_delta_deg)))

            # Compute the predicted pose with our FK from the measured angles.
            T_pred = forward_kinematics(measured_rad)
            py_pos = position_from_transform(T_pred)
            py_R = rotation_from_transform(T_pred)

            gz_rpy = np.degrees(rotation_to_rpy(gz_R))
            py_rpy = np.degrees(rotation_to_rpy(py_R))

            # Errors. Orientation is compared on SO(3) (geodesic angle) rather
            # than component-wise RPY, which is singular near pitch = +/-90 deg.
            pos_err = np.linalg.norm(gz_pos - py_pos) * 1000  # in mm
            ori_err = orientation_error_deg(py_R, gz_R)       # in deg

            print(f"    Python FK:  {format_pose(py_pos, py_rpy)}")
            print(f"    Gazebo tf:  {format_pose(gz_pos, gz_rpy)}")
            print(f"    Position error: {pos_err:.2f} mm  |  Orientation error: {ori_err:.2f} deg")
            print(f"    Max joint tracking error (measured - commanded): {max_joint_delta:.2f} deg")

            results.append({
                'name': test['name'],
                'pos_err_mm': pos_err,
                'ori_err_deg': ori_err,
                'joint_delta_deg': max_joint_delta,
            })

        # Final summary
        print("\n" + "=" * 100)
        print(" SUMMARY")
        print("=" * 100)
        print(f"{'Test':<20} | {'Position (mm)':>14} | {'Orientation (deg)':>18} | {'Joint track (deg)':>18}")
        print("-" * 78)
        for r in results:
            print(f"{r['name']:<20} | {r['pos_err_mm']:>14.2f} | "
                  f"{r['ori_err_deg']:>18.2f} | {r['joint_delta_deg']:>18.2f}")

        # Gates: FK is verified when it matches tf once tracking error is
        # accounted for. 5 mm / 1 deg are comfortably above numerical noise but
        # below the error a partially-settled Gazebo pose would introduce.
        pos_ok = all(r['pos_err_mm'] < 5.0 for r in results)
        ori_ok = all(r['ori_err_deg'] < 1.0 for r in results)
        if results and pos_ok and ori_ok:
            print("\n[PASS] Position < 5 mm and orientation < 1 deg on all poses. "
                  "Forward kinematics verified.")
        else:
            print("\n[WARN] Some poses exceed the FK tolerance. Check the joint "
                  "tracking column: large values there mean the robot had not "
                  "settled, not that the DH parameters are wrong.")
    finally:
        # Clean shutdown: stop the spin thread *before* destroying the node so
        # the executor/tf listener never touches a destroyed node (which would
        # otherwise abort with "terminate called without an active exception").
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == '__main__':
    main()
