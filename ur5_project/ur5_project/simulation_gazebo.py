#!/usr/bin/env python3
"""
simulation_gazebo.py
====================
Live execution of the planned UR5 trajectory in Gazebo (assignment section
4.1.10.2.1 -- video of the robot motion).

This is the ROS2 counterpart of simulation.py. Whereas simulation.py produces
an MP4 of a stick-figure abstraction of the motion, this script streams the
actual joint trajectory to the UR5 controller in Gazebo and the arm executes
it in real time. You then screen-record the Gazebo window for the report.

Workflow
--------
  # Terminal 1: launch Gazebo with the UR5
  ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5

  # Terminal 2 (optional): start screen recording
  ./record_gazebo.sh

  # Terminal 3: send the trajectory
  python3 simulation_gazebo.py

The arm will:
  1. First move smoothly to the start pose of the planned trajectory
     (so the motion can begin from rest at a known initial configuration).
  2. Pause briefly so the recording captures a stable starting frame.
  3. Execute the full planned trajectory in real time (single pass, no loop).
  4. Stop at the end pose.

After the motion completes, stop the recording and the script exits.

Author: Daniel
Course: Kinematics and Dynamics of Robots, Ben-Gurion University, 2026
"""

import time
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg

from kinematics import forward_kinematics, rotation_to_rpy, JOINT_NAMES
from trajectory import plan_trajectory


# ============================================================================
# Demo motion parameters (same as simulation.py for consistency)
# ============================================================================
# Two non-singular joint configurations whose Cartesian poses we use as
# the start and end of the linear-Cartesian trajectory.
Q_START_REF = np.array([0.0,     -np.pi/3, -np.pi/3, -np.pi/3, np.pi/2, 0.0    ])
Q_END_REF   = np.array([np.pi/6, -np.pi/4, -np.pi/4, -np.pi/4, np.pi/2, np.pi/6])

V_MAX        = 0.25
A_MAX        = 0.5
DT           = 0.01

PREMOVE_DUR  = 4.0    # seconds to move from current Gazebo pose to the
                      # planned trajectory's start configuration
HOLD_BEFORE  = 1.5    # pause at start before executing the motion
                      # (gives the recording a clean opening frame)


class GazeboTrajectoryRunner(Node):
    """ROS2 node that builds and publishes a full JointTrajectory."""

    def __init__(self):
        super().__init__('ur5_trajectory_runner')
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )
        time.sleep(2.0)  # wait for the publisher to be discovered
        self.get_logger().info('Gazebo trajectory runner ready.')

    def send_to_pose(self, joint_angles, duration_sec):
        """
        Send a one-point trajectory commanding the arm to a target pose
        over the given duration. Used for the initial pre-move.
        """
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = list(joint_angles)
        pt.time_from_start = self._duration(duration_sec)
        msg.points = [pt]
        self.traj_pub.publish(msg)

    def send_full_trajectory(self, joints, t, hold_initial=HOLD_BEFORE):
        """
        Send the full N-point trajectory as a single JointTrajectory.

        The first point is duplicated at t=0 and t=hold_initial to make
        the arm pause at the start pose, giving the recording a clean
        opening frame. The remaining N-1 points are then sent in real time
        (their original t values shifted by hold_initial).
        """
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        msg.points = []

        # Hold at start pose for HOLD_BEFORE seconds
        pt0 = JointTrajectoryPoint()
        pt0.positions = list(joints[0])
        pt0.time_from_start = self._duration(hold_initial)
        msg.points.append(pt0)

        # Then the rest of the trajectory, shifted in time
        # (skip the very first point since it's already in pt0)
        for i in range(1, len(t)):
            pt = JointTrajectoryPoint()
            pt.positions = list(joints[i])
            # Optional: include velocities for smoother controller tracking
            # (computed by finite differences). Comment out if your driver
            # doesn't accept velocities in the trajectory message.
            if i < len(t) - 1:
                vel = (joints[i + 1] - joints[i - 1]) / (t[i + 1] - t[i - 1])
            else:
                vel = (joints[i] - joints[i - 1]) / (t[i] - t[i - 1])
            pt.velocities = list(vel)
            pt.time_from_start = self._duration(hold_initial + t[i])
            msg.points.append(pt)

        self.traj_pub.publish(msg)
        return hold_initial + t[-1]

    @staticmethod
    def _duration(seconds):
        d = DurationMsg()
        d.sec = int(seconds)
        d.nanosec = int((seconds - int(seconds)) * 1e9)
        return d


def main():
    rclpy.init()
    node = GazeboTrajectoryRunner()

    # --- Plan the trajectory using the same routine as simulation.py ---
    T0 = forward_kinematics(Q_START_REF)
    T1 = forward_kinematics(Q_END_REF)
    start_pose = np.concatenate([T0[:3, 3], rotation_to_rpy(T0[:3, :3])])
    end_pose   = np.concatenate([T1[:3, 3], rotation_to_rpy(T1[:3, :3])])

    print("=" * 72)
    print(" UR5 LIVE GAZEBO SIMULATION")
    print("=" * 72)
    print(f"  Start xyz = {start_pose[:3].round(4)}, "
          f"rpy = {np.degrees(start_pose[3:]).round(1)} deg")
    print(f"  End   xyz = {end_pose[:3].round(4)}, "
          f"rpy = {np.degrees(end_pose[3:]).round(1)} deg")
    print(f"  v_max = {V_MAX} m/s, a_max = {A_MAX} m/s^2, dt = {DT} s")

    traj = plan_trajectory(start_pose, end_pose,
                           v_max=V_MAX, a_max=A_MAX, dt=DT,
                           initial_joints=Q_START_REF)
    T_motion = traj['t'][-1]
    print(f"  Planned: {len(traj['t'])} waypoints, {T_motion:.3f} s motion.")

    # --- Step 1: pre-move to the start configuration ---
    # The UR5 in Gazebo typically wakes up in some default pose; we need
    # to move it to our trajectory's q_start before the motion begins.
    print(f"\n  [1/2] Pre-moving to start pose over {PREMOVE_DUR} s...")
    node.send_to_pose(traj['joints'][0], duration_sec=PREMOVE_DUR)
    time.sleep(PREMOVE_DUR + 0.5)

    # --- Step 2: execute the full trajectory ---
    total = node.send_full_trajectory(traj['joints'], traj['t'],
                                      hold_initial=HOLD_BEFORE)
    print(f"  [2/2] Streaming full trajectory "
          f"(hold {HOLD_BEFORE}s + motion {T_motion:.2f}s = {total:.2f}s)...")
    time.sleep(total + 1.0)

    print("\n  Motion complete. Stopping the recording is up to you.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
