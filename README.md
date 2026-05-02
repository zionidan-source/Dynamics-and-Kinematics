# UR5 Kinematics & Dynamics Project

> עבודה מסכמת – קינמטיקה ודינמיקה של רובוטים (362-1-4231)
> Spring 2026 · Ben-Gurion University of the Negev · Mechanical Engineering

This repository contains a complete simulation of the Universal Robots UR5 manipulator,
including analytical forward and inverse kinematics, workspace analysis, and motion
planning, all integrated with ROS2 Jazzy and Gazebo Harmonic.

ריפו זה כולל סימולציה מלאה של הזרוע הרובוטית UR5, כולל פיתוח אנליטי של קינמטיקה
ישירה והפוכה, ניתוח מרחב עבודה, ותכנון תנועה — הכל משולב עם ROS2 Jazzy ו-Gazebo Harmonic.

---

## 📋 Project Status

| Section | Topic | Status |
|---------|-------|--------|
| 4.1.1 | Cover page | ✅ |
| 4.1.2 | Introduction | ✅ |
| 4.1.3 | Robot description | ✅ |
| 4.1.4 | Forward kinematics | ✅ |
| 4.1.5 | Workspace | ✅ |
| 4.1.6 | Inverse kinematics | ✅ |
| 4.1.7 | Jacobian | 🚧 In progress |
| 4.1.8 | Statics | ⏳ Pending |
| 4.1.9 | Trajectory planning | ⏳ Pending |
| 4.1.10 | Full simulation | ⏳ Pending |
| 4.1.11 | Conclusions | ⏳ Pending |

---

## 🚀 Quick Start

### Prerequisites
- Ubuntu 24.04 LTS (Noble)
- ROS2 Jazzy Jalisco
- Python 3.12+
- Gazebo Harmonic

### 1. Clone and set up workspace

```bash
mkdir -p ~/ros2_workspaces/DandK_ws/src
cd ~/ros2_workspaces/DandK_ws/src

# Official UR packages
git clone -b jazzy https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git
git clone -b ros2 https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation.git

# This project
git clone https://github.com/<YOUR_USERNAME>/DandK_UR5_Project.git
cp -r DandK_UR5_Project/ur5_project .
```

### 2. Install dependencies

```bash
cd ~/ros2_workspaces/DandK_ws

sudo apt update
sudo apt install -y \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-ros-gz \
    ros-jazzy-xacro \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-rviz2

rosdep update
rosdep install --from-paths src --ignore-src -r -y
pip install matplotlib numpy --break-system-packages
```

### 3. Build

```bash
cd ~/ros2_workspaces/DandK_ws
colcon build --symlink-install
source install/setup.bash
```

---

## 🎮 Usage

### Launch the simulation

```bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5
```

### Verify forward kinematics against Gazebo

```bash
cd ~/ros2_workspaces/DandK_ws/src/ur5_project/ur5_project
python3 verify_fk.py
```

The robot moves through 5 test poses; the script compares analytical FK against Gazebo.

### Compute and visualize workspace

```bash
python3 workspace.py --density medium --save ../report_assets
```

### Test inverse kinematics

```bash
python3 inverse_kinematics.py
```

---

## 📁 Project Structure

```
DandK_UR5_Project/
├── README.md
├── LICENSE                            (MIT)
├── .gitignore
│
├── docs/
│   ├── UR5_Report_Hebrew.docx         Final report (Hebrew)
│   └── report_assets/                 Images for the report
│       ├── pose1_home.png             FK verification screenshots
│       ├── pose2_straight_up.png
│       ├── pose3_forward_reach.png
│       ├── pose4_side_fold.png
│       ├── pose5_twisted.png
│       ├── workspace_3d.png           Workspace visualizations
│       ├── workspace_top.png
│       ├── workspace_front.png
│       └── workspace_side.png
│
└── ur5_project/                       ROS2 Python package
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── resource/ur5_project
    └── ur5_project/
        ├── __init__.py
        ├── kinematics.py              Forward kinematics + DH parameters
        ├── inverse_kinematics.py      Analytical IK (8 solutions)
        ├── workspace.py               Workspace computation & plotting
        └── verify_fk.py               FK verification node
```

---

## 🧠 Module Overview

**`kinematics.py`** — UR5 DH parameter table, single-link DH matrix, full chain forward kinematics, helpers for position/rotation extraction and RPY conversion.

**`inverse_kinematics.py`** — Closed-form analytical IK returning up to 8 solutions per pose, nearest-neighbor solution selector, FK-based verification. Algorithm follows Ryan Keating's UR5 formulation.

**`workspace.py`** — Vectorized FK over 3.4M+ joint configurations using smart sampling. Generates 4 views (3D isometric, top, front, side).

**`verify_fk.py`** — ROS2 node that commands the UR5 to test poses, reads back actual `tool0` pose via tf2, and compares against analytical FK with errors reported in mm.

---

## 🐛 Troubleshooting

**`undefined symbol: HardwareComponentInterface`** — Version mismatch between `ros2_control` and `gz_ros2_control`. Fix: `sudo apt update && sudo apt upgrade -y`.

**Gazebo doesn't open** — GPU may not support OGRE2. Try the OGRE fallback rendering engine.

**`Aborted (core dumped)` at end of `verify_fk.py`** — Harmless rclpy threading cleanup issue. Data is already printed.

---

## 👥 Team

- Daniel — Forward/inverse kinematics, simulation integration, report
- (Partner name) — TBD

## 📜 License

MIT License — see [LICENSE](LICENSE).

## 🙏 Credits

- Inverse kinematics algorithm: Ryan Keating, Johns Hopkins University
- Official UR ROS2 packages: Universal Robots A/S
- Course instructor: Prof. Amir Shapiro, Ben-Gurion University

---

## 🎓 Course Information

**Course:** Kinematics and Dynamics of Robots (362-1-4231)
**Instructor:** Prof. Amir Shapiro
**Department:** Mechanical Engineering, Ben-Gurion University of the Negev
**Term:** Spring 2026
