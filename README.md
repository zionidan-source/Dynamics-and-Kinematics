# UR5 Kinematics, Dynamics, and Trajectory Simulation

Final project for the course **Kinematics and Dynamics of Robots** (362-1-4231),
Ben-Gurion University of the Negev, Spring 2026.

**Authors:** Daniel and Itai
**Instructor:** Prof. Amir Shapiro

A full simulation of the Universal Robots UR5 6-DOF arm, with kinematics,
Jacobian, statics, and trajectory planning all derived analytically and
exercised live in Gazebo Harmonic.

## Architecture

The package is structured as three layers:

```
                     ┌─────────────────────────────────┐
   library layer →   │  kinematics  inverse_kinematics │
                     │  jacobian    trajectory         │
                     └────────────────┬────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
   tool layer    →   │  workspace      verify_fk       │
                     │  test_trajectory                │
                     └────────────────┬────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
   driver layer  →   │  simulation     simulation_gazebo │
                     └─────────────────────────────────┘
```

### Library modules (no `main()`, imported only)

| Module                 | Purpose                                                |
|------------------------|--------------------------------------------------------|
| `kinematics.py`        | DH parameters, `forward_kinematics()`, `T_BASE_CORRECTION` |
| `inverse_kinematics.py`| Closed-form IK; returns up to 8 filtered solutions     |
| `jacobian.py`          | Analytical 6×6 Jacobian, numerical Jacobian, `compute_statics()` |
| `trajectory.py`        | `plan_trajectory()` — Cartesian-linear + trapezoidal v |

### Executable nodes (via `ros2 run`)

| Node                  | Purpose                                            | Needs Gazebo? |
|-----------------------|----------------------------------------------------|---------------|
| `workspace`           | Compute and plot the reachable workspace           | No            |
| `trajectory`          | Plan one trajectory and plot it (FK/joints/speed)  | No            |
| `simulation`          | Full offline simulation with stick diagrams + MP4  | No            |
| `test_trajectory`     | Validate a pair of start/end poses before running  | No            |
| `verify_fk`           | Validate forward kinematics against Gazebo via tf2 | **Yes**       |
| `simulation_gazebo`   | Execute the planned trajectory live in Gazebo      | **Yes**       |

---

## Prerequisites

This project assumes **Ubuntu 24.04 LTS** with **ROS2 Jazzy Jalisco** and
**Gazebo Harmonic** already installed and working. Follow the official guides
at <https://docs.ros.org/en/jazzy/Installation.html> and
<https://gazebosim.org/docs/harmonic/install_ubuntu/> if you do not have these.

Verify your installation:

```bash
ros2 --version          # should print "ros2 cli version ..."
gz sim --version        # should print Gazebo's version
```

Python 3.12 ships with Ubuntu 24.04. The following Python packages are needed:

```bash
pip install numpy matplotlib scipy --break-system-packages
```

(`--break-system-packages` is required on Ubuntu 24.04 because of PEP 668.
Alternatively use a virtual environment.)

For MP4 rendering, `ffmpeg` must also be available:

```bash
sudo apt install ffmpeg
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zionidan-source/Dynamics-and-Kinematics.git ~/projects/DandK_UR5_Project
```

### 2. Set up the ROS2 workspace

The repository contains the `ur5_project` ROS2 package. It must be made
visible to colcon by placing it inside a workspace's `src/` directory. The
recommended approach is to **symlink** the package into your workspace, so
edits made to the repo are immediately reflected:

```bash
mkdir -p ~/ros2_workspaces/DandK_ws/src
ln -s ~/projects/DandK_UR5_Project/ur5_project ~/ros2_workspaces/DandK_ws/src/ur5_project
```

You also need Universal Robots' official ROS2 description and Gazebo
simulation packages alongside it:

```bash
cd ~/ros2_workspaces/DandK_ws/src
git clone -b jazzy  https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git
git clone -b ros2   https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation.git
```

### 3. Build the workspace

```bash
cd ~/ros2_workspaces/DandK_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

### 4. Source the overlay

This must be done in every new shell where you want to run package commands.
Consider adding it to your `~/.bashrc`:

```bash
source ~/ros2_workspaces/DandK_ws/install/setup.bash
```

### 5. Verify the install

```bash
ros2 pkg list | grep ur5_project          # should print: ur5_project
ros2 pkg executables ur5_project          # should list all 6 nodes
```

---

## Running the simulation

All commands below assume you have sourced the workspace overlay
(`source ~/ros2_workspaces/DandK_ws/install/setup.bash`).

### Computing the workspace (Section 4 of the report)

```bash
# Default — medium density, fast, saves to docs/report_assets/
ros2 run ur5_project workspace

# Report-grade — 3.4M configurations, ~5 seconds, exact statistics
ros2 run ur5_project workspace --density ultra
```

**Outputs:** `workspace_3d.png`, `workspace_top.png`, `workspace_front.png`,
`workspace_side.png` in `docs/report_assets/`.

### Validating forward kinematics against Gazebo (Section 3.3)

This requires Gazebo to be already running.

**Terminal 1** — launch the simulator:

```bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5
```

Wait until you see `Configured and activated joint_trajectory_controller` in
the log and the arm is visible in Gazebo.

**Terminal 2** — run the validation:

```bash
ros2 run ur5_project verify_fk
```

The node moves the arm to five test configurations, reads the actual `tool0`
pose from tf2, and prints the position error of each against the analytical
prediction. All errors should be effectively zero.

### Planning a trajectory (Section 8)

```bash
ros2 run ur5_project trajectory
```

This plans one Cartesian-linear trajectory with a trapezoidal velocity profile
between the demo poses defined in the source file, and produces:

- `trajectory_cartesian_path.png` — 3D Cartesian path
- `trajectory_joint_angles.png` — six joint angles vs. time
- `trajectory_speed_profile.png` — end-effector speed vs. time

### Offline simulation (Section 9)

```bash
ros2 run ur5_project simulation
```

This is the most expensive offline step (~30 s for the MP4 to render). It
produces:

- `sim_stick_diagrams.png` — five stick-figure snapshots along the path
- `sim_ee_pose.png` — end-effector pose vs. time
- `sim_joint_angles.png` — six joint angles vs. time
- `sim_velocity_compare.png` — numerical vs. analytical |v(t)|
- `sim_torques.png` — joint torques for a 3 kg payload
- `sim_motion.mp4` — animated visualization

Pass `--no_video` to skip the MP4 rendering during quick iterations.

### Validating arbitrary start/end poses before running

```bash
ros2 run ur5_project test_trajectory
```

This is a sanity check before `simulation_gazebo`: it confirms that both
endpoints have IK solutions, and that every intermediate sample along the
straight-line Cartesian path is reachable. If validation passes, it produces
a full set of plots in `docs/report_assets/test_<timestamp>/`.

### Live Gazebo simulation (Section 9.2.1)

This is the headline demo. It requires Gazebo to be already running.

**Terminal 1** — launch the simulator (same command as for `verify_fk`):

```bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5
```

**Terminal 2 (optional)** — start screen recording:

```bash
cd ~/projects/DandK_UR5_Project
./record_gazebo.sh gazebo_motion.mp4
```

**Terminal 3** — run the motion:

```bash
# Default — uses the canonical demo poses from the source
ros2 run ur5_project simulation_gazebo

# Or pick your own start and end (X Y Z in meters, R P YAW in degrees):
ros2 run ur5_project simulation_gazebo \
    --start  0.5  0.1  0.3   0  90  0 \
    --end    0.3  0.4  0.5  45  90  0
```

When picking your own poses, run `test_trajectory` first to make sure the
pair is reachable and the linear path between them lies inside the workspace.

---

## CLI reference

Every node accepts `-h`/`--help`. The most useful flags are summarized below.

### `workspace`

| Flag                                                          | Meaning |
|---------------------------------------------------------------|---------|
| `--density {coarse,medium,fine,ultra}`                        | Sampling density (default: medium). `ultra` matches the report's 3.4M-configuration statistics. |
| `--save DIR`                                                  | Where to save PNGs (default: `docs/report_assets/`). |
| `--no-save`                                                   | Display only, do not write any files. |

### `simulation`

| Flag                       | Meaning |
|----------------------------|---------|
| `--payload KG`             | Payload mass for the torque computation (default 3.0). |
| `--v_max M_S`              | Peak Cartesian speed (default 0.25 m/s). |
| `--a_max M_S2`             | Cartesian acceleration (default 0.5 m/s²). |
| `--dt SEC`                 | Sampling step (default 0.01 s). |
| `--save DIR`               | Output directory for PNGs and MP4. |
| `--fps FPS`                | MP4 frame rate (default 30). |
| `--slowdown FACTOR`        | Slow the MP4 down by this factor (default 3.0). |
| `--no_video`               | Skip MP4 rendering. |
| `--no_show`                | Do not pop up plot windows. |

### `simulation_gazebo`

| Flag                                | Meaning |
|-------------------------------------|---------|
| `--start X Y Z R P YAW`             | Start pose: meters and degrees. If omitted, uses the canonical demo pose. |
| `--end   X Y Z R P YAW`             | End pose, same format. |

### `trajectory`

| Flag             | Meaning |
|------------------|---------|
| `--v_max M_S`    | Peak Cartesian speed (default 0.25). |
| `--a_max M_S2`   | Cartesian acceleration (default 0.5). |
| `--dt SEC`       | Sampling step (default 0.01). |
| `--save DIR`     | Output directory. |
| `--no_show`      | Suppress interactive plot windows. |

### `test_trajectory`

| Flag                 | Meaning |
|----------------------|---------|
| `--tag NAME`         | Subfolder name suffix (default: timestamp). |
| `--no_video`         | Skip MP4 rendering. |
| `--no_show`          | Do not pop up plot windows. |

### `verify_fk`

No command-line flags. Reads five hard-coded test configurations from the
source, publishes them to the controller, and compares the resulting `tool0`
pose against `forward_kinematics()`.

---

## Output files

All generated assets are written to `docs/report_assets/` by default:

```
docs/report_assets/
├── workspace_3d.png            (from `workspace`)
├── workspace_top.png
├── workspace_front.png
├── workspace_side.png
├── trajectory_cartesian_path.png    (from `trajectory`)
├── trajectory_joint_angles.png
├── trajectory_speed_profile.png
├── sim_stick_diagrams.png      (from `simulation`)
├── sim_ee_pose.png
├── sim_joint_angles.png
├── sim_velocity_compare.png
├── sim_torques.png
├── sim_motion.mp4
└── test_<timestamp>/           (from `test_trajectory`, per-run subfolder)
    ├── sim_stick_diagrams.png
    └── ...
```

MP4 files are not tracked in Git (see `.gitignore`).

---

## Repository structure

```
DandK_UR5_Project/
├── README.md                    ← this file
├── LICENSE                      ← MIT
├── .gitignore
├── record_gazebo.sh             ← Wayland-safe Gazebo screen recorder
├── docs/
│   ├── UR5_Report_Hebrew.docx   ← the technical report
│   └── report_assets/           ← all generated figures and videos
└── ur5_project/                 ← the ROS2 package itself
    ├── package.xml              ← ROS2 metadata
    ├── setup.py                 ← Python metadata + entry_points
    ├── setup.cfg                ← install paths for ros2 run
    ├── resource/ur5_project     ← ament_index marker (empty file)
    └── ur5_project/             ← the Python module
        ├── __init__.py
        ├── kinematics.py
        ├── inverse_kinematics.py
        ├── jacobian.py
        ├── trajectory.py
        ├── workspace.py
        ├── verify_fk.py
        ├── simulation.py
        ├── simulation_gazebo.py
        └── test_trajectory.py
```

---

## Troubleshooting

**`ros2: command not found`**
You haven't sourced ROS2. Run `source /opt/ros/jazzy/setup.bash`.

**`Package 'ur5_project' not found`**
You haven't sourced the workspace overlay. Run
`source ~/ros2_workspaces/DandK_ws/install/setup.bash` from any shell where
you want to use the package.

**`undefined symbol: HardwareComponentInterface`** when launching Gazebo
A version skew between `hardware-interface` and `gz_ros2_control`. Run
`sudo apt update && sudo apt upgrade -y` and reboot.

**`No solution found` from `inverse_kinematics`**
The requested pose is outside the workspace, or on a singularity. Use
`workspace` to visualize the reachable set, and `test_trajectory` to
validate a candidate start/end pair before running `simulation_gazebo`.

**Gazebo opens but the arm doesn't move when `simulation_gazebo` runs**
The trajectory controller probably hasn't loaded yet. Wait until you see
`Configured and activated joint_trajectory_controller` in the Gazebo
terminal's log before launching anything else.

**MP4 rendering fails**
Install ffmpeg: `sudo apt install ffmpeg`.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
