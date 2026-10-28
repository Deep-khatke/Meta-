# RoboForge — Feature Gaps and Upgrade Roadmap

**Document Version:** 1.0.0  
**Date:** October 2026  
**Auditor:** Senior Robotics Simulation Engineer & Software Architect  
**Repository:** `RoboForge (anti-meta)`  

---

## 1. Overview

This document formalizes the gaps identified during the comprehensive audit of RoboForge against leading robotics simulation environments (NVIDIA Isaac Sim, Gazebo Sim, MuJoCo, Webots, CoppeliaSim, PyBullet, and ROS 2). Each gap is characterized by severity, root cause, architectural dependencies, planned implementation, and test verification strategy.

---

## 2. Prioritized Gap Matrix

| ID | Feature Domain | Gap Description | Severity | Impact on User Journey | Planned Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GAP-01** | Physics Core | Missing simulation snapshot & restore mechanism; cannot rewind state during experiments without rebuilding model. | **High** | Experiment workflows cannot rewind to intermediate states or reset rollouts deterministically. | Implement `get_snapshot()` and `restore_snapshot()` in `MuJoCoEngine` capturing `qpos`, `qvel`, `act`, `time`, `step_count`. |
| **GAP-02** | Physics Core | Contact reporting lacks 3D contact forces ($F_{normal}$, $F_{tangent}$) and penetration depth from MuJoCo contact solver. | **Medium** | Users and reinforcement learning agents cannot measure impact intensity or contact friction forces. | Use `mujoco.mj_contactForce()` to compute normal and tangential contact force vectors for all active contact pairs. |
| **GAP-03** | Physics Core | Real-Time Factor (RTF) is not calculated or surfaced in telemetry. | **Low** | Users cannot assess whether simulation runs faster or slower than real-time. | Track rolling window $dt_{sim} / dt_{wall}$ and report in `SimulationState`. |
| **GAP-04** | Scene Editor | Objects cannot be locked in the 3D viewport; missing scene overlap and out-of-bounds diagnostics before run. | **High** | Users accidentally displace static objects when orbiting; robots spawn inside obstacles causing immediate physics explosion. | Add `is_locked` attribute to `SceneObject`; add `/api/scene/validate` endpoint for AABB overlap and bounds checks. |
| **GAP-05** | Robot Modeling | Lacks native MJCF XML import and export (only URDF supported). | **Medium** | MuJoCo users cannot import or export native `.mjcf` / `.xml` models directly. | Implement `export_robot_mjcf()` and `parse_mjcf_string()` in `backend/mjcf_importer.py`. |
| **GAP-06** | Sensors | Backend lacks offscreen camera rendering endpoint; camera PiP exists in browser but backend cannot capture image bytes. | **High** | Synthetic perception, automated test verification, and headless computer vision pipelines cannot access camera frames. | Implement `mujoco.Renderer` offscreen camera render endpoint `/api/sensors/camera/capture` returning base64 JPEG/PNG and depth. |
| **GAP-07** | Sensors | Sensors lack realistic noise models (Gaussian noise, IMU bias drift, wheel encoder quantization). | **Medium** | Policies trained on idealized sensor data suffer from sim-to-real gap and overfit to noise-free data. | Implement configurable Gaussian range noise, IMU bias drift, and wheel encoder tick quantization. |
| **GAP-08** | Controllers | No closed-loop PID controller or trajectory tracking controller; teleop writes raw velocities without feedback. | **Medium** | Joint position and velocity control cannot execute smooth trajectory tracking or reject disturbances. | Implement `PIDController` class with proportional, integral, derivative gains and integral anti-windup. |
| **GAP-09** | Motion Planning | Lacks collision-aware global path planning and continuous waypoint execution in live MuJoCo loop. | **High** | Robots stopped between discrete waypoints or hit concavities. | **CLOSED**: Implemented A* grid planner + `ContinuousTrajectoryController` pure-pursuit steering directly in MuJoCo physics loop without intermediate halts. |
| **GAP-10** | Synthetic Data | No synthetic perception pipeline (2D bounding box projections, domain randomization, COCO dataset export). | **High** | Users cannot generate training datasets for computer vision or object detection models. | **CLOSED**: Implemented `backend/synthetic_data.py` with 3D-to-2D pinhole camera projection, domain randomizer, and COCO export. |
| **GAP-11** | Interoperability | Lacks standard ROS 2 message serialization (`geometry_msgs/Twist`, `sensor_msgs/LaserScan`, `nav_msgs/Odometry`, `sensor_msgs/Imu`, `sensor_msgs/JointState`). | **High** | Cannot integrate with standard ROS 2 nodes, external autonomy stacks, or micro-ROS agents. | **CLOSED**: Implemented `backend/ros2_bridge.py` serializing simulation states into standard ROS 2 topic dictionaries. |
| **GAP-12** | Experiments | Multi-seed evaluations lack statistical uncertainty reporting (mean $\pm$ std, confidence intervals). | **Medium** | Researchers cannot evaluate policy stability or variance across random seeds. | **CLOSED**: Added multi-trial experiment aggregator computing mean, standard deviation, min, max, and success rate. |
| **GAP-13** | Physics / Autonomy | **Gap A: Continuous Trajectory Execution**: Disconnected planner and stop-and-go jerking at intermediate waypoints. | **High** | Playback paused between waypoints; commanded trajectory failed to step at authoritative physics rate. | **CLOSED**: Connected `ContinuousTrajectoryController` directly into `simulation_loop` with lookahead steering, smooth waypoint advancement ($\le 0.28$m), and obstacle collision abort. |
| **GAP-14** | Perception Core | **Gap B: Semantic Segmentation**: Missing genuine pixel-level geometry ID masks; relied on approximations or raw RGB. | **High** | Scientific agents and computer vision models could not obtain pixel-level ground truth geom masks. | **CLOSED**: Implemented MuJoCo `renderer.enable_segmentation_rendering()` extracting channel-0 geom IDs, label dictionaries, background handling, and color-coded PNG masks. |
| **GAP-15** | 3D Editor / Studio | **Gap C: Robot Assembly Snapping**: Missing surface/attachment-point snapping for compatible robot links and objects. | **Medium** | Users struggled to align joints to parent links; objects floated or penetrated ground and stacking surfaces. | **CLOSED**: Implemented Joint Origin Frame snapping (`Snap Parent Tip (+X)`, `Snap Parent Top (+Z)`, `Snap Origin (0,0,0)`) in `RobotStudioModal.tsx`, and Ground/Stacking/Flange snapping in `Viewport3D.tsx`. |
| **GAP-16** | Agent Interface | **Agent-Ready Simulation Interface**: Need typed, structured interface for autonomous scientific research agent. | **High** | Agent had no unified programmatic API for inspection, telemetry, perception, snapshots, and experiments. | **CLOSED**: Implemented `AgentSimulationInterface` (`backend/agent_interface.py`) with typed Pydantic models and full REST endpoint exposure. |

---

## 3. Explicit Architectural Limitations & Non-Goals

1. **GPU RTX Photorealism:**
   - *Constraint:* NVIDIA Omniverse RTX rendering requires proprietary NVIDIA GPUs and multi-gigabyte drivers/containers.
   - *Decision:* RoboForge uses MuJoCo's native hardware/software offscreen renderer and Three.js WebGL in the browser. Truthful geometric and physical correctness is prioritized over non-local RTX raytracing.
2. **Native Windows ROS 2 Distribution:**
   - *Constraint:* Installing full ROS 2 Humble/Iron on Windows requires multi-GB binaries, Visual Studio build tools, and administrator privileges.
   - *Decision:* Provide a clean, zero-dependency ROS 2 JSON/WebSocket topic interface conforming to official ROS 2 message schemas (`sensor_msgs`, `nav_msgs`, `geometry_msgs`).
3. **Cross-Engine Physics Parity:**
   - *Constraint:* Simulating identical contact dynamics across both MuJoCo and Bullet/PhysX introduces non-deterministic discrepancies.
   - *Decision:* MuJoCo remains the sole authoritative physics engine throughout RoboForge.

