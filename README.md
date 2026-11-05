# RoboForge — Robotics Simulation & Control Studio

<p align="center">
  <strong>A high-fidelity browser-based robotics modeling, simulation, and control platform powered by MuJoCo physics and Three.js.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Physics-MuJoCo%203.17-blue.svg" alt="MuJoCo" />
  <img src="https://img.shields.io/badge/Backend-FastAPI%20%2B%20WebSockets-green.svg" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Frontend-React%20%2B%20Three.js%20%2B%20Vite-purple.svg" alt="React + Three.js" />
  <img src="https://img.shields.io/badge/Tests-86%20Passing-brightgreen.svg" alt="Tests" />
  <img src="https://img.shields.io/badge/License-MIT-orange.svg" alt="License" />
</p>

---

## 🚀 Overview

**RoboForge** is an engineering simulation platform that blends true **MuJoCo physics dynamics** with a desktop IDE-inspired engineering user interface. It provides an end-to-end workflow for designing articulated robots, importing standard URDF definitions, controlling joints interactively, solving forward/inverse kinematics, running autonomous navigation with LiDAR & vision sensors, and optimizing collision-free policies.

The workspace is designed with intelligent space allocation: **the 3D simulation scene is permanently mounted and visible**, while modeling studios, asset libraries, telemetry, and diagnostics smoothly dock around it.

---

## ⚡ Headline Capabilities

> Design a robot → command it in plain English → train it with real RL → prove it → deploy the trained brain to a real ROS 2 robot. All in the browser, all on live MuJoCo physics.

- **🗣️ Natural-language control** — talk to the scene: *"drive in a square"*, *"go to the red box"*, *"spawn a warehouse"*. The agent plans and executes on real physics (no scripting, no coordinates).
- **🏗️ One-prompt missions** — *"build a warehouse and make the rover patrol it"* generates the environment, plans a **collision-free A\* patrol loop**, loads it into the visual **Mission Planner**, and drives it.
- **👁️ Robot vision** — the onboard camera feeds a vision-language model: ask *"what do you see?"* or *"go to the red box"* and it finds the target by sight.
- **🧠 Live swarm reinforcement learning** — PPO trains **8 rovers in parallel**, rendered live as translucent "ghosts" that are the *actual* environments the policy learns from — watch them explore, fail, and converge ("watch it learn," Isaac-Lab style).
- **📊 Robustness scorecard** — score any trained policy across **domain-randomized worlds** (friction, mass, gravity, sensor noise) vs a clean baseline — research-grade sim-to-real evidence.
- **📦 Deploy to real robots** — export a trained policy to **ONNX + a ready-to-run ROS 2 node** (subscribes `/scan`, `/odom`; publishes `/cmd_vel`).
- **🔌 Real ROS 2 interoperability** — a built-in **rosbridge v2 server** (`ws://localhost:9090`) streams `/odom`, `/scan`, `/imu`, `/tf`, `/clock` and accepts `/cmd_vel`, `/goal_pose` — connect Foxglove, roslibpy, or a real ROS 2 graph.
- **🚁 Genuine quadcopter flight** — drones fly on **real per-rotor thrust physics** (lift and attitude emerge from 4 independent rotors), with a 3D waypoint flight-follower and the same Mission Planner.

---

## 🌟 Key Features

### 1. 🤖 Robot Modeling & Control Studio
- **Articulated Robot Builder**: Create and edit multi-link serial manipulators and branched kinematic chains with custom visual geometry (Box, Cylinder, Sphere), collision envelopes, and auto-computed mass/inertial tensors.
- **Joint Types Supported**: Revolute, Continuous, Prismatic, and Fixed joints compiled natively for MuJoCo physics.
- **Kinematic Hierarchy & Visualizer**: Toggle real-time joint rotation axes, coordinate reference frames, joint limit arcs, link hierarchies, and end-effector pose readouts directly in the 3D scene.
- **Interactive Joint Control Studio**: Multi-mode joint actuation (**Position**, **Velocity**, **Torque**) with real-time limit enforcement, target readouts, and quick actions (*Zero All*, *Center Limits*, *Hold Position*, *Relax Torques*).
- **Kinematics Engine**:
  - **Forward Kinematics (FK)**: Live coordinate computation for every link frame in 3D world space.
  - **Inverse Kinematics (IK)**: Damped Least-Squares Jacobian solver with joint limit avoidance and automatic reachability validation.
- **Physical Diagnostics**: Automatic validation for singular configurations, mass distribution balance, and self-collision clearance.
- **Bidirectional URDF Interchange**: Import standard `.urdf` XML models directly into MuJoCo, export modified robot models, and share portable JSON schemas.

### 2. ⚡ Real-Time Physics & Simulation
- **MuJoCo 3.17 Physics Ownership**: Authoritative 60–200 Hz physical simulation running with `implicitfast` integration.
- **Differential-Drive Mobile Robotics**: Differential-drive kinematics, chassis dynamics, driven wheels with cylinder friction contacts, and passive caster stabilizers.
- **Dynamic & Static Obstacles**: Spawning and manipulation of boxes, cylinders, and spheres with customizable mass, friction, and restitution.
- **Execution & Playback Controls**: Run, Pause, Single-Step, Reset, Speed multipliers (0.1x to 5.0x), and sub-millisecond timestep configuration (0.001s to 0.02s).

### 3. 📡 Sensor Suite & Telemetry
- **Raycast LiDAR**: Configurable 180° and 360° laser rangefinders with live beam visualization and obstacle impact distance tracking.
- **6-DOF Inertial Measurement Unit (IMU)**: Accelerometer ($m/s^2$) and gyroscope ($rad/s$) telemetry extracted directly from MuJoCo physics equations of motion.
- **RGB Vision Camera**: Real-time robot eye viewpoint rendering with selectable Field of View (FOV).
- **Collapsible Telemetry Deck**: High-frequency SVG velocity & torque plots, contact counter, odometry tracking, and streaming backend execution logs.

### 4. 📦 Reusable Asset & Scene Library
- Curated system library of mobile robots, industrial arms (e.g., 2-Link Planar, 3-Link SCARA), obstacle configurations, and test environments.
- Save custom robot designs and complete scene snapshots into local persistence.
- Single-column docked catalog with category filters, tag searching, and live asset inspector.

### 5. 🧠 AI Policy Optimizer & Autonomous Navigation
- Trajectory recording and deterministic replay with scrubbing.
- Automated waypoint navigation with repulsive potential fields and obstacle avoidance.
- Genetic/CMA-ES parameter optimization rollouts evaluated inside MuJoCo for zero-collision policies.

---

## 🏛️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Frontend (React 19 + Vite)                      │
│  - Three.js 3D Viewport with OrbitControls & Dynamic Raycaster         │
│  - Docked Robot Modeling & Control Studio (Builder, Joints, FK/IK)     │
│  - Reusable Asset Library with Live Specification Inspector            │
│  - Unified 44px Control Deck & Collapsible SVG Telemetry Bar          │
│  - Real-time Robot Eye HUD Camera & Raycast LiDAR Beams               │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ WebSocket (60Hz Telemetry & Controls)
                                    │ REST API (Projects, URDF, Models)
┌───────────────────────────────────▼────────────────────────────────────┐
│                        Backend (FastAPI + Python 3.11)                 │
│  - Simulation Manager & WebSocket Hub (60–200 FPS step loop)          │
│  - Robot Modeling Compiler (GenericRobotModel -> MuJoCo MJCF XML)      │
│  - Kinematics Engine (FK Link Poses & Numerical Damped Least-Squares) │
│  - URDF XML Parser & Serializer with Validation Checks                 │
│  - Project & Asset Persistence Manager (JSON Schemas)                  │
│  - RL & AI Policy Optimizer for Collision Avoidance                    │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ C-Bindings
┌───────────────────────────────────▼────────────────────────────────────┐
│                      MuJoCo 3.17 Physics Core                          │
│  - Articulated Multibody Dynamics (Revolute, Prismatic, Continuous)    │
│  - Contact Constraints, Solvers & Friction Geometries                 │
│  - Position, Velocity, and Torque Actuators                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Physics Core**: [MuJoCo](https://mujoco.org/) 3.17 (Official Python bindings)
- **AI Reasoning**: [Meta Model API](https://dev.meta.ai/) — Meta **Muse** models (`muse-spark-1.3`) power the natural-language control, mission planning, and agentic tool-calling layer (OpenAI-compatible)
- **Reinforcement Learning**: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) on [PyTorch](https://pytorch.org/), with [ONNX](https://onnx.ai/) policy export
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), WebSockets, [Pydantic v2](https://docs.pydantic.dev/)
- **Mathematics & Kinematics**: [NumPy](https://numpy.org/)
- **Frontend Framework**: [React 19](https://react.dev/), [TypeScript](https://www.typescriptlang.org/), [Vite](https://vite.dev/)
- **3D Graphics & Rendering**: [Three.js](https://threejs.org/)
- **Robotics Interop**: [ROS 2](https://docs.ros.org/) via rosbridge v2
- **Icons & UI**: [Lucide React](https://lucide.dev/), Vanilla CSS Design Tokens
- **Testing**: [pytest](https://pytest.org/)

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- **Python 3.11+** installed
- **Node.js 18+** and **npm** installed

### 2. Clone the Repository
```bash
git clone https://github.com/Deep-khatke/Meta-.git
cd Meta-
```

### 3. Setup Backend Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install fastapi "uvicorn[standard]" websockets mujoco numpy pytest pydantic httpx
```

### 4. Setup Frontend Environment
```bash
cd frontend
npm install
cd ..
```

### 5. Launch RoboForge

#### Option A: Using the PowerShell Launcher (Windows)
```powershell
.\start.ps1
```

#### Option B: Manual Launch
In **Terminal 1** (Backend):
```bash
.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In **Terminal 2** (Frontend):
```bash
cd frontend
npm run dev
```

Open your browser and navigate to: **`http://localhost:5173`**  
Interactive API Docs (Swagger): **`http://localhost:8000/docs`**

---

## 🧪 Testing & Validation

RoboForge includes an automated test suite verifying physics stability, kinematic accuracy, joint limits, asset validation, and project persistence.

Run all backend tests:
```bash
.venv\Scripts\python -m pytest backend/tests/ -v
```

Output:
```text
======================== 86 passed in 5.83s ========================
```

Validate frontend TypeScript and bundle compilation:
```bash
cd frontend
npm run build
```

---

## 📂 Project Structure

```text
├── backend/
│   ├── app.py                 # FastAPI application, WebSocket hub & Auth endpoints (/api/auth)
│   ├── database.py            # SQLite database with PBKDF2 hashing & session management
│   ├── ai_reasoning_service.py # Tiered reasoning engine (NVIDIA Nemotron via OpenRouter, Meta API)
│   ├── simulator.py           # Core MuJoCo physics engine & MJCF generator
│   ├── robot_studio.py        # Robot modeling engine, FK/IK solvers & URDF parser
│   ├── robot_validator.py     # Structural validation, singular configs & diagnostics
│   ├── asset_library.py       # Reusable asset catalog management
│   ├── project.py             # Project persistence & schema versioning
│   ├── rl_env.py              # MuJoCo environment for policy rollout evaluation
│   ├── rl_evaluator.py        # AI policy optimizer & collision minimization
│   ├── models.py              # Pydantic schemas & state models
│   └── tests/                 # 86 automated unit and integration tests
├── frontend/
│   ├── src/
│   │   ├── context/
│   │   │   └── AuthContext.tsx        # Persistent user authentication & session management
│   │   ├── components/
│   │   │   ├── GoogleSignInButton.tsx # Official Google OAuth 2.0 & One-Tap button
│   │   │   ├── UserProfileBadge.tsx   # User profile pill with avatar and initials fallback
│   │   │   ├── Viewport3D.tsx         # Three.js 3D viewport & visualizer
│   │   │   ├── RobotStudioModal.tsx   # Docked Robot Modeling & Control Studio
│   │   │   ├── AssetLibraryPanel.tsx  # Docked Asset & Template Catalog
│   │   │   ├── Toolbar.tsx            # Unified 44px control deck
│   │   │   ├── ScenePanel.tsx         # Scene hierarchy & obstacle spawner
│   │   │   ├── PropertiesPanel.tsx    # Robot limits & physical parameters
│   │   │   ├── BottomPanel.tsx        # Collapsible SVG telemetry & logs
│   │   │   ├── RobotCreatorModal.tsx  # Mobile/Articulated robot generator
│   │   │   └── EnvironmentCreatorModal.tsx
│   │   ├── pages/
│   │   │   ├── LoginPage.tsx          # Login page with 3D interactive RoboBuddy
│   │   │   ├── SignupPage.tsx         # Signup page with full credentials validation
│   │   │   └── StudioPage.tsx         # Full engineering workspace
│   │   ├── services/          # REST & WebSocket API clients
│   │   ├── types.ts           # TypeScript interfaces & state definitions
│   │   └── App.tsx            # Main application layout & state orchestrator
│   └── package.json
├── assets/                    # Built-in simulation assets & catalogs
├── projects/                  # Default and saved simulation projects
├── start.ps1                  # Single-click PowerShell startup script
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
