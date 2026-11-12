# 🤖 RoboForge — Next-Gen AI Robotics Simulation & Control Studio

<p align="center">
  <strong>A high-fidelity browser-based robotics modeling, simulation, and autonomous control platform powered by MuJoCo 3.17 physics, Three.js, and NVIDIA Nemotron AI.</strong>
</p>

<p align="center">
  <a href="https://roboforge.duckdns.org" target="_blank">
    <img src="https://img.shields.io/badge/Live%20Platform-roboforge.duckdns.org-2ea44f?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Live Platform" />
  </a>
  <a href="https://roboforge.duckdns.org" target="_blank">
    <img src="https://img.shields.io/badge/Hosted%20On-AWS%20EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS EC2" />
  </a>
  <a href="https://mujoco.org" target="_blank">
    <img src="https://img.shields.io/badge/Physics-MuJoCo%203.17-007ACC?style=for-the-badge" alt="MuJoCo 3.17" />
  </a>
  <a href="https://openrouter.ai" target="_blank">
    <img src="https://img.shields.io/badge/AI%20Brain-NVIDIA%20Nemotron-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA Nemotron" />
  </a>
  <a href="https://roboforge.duckdns.org/login" target="_blank">
    <img src="https://img.shields.io/badge/Auth-Google%20OAuth%202.0%20%2B%20SQLite-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google OAuth" />
  </a>
  <img src="https://img.shields.io/badge/Tests-86%20Passing-brightgreen?style=for-the-badge" alt="86 Passing" />
  <img src="https://img.shields.io/badge/License-MIT-F97316?style=for-the-badge" alt="MIT License" />
</p>

---

## 🌐 Live Production Deployment

RoboForge is deployed live on AWS EC2 with continuous systemd orchestration, DuckDNS dynamic routing, and Nginx reverse proxy:

| Service | Access Link | Description |
| :--- | :--- | :--- |
| **Live Studio Web App** | **[https://roboforge.duckdns.org](https://roboforge.duckdns.org)** | Full browser-based 3D robotics workspace & simulation |
| **Secure Login / Signup** | **[https://roboforge.duckdns.org/login](https://roboforge.duckdns.org/login)** | Interactive 3D companion robot + Google Sign-In & Email Auth |
| **Interactive API Docs** | **[https://roboforge.duckdns.org/docs](https://roboforge.duckdns.org/docs)** | OpenAPI Swagger documentation for all backend endpoints |
| **System Health Check** | **[https://roboforge.duckdns.org/api/health](https://roboforge.duckdns.org/api/health)** | Live engine state, WebSocket broadcast, and schema version |

---

## ⚡ Headline Capabilities

> **Design a robot → command it in plain English → train it with real RL → prove it → deploy the trained brain to a real ROS 2 robot. All in the browser, all on live MuJoCo physics.**

- **🗣️ Natural-Language Robotics Control** — Talk directly to the simulation scene: *"drive in a square"*, *"go to the red cylinder"*, *"spawn a warehouse with 4 obstacles"*. The AI planning engine translates intents into verified physical trajectories without manual coding or coordinates.
- **🧠 NVIDIA Nemotron & Meta Muse Tiered Reasoning** — Powered by NVIDIA's high-capacity reasoning model (`nvidia/nemotron-3.5-super-vl:free`) via OpenRouter and Meta Model API (`muse-spark-1.3`), with automatic graceful fallback to local Ollama and deterministic planners.
- **🏗️ One-Prompt Missions & Autonomous Waypoint Planning** — *"Build a warehouse and make the rover patrol it"* automatically generates the 3D environment, computes a **collision-free A\* patrol loop**, loads waypoints into the visual **Mission Planner**, and executes the mission.
- **👁️ Robot Vision & Onboard Eye Camera** — An onboard RGB camera feeds live sensor frames to the vision-language model: ask *"what do you see?"* or *"navigate to the nearest obstacle"* and the robot locates and tracks its target visually.
- **🌌 Live Swarm Reinforcement Learning (PPO)** — Trains **8 rovers in parallel** using Proximal Policy Optimization (PPO), rendered live in Three.js as translucent "ghost" agents that are the *actual* environments the policy learns from — watch exploration, failure, and convergence in real time (*Isaac-Lab style*).
- **📊 Sim-to-Real Robustness Scorecard** — Stress-test any trained policy across **domain-randomized worlds** (variable friction, payload mass drift, gravity anomalies, sensor Gaussian noise) vs. a clean baseline for empirical sim-to-real evidence.
- **📦 Direct ROS 2 Node & ONNX Deployment** — Export any trained reinforcement learning policy to **ONNX** alongside an immediately deployable **ROS 2 Python node** (`subscribes /scan, /odom; publishes /cmd_vel`).
- **🔌 Built-in ROS 2 Bridge Server** — Integrated **rosbridge v2 server** (`ws://localhost:9090`) streams live telemetry topics (`/odom`, `/scan`, `/imu`, `/tf`, `/clock`) and ingests commands (`/cmd_vel`, `/goal_pose`) compatible with Foxglove Studio, roslibpy, or hardware robot graphs.
- **🚁 Genuine Quadcopter Flight Dynamics** — Drones fly on **authentic per-rotor thrust physics** (lift, roll, pitch, and yaw emerge from 4 independent aerodynamic rotor forces), coupled with a 3D waypoint flight-follower.
- **🔐 Multi-User Authentication & Persistence** — Production-grade persistent authentication with **Google OAuth 2.0 & Google One-Tap**, email/password with salted **PBKDF2-HMAC-SHA256** hashing, SQLite session management, and dynamic user profile badges with avatar fallbacks.
- **🤖 Interactive 3D RoboBuddy Companion** — Custom procedural 3D companion robot on `/login` and `/signup` with real-time cursor tracking, interactive behaviors (*Pet, Play Trick, Feed Treat, High Five*), and a **Web Audio API procedural sound synthesizer** (zero external audio files).

---

## 🌟 Core Features & Modules

### 1. 🤖 Robot Modeling & Control Studio
- **Articulated Multibody Builder**: Design multi-link serial manipulators, branched kinematic chains, and mobile bases with custom visual and collision geometry (Box, Cylinder, Sphere) and auto-computed mass/inertial tensors.
- **Supported Joint Topologies**: Revolute, Continuous, Prismatic, and Fixed joints compiled natively for MuJoCo physics.
- **Kinematic Visualization**: Toggle real-time joint rotation axes, coordinate reference frames, joint limit arcs, link hierarchies, and live end-effector 6D pose readouts.
- **Interactive Control Deck**: Multi-mode joint actuation (**Position**, **Velocity**, **Torque**) with real-time limit clamping and quick triggers (*Zero All*, *Center Limits*, *Hold Position*, *Relax Torques*).
- **Dual Kinematics Solvers**:
  - **Forward Kinematics (FK)**: Live coordinate transformation for every link frame in 3D world space.
  - **Inverse Kinematics (IK)**: Damped Least-Squares (DLS) Jacobian solver with joint limit avoidance and singularity dampening.
- **Bidirectional URDF Interchange**: Import standard `.urdf` XML models directly into MuJoCo, modify properties in the visual studio, and export synchronized URDF / MJCF / JSON definitions.

### 2. ⚡ Real-Time MuJoCo Physics Core
- **Authoritative Physical Simulation**: 60–200 Hz continuous physical integration utilizing MuJoCo 3.17 `implicitfast` solver.
- **Differential-Drive Mobile Robotics**: Differential kinematics, chassis inertial dynamics, driven wheels with cylinder friction contacts, and passive caster stabilizers.
- **Interactive Scene Manipulation**: Real-time 3D translation/rotation gizmos to manipulate robots, static obstacles, and dynamic targets.
- **Execution & Playback Controls**: Run, Pause, Single-Step, Reset, Speed multipliers (0.1x to 5.0x), and sub-millisecond timestep configuration (0.001s to 0.02s).

### 3. 📡 Sensor Suite & Telemetry Dashboard
- **Raycast LiDAR**: Configurable 180° and 360° laser rangefinders with live visual ray beam rendering and obstacle impact distance tracking.
- **6-DOF IMU**: Live accelerometer ($m/s^2$) and gyroscope ($rad/s$) telemetry extracted directly from equations of motion.
- **Real-Time Telemetry Bar**: Collapsible SVG velocity, orientation, and actuator torque plots, contact counter, odometry tracking, and streaming backend execution logs.

### 4. 🔐 Security, Authentication & User Profiles
- **Google OAuth 2.0**: One-click Google sign-in via Google Identity Services (GSI) with automatic profile image retrieval and account synchronization.
- **Salted PBKDF2 Password Hashing**: Passwords stored using 100,000-iteration PBKDF2-HMAC-SHA256 with cryptographically random 16-byte salts.
- **SQLite Database Persistence**: Isolated SQLite store (`backend/roboforge.db`) tracking users, credentials, session tokens, and timestamps.
- **User Profile Pill**: Navbar badge displaying live user avatar (with `referrerPolicy="no-referrer"` for Google CDN compatibility), user status dropdown, account switcher, and initials fallback.

---

## 🏛️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   Frontend (React 19 + Vite + TypeScript)              │
│  - Three.js 3D Viewport with OrbitControls & Dynamic Raycaster         │
│  - Docked Robot Modeling & Control Studio (Builder, Joints, FK/IK)     │
│  - 3D Interactive RoboBuddy with Web Audio API Procedural Synthesizer  │
│  - AuthContext (Google OAuth 2.0, Persistent Sessions, Profile Badge)  │
│  - Swarm RL Visualizer (8-Rover Ghost Training & Robustness Matrix)    │
│  - Mission Planner (Waypoint Editor, Collision-Free A* Paths)          │
│  - Collapsible SVG Telemetry Deck & Live LiDAR Beam Renderer           │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ WebSocket (60Hz Telemetry & Controls)
                                    │ REST API (Projects, URDF, Models, Auth)
┌───────────────────────────────────▼────────────────────────────────────┐
│                    Backend Server (FastAPI + Python 3.11)              │
│  - Simulation Manager & WebSocket Hub (60–200 FPS physics loop)        │
│  - User Auth & Session Controller (/api/auth/register, login, google)  │
│  - SQLite Database Store (roboforge.db with PBKDF2 Salted Hashing)     │
│  - AI Reasoning Service (NVIDIA Nemotron via OpenRouter, Meta Muse)    │
│  - PayPal Subscriptions & Credit Engine (/api/billing/*, Webhooks)     │
│  - Robot Modeling Compiler (GenericRobotModel -> MuJoCo MJCF XML)      │
│  - Kinematics Engine (FK Link Poses & Numerical Damped Least-Squares) │
│  - Built-in ROS 2 Bridge Server (ws://0.0.0.0:9090)                    │
│  - PPO Reinforcement Learning Swarm & ONNX / ROS 2 Exporter            │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ C-Bindings
┌───────────────────────────────────▼────────────────────────────────────┐
│                       MuJoCo 3.17 Physics Core                         │
│  - Articulated Multibody Dynamics (Revolute, Prismatic, Continuous)    │
│  - Contact Constraints, Solvers & Friction Geometries                 │
│  - Position, Velocity, and Torque Actuators                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Physics Core** | [MuJoCo 3.17](https://mujoco.org/) (Official Python bindings, `implicitfast` solver) |
| **AI Reasoning Brain** | [NVIDIA Nemotron](https://openrouter.ai/) (`nemotron-3.5-super-vl:free`), [Meta Muse](https://dev.meta.ai/) (`muse-spark-1.3`), Ollama |
| **Reinforcement Learning** | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO), [PyTorch](https://pytorch.org/), [ONNX](https://onnx.ai/) Export |
| **Backend Engine** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), WebSockets, [Pydantic v2](https://docs.pydantic.dev/) |
| **Database & Security** | SQLite3, PBKDF2-HMAC-SHA256, Google OAuth 2.0 (GSI), Secrets |
| **Billing & Subscriptions** | PayPal Subscriptions REST API (Sandbox), Append-only Credit Ledger, Two-Phase Reservations |
| **Robotics Interoperability** | [ROS 2](https://docs.ros.org/) via integrated rosbridge v2 protocol |
| **Frontend Framework** | [React 19](https://react.dev/), [TypeScript](https://www.typescriptlang.org/), [Vite 8](https://vite.dev/) |
| **3D Graphics & Animations** | [Three.js](https://threejs.org/), [GSAP](https://greensock.com/gsap/), Web Audio API |
| **Testing Suite** | [pytest](https://pytest.org/) (86 automated unit & physics tests) |
| **Cloud Infrastructure** | AWS EC2 (Ubuntu 24.04 LTS), Nginx Reverse Proxy, DuckDNS, Systemd |

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **Node.js 18+** and **npm**
- **Git**

### 2. Clone the Repository
```bash
git clone https://github.com/Deep-khatke/Meta-.git
cd Meta-
```

### 3. Environment Configuration (`.env`)
Create a root `.env` file in the project directory:

```env
# AI Reasoning Brain Configuration (NVIDIA Nemotron via OpenRouter)
AI_PROVIDER=openai_compat
OPENAI_COMPAT_BASE_URL=https://openrouter.ai/api/v1
OPENAI_COMPAT_MODEL=nvidia/nemotron-3.5-super-vl:free
OPENAI_COMPAT_API_KEY=your-openrouter-or-nvidia-api-key

# Google OAuth 2.0 Client Configuration
VITE_GOOGLE_CLIENT_ID=567256736468-3m5c2l0i74tronhurhm14mu174qkjbtc.apps.googleusercontent.com

# Server & Physics Settings
PORT=8000
HOST=127.0.0.1
SIM_FREQUENCY=60
```

Also ensure `frontend/.env` contains your Google Client ID:
```env
VITE_GOOGLE_CLIENT_ID=567256736468-3m5c2l0i74tronhurhm14mu174qkjbtc.apps.googleusercontent.com
```

### 4. Setup Backend Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or manually:
pip install fastapi "uvicorn[standard]" websockets mujoco numpy pytest pydantic httpx
```

### 5. Setup Frontend Environment
```bash
cd frontend
npm install
cd ..
```

### 6. Launch RoboForge

#### Option A: One-Click Startup Script (Windows)
Double-click `START_ROBOFORGE.bat` or run:
```powershell
.\start.ps1
```

#### Option B: Manual Launch
In **Terminal 1** (Backend):
```bash
.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

In **Terminal 2** (Frontend):
```bash
cd frontend
npm run dev
```

Open your browser:
- **Web App**: [`http://localhost:5173`](http://localhost:5173)
- **Interactive Swagger Docs**: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **ROS 2 Bridge**: `ws://localhost:9090`

---

## 🧪 Testing & Verification

RoboForge includes an automated test suite verifying physics stability, kinematic accuracy, joint limits, asset validation, and project persistence.

### Run Backend Unit & Physics Tests:
```bash
.venv\Scripts\python -m pytest backend/tests/ -v
```

Expected output:
```text
======================== 86 passed in 5.83s ========================
```

### Validate Frontend TypeScript Compilation:
```bash
cd frontend
npm run build
```

Expected output:
```text
✓ 2261 modules transformed.
✓ built in 1.84s
```

---

## 📡 REST API & WebSocket Specifications

### Authentication Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/auth/register` | Register a new user with name, email, and password |
| `POST` | `/api/auth/login` | Authenticate with email/password and obtain session cookie |
| `POST` | `/api/auth/google` | Verify Google ID token or Google OAuth profile and issue session |
| `GET` | `/api/auth/me` | Fetch currently authenticated user session state |
| `POST` | `/api/auth/logout` | Terminate session and clear credentials |

### Simulation & Modeling Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status, physics engine state, and schema version |
| `GET` | `/api/state` | Full simulation snapshot (bodies, joints, sensors, contacts) |
| `POST` | `/api/simulation/start` | Start or resume MuJoCo physics stepping |
| `POST` | `/api/simulation/pause` | Pause continuous physics loop |
| `POST` | `/api/simulation/reset` | Reset simulation state to initial project configuration |
| `POST` | `/api/robot/joint_control` | Command joint target position, velocity, or torque |
| `POST` | `/api/robot/solve_ik` | Solve Damped Least-Squares inverse kinematics for a target pose |
| `POST` | `/api/ai/command` | Submit natural-language robotics commands to NVIDIA Nemotron |

### WebSocket Channels
- `ws://localhost:8000/ws/simulation` — High-frequency (60Hz) physics telemetry, joint angles, velocities, and contact dynamics.
- `ws://localhost:9090` — Built-in **rosbridge v2 protocol** publishing `/odom`, `/scan`, `/imu`, `/tf`, `/clock`.

---

## 📂 Project Directory Structure

```text
├── backend/
│   ├── app.py                     # FastAPI application, WebSocket hub & Auth routes
│   ├── database.py                # SQLite persistent store with PBKDF2 salted hashing
│   ├── ai_reasoning_service.py    # Tiered AI engine (NVIDIA Nemotron, Meta Muse, Ollama)
│   ├── simulator.py               # Authoritative MuJoCo 3.17 physics engine & MJCF compiler
│   ├── robot_studio.py            # Modeling studio, FK/IK solvers & URDF parser
│   ├── robot_validator.py         # Structural diagnostics & self-collision checks
│   ├── asset_library.py           # Pre-built robot and obstacle catalog manager
│   ├── project.py                 # Project persistence & schema versioning
│   ├── rl_env.py                  # Parallel MuJoCo environment for policy evaluation
│   ├── rl_evaluator.py            # PPO optimizer, collision minimization & ONNX exporter
│   ├── models.py                  # Pydantic schemas & state models
│   └── tests/                     # 86 automated unit and physics integration tests
├── frontend/
│   ├── src/
│   │   ├── context/
│   │   │   └── AuthContext.tsx    # Multi-user session management & Google OAuth state
│   │   ├── components/
│   │   │   ├── GoogleSignInButton.tsx # Permanent, styled Google OAuth & One-Tap button
│   │   │   ├── UserProfileBadge.tsx   # User profile pill with avatar and initials fallback
│   │   │   ├── Viewport3D.tsx         # Three.js 3D viewport, OrbitControls & raycasters
│   │   │   ├── RobotStudioModal.tsx   # Docked Robot Modeling & Control Studio
│   │   │   ├── AssetLibraryPanel.tsx  # Docked Asset & Template Catalog
│   │   │   ├── MissionPlanner.tsx     # Visual A* waypoint mission planner
│   │   │   ├── Toolbar.tsx            # Unified 44px control deck & user badge
│   │   │   ├── ScenePanel.tsx         # Scene hierarchy & obstacle spawner
│   │   │   ├── PropertiesPanel.tsx    # Joint limits & physical parameters
│   │   │   └── BottomPanel.tsx        # Collapsible SVG telemetry & backend logs
│   │   ├── pages/
│   │   │   ├── LoginPage.tsx          # Login page with 3D companion RoboBuddy
│   │   │   ├── SignupPage.tsx         # User registration with credentials validation
│   │   │   └── StudioPage.tsx         # Full engineering robotics workspace
│   │   ├── services/              # REST & WebSocket API clients
│   │   ├── types.ts               # TypeScript interfaces & state definitions
│   │   └── App.tsx                # Main routing and layout orchestrator
│   └── package.json
├── assets/                        # Built-in simulation assets & URDF definitions
├── projects/                      # Default and saved simulation projects
├── START_ROBOFORGE.bat            # Single-click Windows launcher
├── start.ps1                      # PowerShell startup script
├── .gitignore                     # Git ignore rules (protects keys, DBs, logs)
└── README.md                      # Comprehensive project documentation
```

---

## ☁️ Production Deployment Guide (AWS EC2)

RoboForge is configured for continuous cloud execution on AWS EC2 instances:

1. **Host Configuration**:
   - Ubuntu 24.04 LTS / 22.04 LTS
   - Nginx reverse-proxy on Ports 80 & 443 with HTTPS (Let's Encrypt / DuckDNS)
   - Uvicorn backend running on internal port `8000` via systemd (`roboforge.service`)
2. **Update & Redeploy in One Command**:
   ```bash
   ssh -i roboforge-key.pem ubuntu@3.25.203.132 "cd /var/www/meta-project && git pull origin main && cd frontend && npm run build && sudo systemctl restart roboforge && sudo systemctl reload nginx"
   ```
3. **Inspect Production Logs**:
   ```bash
   sudo journalctl -u roboforge.service -f
   ```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
