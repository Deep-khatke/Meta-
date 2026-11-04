"""Export a trained PPO policy into a portable, real-robot-ready package.

Produces a zip containing:
  - policy.onnx                : the actor network in ONNX (obs[16] -> action[2]), runs anywhere
    via onnxruntime, no stable-baselines3 / torch needed on the robot.
  - roboforge_policy_node.py   : a ROS 2 node that loads policy.onnx, subscribes to /scan + /odom,
    builds the exact 16-dim observation this policy was trained on, and publishes /cmd_vel.
  - README.md                  : observation/action layout + run instructions.
  - metadata.json              : the policy's training metadata.

This is the "design in the browser -> train -> download -> run on a real robot" path.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("roboforge.export")


class _ActorWrapper:
    """Wrap an SB3 PPO policy so ONNX export yields a clean obs -> deterministic action graph."""
    def __init__(self, policy):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, sb3_policy):
                super().__init__()
                self.policy = sb3_policy

            def forward(self, obs):
                # Deterministic mean action (what we use when deploying).
                actions, _, _ = self.policy(obs, deterministic=True)
                return actions

        self.net = _Net(policy)


def export_policy_onnx_bytes(model, obs_dim: int = 16) -> bytes:
    """Return the ONNX graph bytes for a loaded SB3 PPO model's actor."""
    import torch

    policy = model.policy
    policy.set_training_mode(False)
    wrapper = _ActorWrapper(policy).net
    dummy = torch.zeros((1, obs_dim), dtype=torch.float32)
    buf = io.BytesIO()
    try:
        torch.onnx.export(
            wrapper, dummy, buf,
            input_names=["observation"], output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
    except TypeError:
        # Older torch without the `dynamo` kwarg uses the legacy exporter by default.
        buf = io.BytesIO()
        torch.onnx.export(
            wrapper, dummy, buf,
            input_names=["observation"], output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
    return buf.getvalue()


_ROS_NODE_TEMPLATE = '''#!/usr/bin/env python3
"""
RoboForge trained policy -> ROS 2 node.
Loads policy.onnx and drives a differential-drive robot: subscribes /scan + /odom, publishes /cmd_vel.

Deps on the robot:  pip install onnxruntime numpy   (plus a ROS 2 install: rclpy)
Run:                python3 roboforge_policy_node.py            # expects policy.onnx beside it
Topics:             sub /scan (LaserScan), /odom (Odometry)  ·  pub /cmd_vel (Twist)
"""
import math, os
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# --- Policy config baked in at export time ---
TARGET = {target!r}              # [x, y] goal the policy was trained toward
MAX_LIN = {max_lin}              # m/s
MAX_ANG = {max_ang}              # rad/s
BEAM_ANGLES_DEG = [-180, -135, -90, -45, 0, 45, 90, 135]
LIDAR_MAX = 3.0                  # m (observation normalization)

def quat_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

class PolicyNode(Node):
    def __init__(self):
        super().__init__("roboforge_policy_node")
        here = os.path.dirname(os.path.abspath(__file__))
        self.sess = ort.InferenceSession(os.path.join(here, "policy.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.scan = None
        self.odom = None
        self.create_subscription(LaserScan, "/scan", self._scan, 10)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(0.05, self._tick)  # 20 Hz control
        self.get_logger().info("RoboForge policy node up. Target=%s" % (TARGET,))

    def _scan(self, msg): self.scan = msg
    def _odom(self, msg): self.odom = msg

    def _sample_beams(self):
        if self.scan is None or not self.scan.ranges:
            return [1.0] * 8
        n = len(self.scan.ranges)
        amin, ainc = self.scan.angle_min, self.scan.angle_increment
        out = []
        for deg in BEAM_ANGLES_DEG:
            ang = math.radians(deg)
            idx = int(round((ang - amin) / ainc)) if ainc else 0
            if 0 <= idx < n:
                r = self.scan.ranges[idx]
                r = LIDAR_MAX if (r != r or r == float("inf")) else min(r, LIDAR_MAX)
            else:
                r = LIDAR_MAX
            out.append(max(0.0, min(1.0, r / LIDAR_MAX)))
        return out

    def _obs(self):
        o = self.odom
        px = o.pose.pose.position.x; py = o.pose.pose.position.y
        q = o.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        vx = o.twist.twist.linear.x; wz = o.twist.twist.angular.z
        dx = TARGET[0] - px; dy = TARGET[1] - py
        cy, sy = math.cos(yaw), math.sin(yaw)
        x_rel = dx * cy + dy * sy
        y_rel = -dx * sy + dy * cy
        dist = math.hypot(dx, dy)
        heading_err = math.atan2(y_rel, x_rel)
        beams = self._sample_beams()
        obs = [x_rel, y_rel, min(dist, 15.0), heading_err / math.pi,
               max(-1, min(1, vx / MAX_LIN)), max(-1, min(1, wz / MAX_ANG))]
        obs += beams
        obs += [max(-2, min(2, px / 5.0)), max(-2, min(2, py / 5.0))]
        return np.asarray([obs], dtype=np.float32)

    def _tick(self):
        if self.odom is None:
            return
        act = self.sess.run(None, {{self.in_name: self._obs()}})[0][0]
        t = Twist()
        t.linear.x = float(np.clip(act[0], -1, 1) * MAX_LIN)
        t.angular.z = float(np.clip(act[1], -1, 1) * MAX_ANG)
        self.pub.publish(t)

def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
'''


def _readme(meta: Dict[str, Any], target, max_lin, max_ang) -> str:
    return f"""# RoboForge Policy Package — {meta.get('name', meta.get('model_id', 'policy'))}

A PPO policy trained in RoboForge, exported to run on a real (or simulated) ROS 2 robot.

## Contents
- `policy.onnx` — actor network. Input `observation` float32[16], output `action` float32[2] in [-1, 1] = [forward, turn].
- `roboforge_policy_node.py` — ROS 2 node: subscribes `/scan`, `/odom`; publishes `/cmd_vel`.
- `metadata.json` — full training metadata.

## Observation layout (16)
0-1 target vector in robot frame (m) · 2 distance to target (m) · 3 heading error /pi ·
4 linear speed /max · 5 angular speed /max · 6-13 eight LiDAR beams [-180..135 deg] normalized by 3 m ·
14-15 global x,y /5.

## Action (2), each in [-1, 1]
0 forward (× {max_lin} m/s) · 1 turn (× {max_ang} rad/s)

## Run on a robot
```bash
pip install onnxruntime numpy          # plus your ROS 2 (rclpy) install
python3 roboforge_policy_node.py       # policy.onnx must sit beside this file
```
The robot must publish `/scan` (sensor_msgs/LaserScan) and `/odom` (nav_msgs/Odometry) and accept
`/cmd_vel` (geometry_msgs/Twist). Trained toward target {list(target)}.

## Try it against this simulator (no real robot needed)
RoboForge's rosbridge server speaks the same topics at `ws://localhost:9090` — point a rosbridge
client there, or run this node on a ROS 2 machine bridged to it.
"""


def build_policy_package(model, meta, project) -> bytes:
    """Build the downloadable .zip (bytes) for a loaded PPO model + its metadata + project."""
    md = meta.model_dump() if hasattr(meta, "model_dump") else dict(meta or {})
    target = md.get("target_position") or [2.5, 0.5]
    target = [float(target[0]), float(target[1])]
    robot = getattr(project, "robot", None)
    max_lin = float(getattr(robot, "max_linear_velocity", 1.5) or 1.5)
    max_ang = float(getattr(robot, "max_angular_velocity", 3.0) or 3.0)

    onnx_bytes = export_policy_onnx_bytes(model, obs_dim=16)
    node_src = _ROS_NODE_TEMPLATE.format(target=target, max_lin=max_lin, max_ang=max_ang)
    readme = _readme(md, target, max_lin, max_ang)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("policy.onnx", onnx_bytes)
        z.writestr("roboforge_policy_node.py", node_src)
        z.writestr("README.md", readme)
        z.writestr("metadata.json", json.dumps(md, indent=2))
    return buf.getvalue()
