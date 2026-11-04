"""rosbridge v2 protocol server for RoboForge.

Real ROS tooling speaks the rosbridge v2 JSON-over-WebSocket protocol (the protocol of
`rosbridge_suite`). By implementing it here, external clients connect to the live MuJoCo
simulation exactly as they would to a ROS 2 robot running `rosbridge_server` — no ROS install
needed on this machine:

  * Foxglove Studio  -> "Rosbridge" connection, ws://localhost:9090
  * roslibpy / roslibjs (Python / web ROS clients)
  * any ROS 2 graph, via a rosbridge client node

Published (sim -> clients): /odom, /scan, /imu, /joint_states, /tf, /clock
Subscribed (clients -> sim): /cmd_vel (geometry_msgs/Twist), /goal_pose (geometry_msgs/PoseStamped),
                             /joint_commands (sensor_msgs/JointState or std_msgs/Float64MultiArray)
Services: /rosapi/topics, /rosapi/topic_type, /rosapi/nodes, /rosapi/get_time
Any other topic a client publishes is relayed to subscribed clients (hub behavior).

Binds to 127.0.0.1 by default. Set ROSBRIDGE_HOST=0.0.0.0 to accept robots on your LAN
(note: that exposes robot control to the network).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Set

from backend.ros2_bridge import ROS2Bridge

logger = logging.getLogger("roboforge.rosbridge")


def _normalize(obj: Any) -> Any:
    """Convert RoboForge's flat header fields into standard ROS 2 JSON
    ({stamp: {sec, nanosec}, frame_id}) recursively."""
    if isinstance(obj, dict):
        if "stamp_sec" in obj and "stamp_nanosec" in obj:
            return {"stamp": {"sec": int(obj["stamp_sec"]), "nanosec": int(obj["stamp_nanosec"])},
                    "frame_id": obj.get("frame_id", "")}
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj


class RosbridgeServer:
    PUBLISH_HZ = 20.0

    def __init__(self, engine: Any, agent_interface: Any, port: int = 9090, host: Optional[str] = None):
        self.engine = engine
        self.agent_interface = agent_interface
        self.port = int(os.getenv("ROSBRIDGE_PORT", port))
        self.host = host or os.getenv("ROSBRIDGE_HOST", "127.0.0.1")
        self._server = None
        self._pub_task: Optional[asyncio.Task] = None
        # client -> {topic: {"throttle_ms": int, "last": float}}
        self._subs: Dict[Any, Dict[str, Dict[str, float]]] = {}
        self._advertised: Dict[str, str] = {}  # client-advertised topic -> type
        self.stats = {"messages_in": 0, "messages_out": 0, "cmd_vel_received": 0, "started_at": None,
                      "last_cmd_vel": None, "error": None}

        # topic -> (type, builder(state) -> msg | None)
        self.sim_topics: Dict[str, tuple] = {
            "/odom": ("nav_msgs/msg/Odometry", lambda s: ROS2Bridge.serialize_odometry(s)),
            "/scan": ("sensor_msgs/msg/LaserScan", lambda s: ROS2Bridge.serialize_laser_scan(s)),
            "/imu": ("sensor_msgs/msg/Imu", self._imu),
            "/imu/data": ("sensor_msgs/msg/Imu", self._imu),
            "/joint_states": ("sensor_msgs/msg/JointState", lambda s: ROS2Bridge.serialize_joint_states(s)),
            "/tf": ("tf2_msgs/msg/TFMessage", lambda s: ROS2Bridge.serialize_tf(s)),
            "/clock": ("rosgraph_msgs/msg/Clock", lambda s: ROS2Bridge.serialize_clock(s.sim_time)),
        }
        self.input_topics: Dict[str, str] = {
            "/cmd_vel": "geometry_msgs/msg/Twist",
            "/goal_pose": "geometry_msgs/msg/PoseStamped",
            "/joint_commands": "sensor_msgs/msg/JointState",
        }

    @staticmethod
    def _imu(s):
        m = ROS2Bridge.serialize_imu(s)
        if m is not None:
            m.setdefault("orientation_covariance", [0.0] * 9)
            m.setdefault("angular_velocity_covariance", [0.0] * 9)
            m.setdefault("linear_acceleration_covariance", [0.0] * 9)
        return m

    # ------------------------------------------------------------------ lifecycle
    async def start(self):
        from websockets.asyncio.server import serve
        try:
            self._server = await serve(self._handle_client, self.host, self.port, max_size=8 * 1024 * 1024)
            self._pub_task = asyncio.create_task(self._publish_loop())
            self.stats["started_at"] = time.time()
            logger.info("rosbridge v2 server listening on ws://%s:%d", self.host, self.port)
        except OSError as e:
            self.stats["error"] = f"Port {self.port} unavailable: {e}"
            logger.warning("rosbridge server could not start: %s", e)

    async def stop(self):
        if self._pub_task:
            self._pub_task.cancel()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._server is not None and self.stats["error"] is None,
            "url": f"ws://{'localhost' if self.host in ('127.0.0.1', '0.0.0.0') else self.host}:{self.port}",
            "host": self.host,
            "port": self.port,
            "clients": len(self._subs),
            "subscriptions": sorted({t for subs in self._subs.values() for t in subs}),
            "published_topics": {t: v[0] for t, v in self.sim_topics.items()},
            "subscribed_topics": dict(self.input_topics),
            "client_topics": dict(self._advertised),
            **{k: v for k, v in self.stats.items()},
        }

    # ------------------------------------------------------------------ protocol
    async def _send(self, ws, payload: Dict[str, Any]):
        try:
            await ws.send(json.dumps(payload))
            self.stats["messages_out"] += 1
        except Exception:
            pass

    async def _status(self, ws, level: str, msg: str, op_id: Optional[str] = None):
        p = {"op": "status", "level": level, "msg": msg}
        if op_id:
            p["id"] = op_id
        await self._send(ws, p)

    async def _handle_client(self, ws):
        self._subs[ws] = {}
        logger.info("rosbridge client connected (%d total)", len(self._subs))
        try:
            async for raw in ws:
                self.stats["messages_in"] += 1
                try:
                    msg = json.loads(raw)
                except Exception:
                    await self._status(ws, "error", "Invalid JSON")
                    continue
                try:
                    await self._dispatch(ws, msg)
                except Exception as e:
                    logger.warning("rosbridge op %s failed: %s", msg.get("op"), e)
                    await self._status(ws, "error", f"{msg.get('op')} failed: {e}", msg.get("id"))
        except Exception:
            pass
        finally:
            self._subs.pop(ws, None)
            logger.info("rosbridge client disconnected (%d left)", len(self._subs))

    async def _dispatch(self, ws, m: Dict[str, Any]):
        op = m.get("op")
        topic = m.get("topic")
        if op == "subscribe":
            known = topic in self.sim_topics or topic in self._advertised or topic in self.input_topics
            if not known:
                await self._status(ws, "warning", f"Topic {topic} not yet published; subscribed anyway.", m.get("id"))
            self._subs[ws][topic] = {"throttle_ms": float(m.get("throttle_rate", 0) or 0), "last": 0.0}
        elif op == "unsubscribe":
            self._subs[ws].pop(topic, None)
        elif op == "advertise":
            self._advertised[topic] = m.get("type", "")
        elif op == "unadvertise":
            self._advertised.pop(topic, None)
        elif op == "publish":
            await self._on_publish(ws, topic, m.get("msg") or {})
        elif op == "call_service":
            await self._on_service(ws, m)
        elif op in ("set_level", "auth"):
            pass
        else:
            await self._status(ws, "warning", f"Unsupported op '{op}'", m.get("id"))

    async def _on_publish(self, ws, topic: str, msg: Dict[str, Any]):
        if topic == "/cmd_vel":
            cmd = ROS2Bridge.parse_cmd_vel(msg)
            # Manual command from ROS takes over from any autonomous controller.
            try:
                if getattr(self.agent_interface, "active_trajectory_controller", None):
                    self.agent_interface.cancel_motion_trajectory()
            except Exception:
                pass
            if self.engine.status != "running":
                self.engine.play()
            self.engine.set_teleop(cmd.linear, cmd.angular)
            self.stats["cmd_vel_received"] += 1
            self.stats["last_cmd_vel"] = {"linear": cmd.linear, "angular": cmd.angular, "t": time.time()}
        elif topic == "/goal_pose":
            x, y, _yaw = ROS2Bridge.parse_goal_pose(msg)
            try:
                self.engine.disengage_rl_policy()
            except Exception:
                pass
            self.agent_interface.execute_motion_trajectory(target_pos=[x, y, 0.0])
        elif topic == "/joint_commands":
            targets = ROS2Bridge.parse_joint_commands(msg)
            if targets:
                self.engine.joint_targets.update(targets)
        # Relay to any other subscribed clients (hub behaviour, like a ROS topic).
        payload = {"op": "publish", "topic": topic, "msg": msg}
        for client, subs in list(self._subs.items()):
            if client is not ws and topic in subs:
                await self._send(client, payload)

    async def _on_service(self, ws, m: Dict[str, Any]):
        svc = m.get("service")
        args = m.get("args") or {}
        all_topics = {**{t: v[0] for t, v in self.sim_topics.items()}, **self.input_topics, **self._advertised}
        values: Dict[str, Any]
        if svc == "/rosapi/topics":
            values = {"topics": list(all_topics.keys()), "types": list(all_topics.values())}
        elif svc == "/rosapi/topic_type":
            values = {"type": all_topics.get(args.get("topic"), "")}
        elif svc == "/rosapi/nodes":
            values = {"nodes": ["/roboforge_sim", "/rosbridge_websocket"]}
        elif svc == "/rosapi/get_time":
            sec, nsec = ROS2Bridge.sim_time_to_stamp(float(self.engine.sim_time))
            values = {"time": {"sec": sec, "nanosec": nsec}}
        else:
            await self._send(ws, {"op": "service_response", "service": svc, "id": m.get("id"),
                                  "values": f"Service {svc} not available", "result": False})
            return
        await self._send(ws, {"op": "service_response", "service": svc, "id": m.get("id"),
                              "values": values, "result": True})

    # ------------------------------------------------------------------ outbound stream
    async def _publish_loop(self):
        period = 1.0 / self.PUBLISH_HZ
        while True:
            try:
                await asyncio.sleep(period)
                wanted = {t for subs in self._subs.values() for t in subs if t in self.sim_topics}
                if not wanted:
                    continue
                state = self.engine.get_state()
                built: Dict[str, Optional[Dict[str, Any]]] = {}
                for t in wanted:
                    try:
                        raw = self.sim_topics[t][1](state)
                        built[t] = _normalize(raw) if raw is not None else None
                    except Exception as e:
                        logger.debug("rosbridge build %s failed: %s", t, e)
                        built[t] = None
                now = time.time() * 1000.0
                for client, subs in list(self._subs.items()):
                    for t, meta in subs.items():
                        msg = built.get(t)
                        if msg is None:
                            continue
                        if meta["throttle_ms"] and now - meta["last"] < meta["throttle_ms"]:
                            continue
                        meta["last"] = now
                        await self._send(client, {"op": "publish", "topic": t, "msg": msg})
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("rosbridge publish loop error: %s", e)
                await asyncio.sleep(0.5)
