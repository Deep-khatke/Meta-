import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.ros2_bridge import ROS2Bridge, ros2_bag_recorder, SwarmOrchestrator

client = TestClient(app)

def test_ros2_topics_list():
    res = client.get("/api/ros2/topics")
    assert res.status_code == 200
    data = res.json()
    assert data["node"] == "/roboforge_bridge_node"
    topic_names = [t["topic"] for t in data["topics"]]
    assert "/odom" in topic_names
    assert "/scan" in topic_names
    assert "/imu/data" in topic_names
    assert "/joint_states" in topic_names
    assert "/clock" in topic_names
    assert "/tf" in topic_names
    assert "/cmd_vel" in topic_names

def test_ros2_messages():
    # Test odom
    res_odom = client.get("/api/ros2/topic/odom")
    assert res_odom.status_code == 200
    odom_data = res_odom.json()
    assert "header" in odom_data
    assert odom_data["child_frame_id"] == "base_footprint"
    assert "pose" in odom_data

    # Test clock
    res_clock = client.get("/api/ros2/topic/clock")
    assert res_clock.status_code == 200
    assert "clock" in res_clock.json()

    # Test joint states
    res_js = client.get("/api/ros2/topic/joint_states")
    assert res_js.status_code == 200
    assert "name" in res_js.json()

    # Test TF2 tree
    res_tf = client.get("/api/ros2/tf")
    assert res_tf.status_code == 200
    tf_data = res_tf.json()
    assert "transforms" in tf_data
    child_frames = [t["child_frame_id"] for t in tf_data["transforms"]]
    assert "odom" in child_frames
    assert "base_footprint" in child_frames
    assert "base_link" in child_frames

def test_ros2_cmd_vel():
    payload = {
        "linear": {"x": 0.35, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": -0.5}
    }
    res = client.post("/api/ros2/cmd_vel", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["teleop"]["linear"] > 0
    assert data["teleop"]["angular"] < 0

def test_ros2_bag_recording_lifecycle():
    # Check status
    res_stat = client.get("/api/ros2/bag/status")
    assert res_stat.status_code == 200

    # Start recording
    res_start = client.post("/api/ros2/bag/start", json={"name": "test_bag_suite"})
    assert res_start.status_code == 200
    start_data = res_start.json()
    assert "recording" in start_data["status"]
    bag_id = start_data["bag_id"]

    # Check status again
    res_active = client.get("/api/ros2/bag/status")
    assert res_active.status_code == 200
    assert res_active.json()["is_recording"] is True

    # Stop recording
    res_stop = client.post("/api/ros2/bag/stop")
    assert res_stop.status_code == 200
    stop_data = res_stop.json()
    assert "saved" in stop_data["status"]

    # List bags
    res_list = client.get("/api/ros2/bags")
    assert res_list.status_code == 200
    bags = res_list.json()["bags"]
    assert any(b["bag_id"] == bag_id for b in bags)

    # Get specific bag
    res_get = client.get(f"/api/ros2/bag/{bag_id}")
    assert res_get.status_code == 200
    assert res_get.json()["metadata"]["bag_id"] == bag_id

    # Delete bag
    res_del = client.delete(f"/api/ros2/bag/{bag_id}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "ok"

def test_swarm_deployment():
    # Deploy v_shape formation
    res_deploy = client.post("/api/swarm/deploy", json={
        "formation": "v_shape",
        "agent_count": 4,
        "spacing": 1.2
    })
    assert res_deploy.status_code == 200
    deploy_data = res_deploy.json()
    assert deploy_data["status"] == "ok"
    assert deploy_data["swarm"]["formation"] == "v_shape"
    assert len(deploy_data["swarm"]["followers"]) == 3

    # Check status
    res_status = client.get("/api/swarm/status")
    assert res_status.status_code == 200
    assert res_status.json()["active"] is True
    assert "v_shape" in res_status.json()["available_formations"]

    # Clear swarm
    res_clear = client.post("/api/swarm/clear")
    assert res_clear.status_code == 200
    assert res_clear.json()["status"] == "ok"

    res_status_after = client.get("/api/swarm/status")
    assert res_status_after.json()["active"] is False
