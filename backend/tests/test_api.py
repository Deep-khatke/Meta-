import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_check():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"

def test_get_project():
    res = client.get("/api/project")
    assert res.status_code == 200
    data = res.json()
    assert "schema_version" in data
    assert "robot" in data
    assert "objects" in data

def test_simulation_controls():
    # Play
    res = client.post("/api/simulation/control", json={"action": "play"})
    assert res.status_code == 200
    assert res.json()["status"] == "running"

    # Pause
    res = client.post("/api/simulation/control", json={"action": "pause"})
    assert res.status_code == 200
    assert res.json()["status"] == "paused"

    # Step
    res = client.post("/api/simulation/control", json={"action": "step"})
    assert res.status_code == 200

    # Reset
    res = client.post("/api/simulation/control", json={"action": "reset"})
    assert res.status_code == 200
    assert res.json()["status"] == "stopped"

def test_teleop_endpoint():
    res = client.post("/api/simulation/teleop", json={"linear": 0.5, "angular": -0.2})
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

def test_add_and_delete_object():
    add_res = client.post("/api/scene/objects", json={
        "name": "API Test Sphere",
        "type": "sphere",
        "position": [0.5, 0.5, 0.3],
        "size": [0.2],
        "color": "#ff5500",
        "mass": 1.5
    })
    assert add_res.status_code == 200
    new_obj = add_res.json()
    obj_id = new_obj["id"]

    # Delete it
    del_res = client.delete(f"/api/scene/objects/{obj_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

def test_duplicate_object():
    add_res = client.post("/api/scene/objects", json={
        "name": "BoxToDup",
        "type": "box",
        "position": [0.5, 0.5, 0.3],
        "size": [0.2, 0.2, 0.2],
        "mass": 1.0
    })
    assert add_res.status_code == 200
    orig_id = add_res.json()["id"]

    dup_res = client.post(f"/api/scene/objects/{orig_id}/duplicate")
    assert dup_res.status_code == 200
    dup_obj = dup_res.json()
    assert dup_obj["id"] != orig_id
    assert "Copy" in dup_obj["name"]

    # Clean up
    client.delete(f"/api/scene/objects/{orig_id}")
    client.delete(f"/api/scene/objects/{dup_obj['id']}")

def test_scene_export_and_import():
    exp_res = client.get("/api/scene/export")
    assert exp_res.status_code == 200
    scene_data = exp_res.json()
    assert "schema_version" in scene_data

    # Modify and re-import
    scene_data["name"] = "Imported Validation Scene"
    imp_res = client.post("/api/scene/import", json=scene_data)
    assert imp_res.status_code == 200
    assert imp_res.json()["status"] == "imported"

def test_import_urdf_api():
    sample_urdf = """<?xml version="1.0"?>
    <robot name="api_arm">
      <link name="base_link">
        <visual><geometry><box size="0.2 0.2 0.05"/></geometry></visual>
      </link>
      <link name="link_1">
        <visual><geometry><cylinder radius="0.04" length="0.3"/></geometry></visual>
      </link>
      <joint name="j1" type="revolute">
        <parent link="base_link"/><child link="link_1"/>
        <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="50.0" velocity="2.0"/>
      </joint>
    </robot>
    """
    res = client.post("/api/robot/import-urdf", json={"urdf_xml": sample_urdf, "set_as_active_robot": True})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("success", "warning")
    assert data["total_links"] == 2
    assert data["total_joints"] == 1

def test_joint_control_api():
    res = client.post("/api/robot/joint-control", json={"joint_name": "j1", "target": 0.5})
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
