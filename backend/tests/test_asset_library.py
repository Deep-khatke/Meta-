import copy
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.models import Project
from backend.project_manager import get_default_project
from backend.asset_library import AssetLibrary
from backend.asset_models import CreateAssetRequest, ApplyAssetRequest

client = TestClient(app)

def test_asset_library_initialization(tmp_path):
    lib = AssetLibrary(storage_dir=tmp_path)
    assets = lib.list_assets()
    assert len(assets) >= 15 # At least 3 mobile, 2 arms, 5 obstacles, 4 sensors, 4 templates

    # Check categories
    robots = lib.list_assets(category="robot")
    assert len(robots) >= 4
    obstacles = lib.list_assets(category="obstacle")
    assert len(obstacles) >= 5
    sensors = lib.list_assets(category="sensor")
    assert len(sensors) >= 4
    templates = lib.list_assets(category="template")
    assert len(templates) >= 4

def test_asset_search_and_tag_filtering(tmp_path):
    lib = AssetLibrary(storage_dir=tmp_path)
    
    # Search by keyword
    results = lib.list_assets(search="slalom")
    assert len(results) >= 2 # pylon and slalom template
    for a in results:
        assert "slalom" in a.name.lower() or "slalom" in a.description.lower() or any("slalom" in t for t in a.tags)

    # Filter by tag
    articulated = lib.list_assets(tag="articulated")
    assert len(articulated) >= 2

def test_custom_asset_crud_and_persistence(tmp_path):
    lib = AssetLibrary(storage_dir=tmp_path)
    req = CreateAssetRequest(
        name="Custom Test Box",
        category="obstacle",
        description="A user created test obstacle",
        tags=["test", "custom"],
        data={"name": "CustomBox", "type": "box", "size": [0.4, 0.4, 0.4], "mass": 5.0}
    )
    created = lib.save_custom_asset(req)
    assert created.id.startswith("custom_")
    assert created.name == "Custom Test Box"
    assert not created.is_built_in

    # Duplicate asset
    clone = lib.duplicate_asset(created.id, new_name="Custom Test Box Clone")
    assert clone.name == "Custom Test Box Clone"
    assert clone.id != created.id

    # Verify persistence by reloading another instance
    lib2 = AssetLibrary(storage_dir=tmp_path)
    loaded = lib2.get_asset(created.id)
    assert loaded is not None
    assert loaded.name == "Custom Test Box"

    # Built-in deletion prevention
    with pytest.raises(ValueError):
        lib.delete_custom_asset("asset_robot_forgebot")

    # Delete custom asset
    res = lib.delete_custom_asset(created.id)
    assert res is True
    assert lib.get_asset(created.id) is None

def test_robot_validation_mobile(tmp_path):
    lib = AssetLibrary(storage_dir=tmp_path)
    
    # Valid mobile robot
    valid_data = {
        "id": "test_bot",
        "name": "TestBot",
        "initial_position": [0.0, 0.0, 0.07],
        "initial_yaw": 0.0,
        "chassis_size": [0.2, 0.15, 0.05],
        "wheel_radius": 0.06,
        "wheel_width": 0.025,
        "wheel_base": 0.28,
        "caster_radius": 0.02,
        "chassis_mass": 3.0,
        "wheel_mass": 0.3,
        "max_linear_velocity": 1.5,
        "max_angular_velocity": 3.0
    }
    v_res = lib.validate_robot_model(valid_data, robot_type="mobile")
    assert v_res.valid is True
    assert len(v_res.errors) == 0

    # Invalid mobile robot (negative dimension)
    invalid_data = dict(valid_data, chassis_size=[-0.2, 0.15, 0.05])
    v_bad = lib.validate_robot_model(invalid_data, robot_type="mobile")
    assert v_bad.valid is False
    assert any("chassis dimensions" in e.lower() for e in v_bad.errors)

def test_robot_validation_articulated(tmp_path):
    lib = AssetLibrary(storage_dir=tmp_path)

    # Valid 2-link arm
    valid_arm = {
        "name": "TestArm",
        "root_link": "base_link",
        "links": [
            {"name": "base_link", "visual": {"type": "cylinder", "size": [0.2, 0.05], "origin_pos": [0,0,0], "origin_rpy": [0,0,0], "color": "#fff"}},
            {"name": "link_1", "visual": {"type": "box", "size": [0.2, 0.04, 0.04], "origin_pos": [0.2,0,0], "origin_rpy": [0,0,0], "color": "#fff"}}
        ],
        "joints": [
            {
                "name": "joint_1",
                "type": "revolute",
                "parent_link": "base_link",
                "child_link": "link_1",
                "origin_pos": [0, 0, 0.1],
                "axis": [0, 0, 1],
                "limit": {"lower": -3.14, "upper": 3.14, "effort": 50.0, "velocity": 5.0}
            }
        ]
    }
    v_arm = lib.validate_robot_model(valid_arm, robot_type="articulated")
    assert v_arm.valid is True

    # Invalid joint missing parent link
    bad_arm = copy.deepcopy(valid_arm)
    bad_arm["joints"][0]["parent_link"] = "non_existent_link"
    v_bad = lib.validate_robot_model(bad_arm, robot_type="articulated")
    assert v_bad.valid is False
    assert any("non-existent parent" in e.lower() for e in v_bad.errors)

    # Inverted joint limits
    bad_limits = copy.deepcopy(valid_arm)
    bad_limits["joints"][0]["limit"] = {"lower": 3.14, "upper": -3.14, "effort": 50.0, "velocity": 5.0}
    v_bad_lim = lib.validate_robot_model(bad_limits, robot_type="articulated")
    assert v_bad_lim.valid is False
    assert any("lower limit" in e.lower() for e in v_bad_lim.errors)

def test_apply_asset_to_project(tmp_path):
    lib = AssetLibrary(storage_dir=tmp_path)
    proj = get_default_project()
    initial_obs_count = len(proj.objects)

    # Apply obstacle
    updated = lib.apply_asset_to_project("asset_obs_cargo_crate", proj)
    assert len(updated.objects) == initial_obs_count + 1
    assert any(o.name == "Cargo Crate" for o in updated.objects)

    # Apply articulated robot
    updated_robot = lib.apply_asset_to_project("asset_robot_2link_arm", updated)
    assert updated_robot.generic_robot is not None
    assert updated_robot.generic_robot.name == "TwoLinkPlanarArm"

    # Apply scene template
    tpl_proj = lib.apply_asset_to_project("asset_tpl_slalom", proj)
    assert "Slalom" in tpl_proj.name
    assert len(tpl_proj.objects) >= 5

def test_asset_api_endpoints():
    # 1. List assets
    res = client.get("/api/assets")
    assert res.status_code == 200
    data = res.json()
    assert "assets" in data
    assert data["total"] >= 15

    # 2. Filter by category
    res_rob = client.get("/api/assets?category=robot")
    assert res_rob.status_code == 200
    assert all(a["category"] == "robot" for a in res_rob.json()["assets"])

    # 3. Validate robot API
    val_res = client.post("/api/assets/validate-robot", json={
        "type": "mobile",
        "robot": {
            "id": "bot_api",
            "name": "BotAPI",
            "initial_position": [0.0, 0.0, 0.07],
            "initial_yaw": 0.0,
            "chassis_size": [0.2, 0.15, 0.05],
            "wheel_radius": 0.06,
            "wheel_width": 0.025,
            "wheel_base": 0.28,
            "caster_radius": 0.02,
            "chassis_mass": 3.0,
            "wheel_mass": 0.3,
            "max_linear_velocity": 1.5,
            "max_angular_velocity": 3.0
        }
    })
    assert val_res.status_code == 200
    assert val_res.json()["valid"] is True

    # 4. Apply template asset via API
    apply_res = client.post("/api/assets/asset_tpl_mobile_nav/apply")
    assert apply_res.status_code == 200
    assert apply_res.json()["status"] == "applied"
