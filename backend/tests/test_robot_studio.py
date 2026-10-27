import pytest
import math
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.models import (
    GenericRobotModel, LinkModel, JointModel, JointLimit,
    LinkGeom, LinkInertial, FKRequest, IKRequest
)
from backend.kinematics import compute_forward_kinematics, solve_inverse_kinematics
from backend.robot_validator import validate_generic_robot, auto_calculate_robot_inertials
from backend.urdf_importer import export_robot_urdf, parse_urdf_string
from backend.simulation import MuJoCoEngine
from backend.project_manager import get_default_project

client = TestClient(app)

def create_sample_2link_arm() -> GenericRobotModel:
    return GenericRobotModel(
        name="Test2LinkArm",
        root_link="base_link",
        base_position=[0.0, 0.0, 0.0],
        base_rpy=[0.0, 0.0, 0.0],
        is_fixed_base=True,
        links=[
            LinkModel(
                name="base_link",
                visual=LinkGeom(type="cylinder", size=[0.15, 0.05], origin_pos=[0, 0, 0.05], color="#334155"),
                collision=LinkGeom(type="cylinder", size=[0.15, 0.05], origin_pos=[0, 0, 0.05], color="#334155"),
                inertial=LinkInertial(mass=4.0, origin_pos=[0, 0, 0.05], ixx=0.04, iyy=0.04, izz=0.04)
            ),
            LinkModel(
                name="link_1",
                visual=LinkGeom(type="box", size=[0.20, 0.04, 0.04], origin_pos=[0.20, 0, 0], color="#2563eb"),
                collision=LinkGeom(type="box", size=[0.20, 0.04, 0.04], origin_pos=[0.20, 0, 0], color="#2563eb"),
                inertial=LinkInertial(mass=1.5, origin_pos=[0.20, 0, 0], ixx=0.02, iyy=0.02, izz=0.02)
            ),
            LinkModel(
                name="link_2",
                visual=LinkGeom(type="box", size=[0.15, 0.03, 0.03], origin_pos=[0.15, 0, 0], color="#f97316"),
                collision=LinkGeom(type="box", size=[0.15, 0.03, 0.03], origin_pos=[0.15, 0, 0], color="#f97316"),
                inertial=LinkInertial(mass=0.8, origin_pos=[0.15, 0, 0], ixx=0.01, iyy=0.01, izz=0.01)
            )
        ],
        joints=[
            JointModel(
                name="shoulder_joint",
                type="revolute",
                parent_link="base_link",
                child_link="link_1",
                origin_pos=[0.0, 0.0, 0.10],
                axis=[0.0, 0.0, 1.0],
                limit=JointLimit(lower=-3.14, upper=3.14, effort=50.0, velocity=5.0),
                damping=0.2,
                actuator_mode="position",
                kp=80.0,
                kv=8.0
            ),
            JointModel(
                name="elbow_joint",
                type="revolute",
                parent_link="link_1",
                child_link="link_2",
                origin_pos=[0.40, 0.0, 0.0],
                axis=[0.0, 0.0, 1.0],
                limit=JointLimit(lower=-2.8, upper=2.8, effort=30.0, velocity=5.0),
                damping=0.15,
                actuator_mode="position",
                kp=60.0,
                kv=6.0
            )
        ]
    )

def test_robot_model_construction_and_validation():
    arm = create_sample_2link_arm()
    res = validate_generic_robot(arm)
    assert res.valid is True
    assert len(res.errors) == 0

def test_validation_detects_disconnected_links():
    arm = create_sample_2link_arm()
    # Add an orphan link
    arm.links.append(LinkModel(
        name="orphan_link",
        visual=LinkGeom(type="sphere", size=[0.1], origin_pos=[0, 0, 0], color="#fff"),
        inertial=LinkInertial(mass=1.0)
    ))
    res = validate_generic_robot(arm)
    assert res.valid is False
    assert any(e.code == "DISCONNECTED_LINK" for e in res.errors)

def test_validation_detects_kinematic_cycle():
    arm = create_sample_2link_arm()
    # Add a loop joint link_2 -> base_link
    arm.joints.append(JointModel(
        name="loop_joint",
        type="revolute",
        parent_link="link_2",
        child_link="base_link",
        origin_pos=[0, 0, 0],
        axis=[0, 0, 1]
    ))
    res = validate_generic_robot(arm)
    assert res.valid is False
    assert any(e.code in ("KINEMATIC_CYCLE", "MULTIPLE_PARENTS_FOR_LINK") for e in res.errors)

def test_validation_detects_invalid_limits_and_inertials():
    arm = create_sample_2link_arm()
    # Invert joint limits: lower > upper
    arm.joints[0].limit = JointLimit(lower=3.14, upper=-3.14, effort=10.0, velocity=2.0)
    # Non-positive mass
    arm.links[1].inertial.mass = -2.0
    res = validate_generic_robot(arm)
    assert res.valid is False
    assert any(e.code == "INVALID_LIMIT_BOUNDS" for e in res.errors)
    assert any(e.code == "NON_POSITIVE_MASS" for e in res.errors)

def test_auto_calculate_inertials():
    arm = create_sample_2link_arm()
    # Clear inertials
    arm.links[0].inertial = None
    arm.links[1].inertial = None
    updated = auto_calculate_robot_inertials(arm)
    for l in updated.links:
        assert l.inertial is not None
        assert l.inertial.mass > 0
        assert l.inertial.ixx > 0
        assert l.inertial.iyy > 0
        assert l.inertial.izz > 0

def test_forward_kinematics_accuracy():
    arm = create_sample_2link_arm()
    # At zero angles: shoulder origin is at (0, 0, 0.1), link_1 is along X by 0.40m, link_2 extends by 0.30m
    # Nominal EE tip is at X = 0.40 + 0.30 = 0.70m, Z = 0.10m
    fk_zero = compute_forward_kinematics(arm, {"shoulder_joint": 0.0, "elbow_joint": 0.0})
    assert fk_zero.success is True
    assert "base_link" in fk_zero.link_poses
    assert "link_1" in fk_zero.link_poses
    assert "link_2" in fk_zero.link_poses
    assert abs(fk_zero.end_effector_position[0] - 0.70) < 0.05
    assert abs(fk_zero.end_effector_position[1] - 0.0) < 0.05
    assert abs(fk_zero.end_effector_position[2] - 0.10) < 0.05

    # 90 degree shoulder rotation (pi/2)
    fk_90 = compute_forward_kinematics(arm, {"shoulder_joint": math.pi / 2, "elbow_joint": 0.0})
    # Should now point along +Y
    assert abs(fk_90.end_effector_position[0] - 0.0) < 0.05
    assert abs(fk_90.end_effector_position[1] - 0.70) < 0.05

def test_inverse_kinematics_reachable_target():
    arm = create_sample_2link_arm()
    target = [0.35, 0.35, 0.10]
    ik_req = IKRequest(
        robot=arm,
        target_position=target,
        tolerance=0.005,
        max_iterations=100
    )
    res = solve_inverse_kinematics(ik_req)
    assert res.success is True
    assert res.status == "SUCCESS"
    assert res.error_distance <= 0.005

    # Verify that FK with solved joint angles actually reaches target
    fk = compute_forward_kinematics(arm, res.joint_solution)
    achieved = np.array(fk.end_effector_position)
    assert np.linalg.norm(achieved - np.array(target)) <= 0.005

def test_inverse_kinematics_unreachable_target_never_claims_success():
    arm = create_sample_2link_arm()
    # Target way outside arm envelope (e.g. 5.0m away)
    unreachable_target = [5.0, 5.0, 0.10]
    ik_req = IKRequest(
        robot=arm,
        target_position=unreachable_target,
        tolerance=0.005,
        max_iterations=50
    )
    res = solve_inverse_kinematics(ik_req)
    # Must NOT claim success!
    assert res.success is False
    assert res.status == "TARGET_OUT_OF_REACH"
    assert res.error_distance > 1.0
    assert "outside maximum theoretical workspace reach" in res.message

def test_inverse_kinematics_joint_limits_blocked():
    arm = create_sample_2link_arm()
    # Constrain shoulder and elbow strictly to [0, 0.1] rad
    arm.joints[0].limit = JointLimit(lower=0.0, upper=0.1, effort=50.0, velocity=5.0)
    arm.joints[1].limit = JointLimit(lower=0.0, upper=0.1, effort=30.0, velocity=5.0)

    # Request target behind the arm in negative X
    blocked_target = [-0.4, 0.0, 0.10]
    ik_req = IKRequest(
        robot=arm,
        target_position=blocked_target,
        tolerance=0.005,
        max_iterations=40
    )
    res = solve_inverse_kinematics(ik_req)
    # Must NOT claim success
    assert res.success is False
    assert res.status in ("JOINT_LIMITS_BLOCKED", "CONVERGENCE_FAILED")
    assert res.error_distance > 0.05

def test_prismatic_joint_support():
    gantry = GenericRobotModel(
        name="GantryRobot",
        root_link="base_link",
        is_fixed_base=True,
        links=[
            LinkModel(name="base_link", visual=LinkGeom(type="box", size=[0.5, 0.1, 0.05], origin_pos=[0, 0, 0], color="#333"), inertial=LinkInertial(mass=5.0)),
            LinkModel(name="slider", visual=LinkGeom(type="box", size=[0.05, 0.05, 0.05], origin_pos=[0, 0, 0], color="#09f"), inertial=LinkInertial(mass=1.0))
        ],
        joints=[
            JointModel(
                name="slide_joint",
                type="prismatic",
                parent_link="base_link",
                child_link="slider",
                origin_pos=[0.0, 0.0, 0.1],
                axis=[1.0, 0.0, 0.0],
                limit=JointLimit(lower=-0.4, upper=0.4, effort=100.0, velocity=2.0)
            )
        ]
    )
    val = validate_generic_robot(gantry)
    assert val.valid is True

    fk = compute_forward_kinematics(gantry, {"slide_joint": 0.25})
    assert fk.success is True
    assert abs(fk.end_effector_position[0] - 0.25) < 0.06

def test_urdf_export_and_reimport():
    arm = create_sample_2link_arm()
    urdf_xml = export_robot_urdf(arm)
    assert '<robot name="Test2LinkArm">' in urdf_xml
    assert '<joint name="shoulder_joint" type="revolute">' in urdf_xml

    reimported = parse_urdf_string(urdf_xml)
    assert reimported.status in ("success", "warning")
    assert reimported.robot is not None
    assert len(reimported.robot.links) == 3
    assert len(reimported.robot.joints) == 2

def test_simulation_actuator_limits_enforced():
    proj = get_default_project()
    arm = create_sample_2link_arm()
    # Set limit on shoulder joint
    arm.joints[0].limit = JointLimit(lower=-1.0, upper=1.0, effort=50.0, velocity=5.0)
    proj.generic_robot = arm

    engine = MuJoCoEngine(proj)
    xml = engine.generate_mjcf_xml()
    # Check that ctrlrange is present in actuator xml
    assert 'ctrlrange="-1.0 1.0"' in xml

    # Command beyond limit
    engine.set_joint_target("shoulder_joint", 3.0)
    engine.apply_control()
    act_id = engine.joint_actuator_ids.get("shoulder_joint")
    assert act_id is not None
    # Engine must have clamped command to upper limit 1.0
    assert engine.data.ctrl[act_id] <= 1.0

def test_api_robot_endpoints():
    arm = create_sample_2link_arm()

    # 1. Validate endpoint
    r_val = client.post("/api/robot/validate", json=arm.model_dump())
    assert r_val.status_code == 200
    assert r_val.json()["valid"] is True

    # 2. FK endpoint
    r_fk = client.post("/api/robot/kinematics/fk", json={
        "robot": arm.model_dump(),
        "joint_positions": {"shoulder_joint": 0.5, "elbow_joint": -0.5}
    })
    assert r_fk.status_code == 200
    assert r_fk.json()["success"] is True

    # 3. IK endpoint
    r_ik = client.post("/api/robot/kinematics/ik", json={
        "robot": arm.model_dump(),
        "target_position": [0.4, 0.2, 0.1],
        "tolerance": 0.01
    })
    assert r_ik.status_code == 200
    assert r_ik.json()["status"] in ("SUCCESS", "TARGET_OUT_OF_REACH", "JOINT_LIMITS_BLOCKED")

    # 4. Presets endpoint
    r_presets = client.get("/api/robot/presets")
    assert r_presets.status_code == 200
    assert len(r_presets.json()["presets"]) >= 3

    # 5. Export URDF endpoint
    r_exp = client.post("/api/robot/export-urdf", json=arm.model_dump())
    assert r_exp.status_code == 200
    assert "<robot" in r_exp.json()["urdf"]

    # 6. Set generic robot in simulation
    r_set = client.post("/api/robot/generic", json=arm.model_dump())
    assert r_set.status_code == 200
    assert r_set.json()["status"] == "updated"

    # 7. Get generic robot
    r_get = client.get("/api/robot/generic")
    assert r_get.status_code == 200
    assert r_get.json()["robot"] is not None

def test_collision_geometry_validation():
    arm = create_sample_2link_arm()
    # 1. Missing collision geometry generates warning
    arm.links[1].collision = None
    res = validate_generic_robot(arm)
    assert any(w.code == "MISSING_COLLISION_GEOM" for w in res.warnings)

    # 2. Non-positive collision dimension generates error
    arm.links[1].collision = LinkGeom(type="box", size=[-0.1, 0.05, 0.05])
    res2 = validate_generic_robot(arm)
    assert res2.valid is False
    assert any(e.code == "NON_POSITIVE_COLLISION_SIZE" for e in res2.errors)

    # 3. Excessively large collision size generates error
    arm.links[1].collision = LinkGeom(type="box", size=[30.0, 0.05, 0.05])
    res3 = validate_generic_robot(arm)
    assert res3.valid is False
    assert any(e.code == "EXCESSIVE_COLLISION_BOUNDS" for e in res3.errors)

def test_actuator_modes_and_parameters_validation():
    arm = create_sample_2link_arm()

    # Negative damping
    arm.joints[0].damping = -0.5
    res = validate_generic_robot(arm)
    assert res.valid is False
    assert any(e.code == "NEGATIVE_DAMPING" for e in res.errors)
    arm.joints[0].damping = 0.2

    # Negative armature
    arm.joints[0].armature = -0.1
    res2 = validate_generic_robot(arm)
    assert res2.valid is False
    assert any(e.code == "NEGATIVE_ARMATURE" for e in res2.errors)
    arm.joints[0].armature = 0.01

    # Velocity mode without velocity limit
    arm.joints[0].actuator_mode = "velocity"
    arm.joints[0].limit.velocity = 0.0
    res3 = validate_generic_robot(arm)
    assert res3.valid is False
    assert any(e.code == "NON_POSITIVE_VELOCITY_LIMIT" for e in res3.errors)

    # Torque mode without effort limit
    arm.joints[0].actuator_mode = "torque"
    arm.joints[0].limit.effort = 0.0
    res4 = validate_generic_robot(arm)
    assert res4.valid is False
    assert any(e.code == "NON_POSITIVE_EFFORT_LIMIT" for e in res4.errors)

def test_continuous_and_fixed_joint_mjcf_generation():
    robot = GenericRobotModel(
        name="HybridRobot",
        root_link="base",
        is_fixed_base=True,
        links=[
            LinkModel(name="base", visual=LinkGeom(type="box", size=[0.2, 0.2, 0.1]), collision=LinkGeom(type="box", size=[0.2, 0.2, 0.1]), inertial=LinkInertial(mass=5.0)),
            LinkModel(name="turret", visual=LinkGeom(type="cylinder", size=[0.1, 0.1]), collision=LinkGeom(type="cylinder", size=[0.1, 0.1]), inertial=LinkInertial(mass=1.5)),
            LinkModel(name="antenna", visual=LinkGeom(type="cylinder", size=[0.02, 0.3]), collision=LinkGeom(type="cylinder", size=[0.02, 0.3]), inertial=LinkInertial(mass=0.2))
        ],
        joints=[
            JointModel(
                name="turret_continuous_yaw",
                type="continuous",
                parent_link="base",
                child_link="turret",
                origin_pos=[0, 0, 0.15],
                axis=[0, 0, 1],
                actuator_mode="velocity",
                kv=15.0,
                limit=JointLimit(lower=0, upper=0, effort=25.0, velocity=8.0)
            ),
            JointModel(
                name="fixed_antenna_bracket",
                type="fixed",
                parent_link="turret",
                child_link="antenna",
                origin_pos=[0.05, 0, 0.2],
                axis=[0, 0, 1]
            )
        ]
    )

    val = validate_generic_robot(robot)
    assert val.valid is True

    proj = get_default_project()
    proj.generic_robot = robot
    engine = MuJoCoEngine(proj)
    xml = engine.generate_mjcf_xml()

    # Continuous joint shouldn't have range in joint tag
    assert '<joint name="turret_continuous_yaw"' in xml
    assert 'range=' not in xml.split('name="turret_continuous_yaw"')[1].split('/>')[0]
    # Actuator is velocity
    assert '<velocity name="act_turret_continuous_yaw"' in xml
    assert 'kv="15.0"' in xml
    assert 'ctrlrange="-8.0 8.0"' in xml
    # Fixed joint has no joint element inside antenna body
    antenna_body_xml = xml.split('name="antenna"')[1].split('</body')[0]
    assert '<joint' not in antenna_body_xml

    # Step physics to verify engine runs stably
    engine.set_joint_target("turret_continuous_yaw", 4.0)
    for _ in range(50):
        engine.step()
    state = engine.get_state()
    assert "turret_continuous_yaw" in state.joint_velocities
    assert abs(state.joint_velocities["turret_continuous_yaw"]) > 0.0

def test_save_robot_model_to_asset_library_endpoint():
    arm = create_sample_2link_arm()
    arm.name = "CustomPersistedArm"

    # Valid robot model persistence
    r_save = client.post("/api/assets/save-robot-model", json={
        "robot": arm.model_dump(),
        "name": "CustomPersistedArm",
        "description": "2-link articulated arm saved from studio"
    })
    assert r_save.status_code == 200
    asset = r_save.json()
    assert asset["name"] == "CustomPersistedArm"
    assert asset["category"] == "robot"
    assert asset["data"]["type"] == "articulated"
    assert asset["data"]["config"]["name"] == "CustomPersistedArm"

    # Verify asset is listed in /api/assets
    r_list = client.get("/api/assets?category=robot")
    assert r_list.status_code == 200
    asset_names = [a["name"] for a in r_list.json()["assets"]]
    assert "CustomPersistedArm" in asset_names

    # Invalid robot model must fail with 400 and validation errors
    arm.links.append(LinkModel(name="orphan_broken", visual=LinkGeom(type="sphere", size=[0.1])))
    r_save_invalid = client.post("/api/assets/save-robot-model", json={
        "robot": arm.model_dump(),
        "name": "InvalidRobot"
    })
    assert r_save_invalid.status_code == 400
    err_body = r_save_invalid.json()
    assert "Robot validation failed" in err_body["detail"]

