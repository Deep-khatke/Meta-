from __future__ import annotations
import math
from typing import List, Dict, Set, Optional, Tuple
import mujoco

from backend.models import (
    GenericRobotModel, LinkModel, JointModel, JointLimit,
    LinkGeom, LinkInertial, RobotValidationResult, RobotDiagnostic
)

def compute_link_inertial(geom: Optional[LinkGeom], mass: float = 1.0) -> LinkInertial:
    """
    Computes physically consistent principal moments of inertia for standard shapes.
    - Box: sizes are half-sizes [hx, hy, hz] (full dimensions are 2*hx, 2*hy, 2*hz)
    - Sphere: size is [radius]
    - Cylinder: size is [radius, half_height]
    """
    safe_mass = max(0.01, mass)
    if not geom or not geom.size:
        return LinkInertial(
            mass=safe_mass,
            origin_pos=[0.0, 0.0, 0.0],
            ixx=safe_mass * 0.01,
            iyy=safe_mass * 0.01,
            izz=safe_mass * 0.01
        )

    g_type = geom.type
    pos = list(geom.origin_pos) if geom.origin_pos else [0.0, 0.0, 0.0]

    if g_type == "box":
        hx = geom.size[0] if len(geom.size) > 0 else 0.1
        hy = geom.size[1] if len(geom.size) > 1 else 0.1
        hz = geom.size[2] if len(geom.size) > 2 else 0.1
        # Full dimensions: dx = 2*hx, dy = 2*hy, dz = 2*hz
        # Ixx = 1/12 * m * (dy^2 + dz^2) = 1/12 * m * 4 * (hy^2 + hz^2) = 1/3 * m * (hy^2 + hz^2)
        ixx = (1.0 / 3.0) * safe_mass * (hy * hy + hz * hz)
        iyy = (1.0 / 3.0) * safe_mass * (hx * hx + hz * hz)
        izz = (1.0 / 3.0) * safe_mass * (hx * hx + hy * hy)
    elif g_type == "sphere":
        r = geom.size[0] if len(geom.size) > 0 else 0.1
        # Solid sphere: I = 2/5 * m * r^2
        isp = 0.4 * safe_mass * (r * r)
        ixx, iyy, izz = isp, isp, isp
    elif g_type == "cylinder":
        r = geom.size[0] if len(geom.size) > 0 else 0.05
        hh = geom.size[1] if len(geom.size) > 1 else 0.1
        h = 2.0 * hh # full length
        # Cylinder aligned with Z axis:
        # Izz = 1/2 * m * r^2
        # Ixx = Iyy = 1/12 * m * (3*r^2 + h^2)
        izz = 0.5 * safe_mass * (r * r)
        ixx = (1.0 / 12.0) * safe_mass * (3.0 * r * r + h * h)
        iyy = ixx
    else:
        ixx = safe_mass * 0.01
        iyy = safe_mass * 0.01
        izz = safe_mass * 0.01

    return LinkInertial(
        mass=safe_mass,
        origin_pos=pos,
        ixx=float(max(1e-5, ixx)),
        iyy=float(max(1e-5, iyy)),
        izz=float(max(1e-5, izz))
    )

def auto_calculate_robot_inertials(robot: GenericRobotModel) -> GenericRobotModel:
    """Auto-populates physically consistent inertial properties for all robot links."""
    updated_links: List[LinkModel] = []
    for link in robot.links:
        geom = link.collision or link.visual
        existing_mass = link.inertial.mass if link.inertial else 1.0
        new_inertial = compute_link_inertial(geom, existing_mass)
        updated_links.append(link.model_copy(update={"inertial": new_inertial}))
    return robot.model_copy(update={"links": updated_links})

def validate_generic_robot(robot: GenericRobotModel) -> RobotValidationResult:
    """
    Performs comprehensive structural, kinematic, and physics validation of a robot model.
    Checks:
    - Root link existence and kinematic tree validity
    - Disconnected links and cycles/loops
    - Inertial mass and positive definite inertia tensors
    - Joint limits, axes, and parent/child associations
    - Actuator parameters (modes, gains, ranges)
    - MuJoCo dry-run compilation
    """
    errors: List[RobotDiagnostic] = []
    warnings: List[RobotDiagnostic] = []

    # 1. Basic Structure
    if not robot.links:
        errors.append(RobotDiagnostic(
            code="NO_LINKS",
            severity="error",
            target="robot",
            message="Robot model has no links. At least one link is required.",
            suggestion="Add a base link (e.g. 'base_link') with visual/collision geometry."
        ))
        return RobotValidationResult(
            valid=False,
            errors=errors,
            warnings=warnings,
            summary="Robot validation failed: No links defined."
        )

    links_by_name: Dict[str, LinkModel] = {l.name: l for l in robot.links}
    link_names: Set[str] = set()
    for link in robot.links:
        if not link.name or not link.name.strip():
            errors.append(RobotDiagnostic(
                code="EMPTY_LINK_NAME",
                severity="error",
                target=f"link:{link.name}",
                message="Encountered link with an empty or whitespace name.",
                suggestion="Specify a descriptive name for each link (e.g. 'base_link', 'arm_link1')."
            ))
        elif link.name in link_names:
            errors.append(RobotDiagnostic(
                code="DUPLICATE_LINK_NAME",
                severity="error",
                target=f"link:{link.name}",
                message=f"Duplicate link name '{link.name}'. Every link name must be unique.",
                suggestion=f"Rename one of the '{link.name}' links to be unique."
            ))
        link_names.add(link.name)

    if robot.root_link not in link_names:
        errors.append(RobotDiagnostic(
            code="INVALID_ROOT_LINK",
            severity="error",
            target="robot:root_link",
            message=f"Root link '{robot.root_link}' does not exist in the list of links.",
            suggestion=f"Set root_link to one of the defined links (e.g. '{robot.links[0].name}')."
        ))

    # 2. Joint Graph & Topology
    joint_names: Set[str] = set()
    child_to_joint: Dict[str, JointModel] = {}
    parent_to_joints: Dict[str, List[JointModel]] = {}

    for joint in robot.joints:
        jname = joint.name
        if not jname or not jname.strip():
            errors.append(RobotDiagnostic(
                code="EMPTY_JOINT_NAME",
                severity="error",
                target=f"joint:{jname}",
                message="Encountered joint with an empty or whitespace name.",
                suggestion="Specify a unique name for each joint."
            ))
        elif jname in joint_names:
            errors.append(RobotDiagnostic(
                code="DUPLICATE_JOINT_NAME",
                severity="error",
                target=f"joint:{jname}",
                message=f"Duplicate joint name '{jname}'. Joint names must be unique.",
                suggestion=f"Rename joint '{jname}' to a unique identifier."
            ))
        joint_names.add(jname)

        # Parent / Child Link Check
        if joint.parent_link not in link_names:
            errors.append(RobotDiagnostic(
                code="UNKNOWN_PARENT_LINK",
                severity="error",
                target=f"joint:{jname}",
                message=f"Joint '{jname}' references non-existent parent link '{joint.parent_link}'.",
                suggestion="Select a valid parent link from the robot's links."
            ))
        if joint.child_link not in link_names:
            errors.append(RobotDiagnostic(
                code="UNKNOWN_CHILD_LINK",
                severity="error",
                target=f"joint:{jname}",
                message=f"Joint '{jname}' references non-existent child link '{joint.child_link}'.",
                suggestion="Select a valid child link from the robot's links."
            ))
        if joint.parent_link == joint.child_link:
            errors.append(RobotDiagnostic(
                code="JOINT_SELF_LOOP",
                severity="error",
                target=f"joint:{jname}",
                message=f"Joint '{jname}' has identical parent and child link '{joint.parent_link}' (self-loop).",
                suggestion="Parent and child links of a joint must be distinct."
            ))

        # Check for multiple incoming joints to same child
        if joint.child_link in child_to_joint:
            prev_j = child_to_joint[joint.child_link]
            errors.append(RobotDiagnostic(
                code="MULTIPLE_PARENTS_FOR_LINK",
                severity="error",
                target=f"link:{joint.child_link}",
                message=(
                    f"Link '{joint.child_link}' has multiple parent joints: '{prev_j.name}' and '{jname}'. "
                    "MuJoCo kinematic trees require each link to have at most one parent."
                ),
                suggestion="Remove the redundant joint so the robot forms a pure tree hierarchy."
            ))
        else:
            child_to_joint[joint.child_link] = joint

        parent_to_joints.setdefault(joint.parent_link, []).append(joint)

        # Joint Axis Check
        axis = joint.axis
        if len(axis) != 3 or (axis[0]**2 + axis[1]**2 + axis[2]**2) < 1e-6:
            errors.append(RobotDiagnostic(
                code="ZERO_JOINT_AXIS",
                severity="error",
                target=f"joint:{jname}",
                message=f"Joint '{jname}' has a zero or invalid axis vector {axis}.",
                suggestion="Set axis vector to a unit direction such as [0, 0, 1] for yaw or [0, 1, 0] for pitch."
            ))

        # Joint Limits Check
        if joint.type in ("revolute", "prismatic"):
            if not joint.limit:
                warnings.append(RobotDiagnostic(
                    code="MISSING_JOINT_LIMIT",
                    severity="warning",
                    target=f"joint:{jname}",
                    message=f"Joint '{jname}' of type '{joint.type}' has no explicit limit defined.",
                    suggestion="Add joint limits [lower, upper] to prevent unbounded motion."
                ))
            else:
                if joint.limit.lower >= joint.limit.upper:
                    errors.append(RobotDiagnostic(
                        code="INVALID_LIMIT_BOUNDS",
                        severity="error",
                        target=f"joint:{jname}",
                        message=(
                            f"Joint '{jname}' lower limit ({joint.limit.lower}) is greater than or equal to "
                            f"upper limit ({joint.limit.upper})."
                        ),
                        suggestion="Ensure lower limit is strictly less than upper limit."
                    ))
                if joint.limit.effort <= 0:
                    warnings.append(RobotDiagnostic(
                        code="NON_POSITIVE_EFFORT",
                        severity="warning",
                        target=f"joint:{jname}",
                        message=f"Joint '{jname}' maximum effort is {joint.limit.effort} <= 0.",
                        suggestion="Set effort limit to a positive number (e.g. 50.0 Nm)."
                    ))
                if joint.limit.velocity <= 0:
                    warnings.append(RobotDiagnostic(
                        code="NON_POSITIVE_VELOCITY",
                        severity="warning",
                        target=f"joint:{jname}",
                        message=f"Joint '{jname}' maximum velocity is {joint.limit.velocity} <= 0.",
                        suggestion="Set velocity limit to a positive number (e.g. 5.0 rad/s)."
                    ))

        # Actuator Parameters Check
        mode = joint.actuator_mode
        if mode not in ("position", "velocity", "torque", None):
            errors.append(RobotDiagnostic(
                code="INVALID_ACTUATOR_MODE",
                severity="error",
                target=f"joint:{jname}",
                message=f"Joint '{jname}' has unsupported actuator mode '{mode}'.",
                suggestion="Set actuator_mode to 'position', 'velocity', or 'torque'."
            ))
        elif mode == "position" and joint.kp <= 0:
            warnings.append(RobotDiagnostic(
                code="ZERO_POSITION_KP",
                severity="warning",
                target=f"joint:{jname}",
                message=f"Joint '{jname}' is in 'position' mode but has kp={joint.kp} <= 0.",
                suggestion="Set kp to a positive stiffness gain (e.g. 80.0) so the actuator tracks position."
            ))
        elif mode == "velocity":
            if joint.kv <= 0:
                warnings.append(RobotDiagnostic(
                    code="ZERO_VELOCITY_KV",
                    severity="warning",
                    target=f"joint:{jname}",
                    message=f"Joint '{jname}' is in 'velocity' mode but has kv={joint.kv} <= 0.",
                    suggestion="Set kv to a positive damping gain (e.g. 10.0)."
                ))
            if joint.limit and joint.limit.velocity <= 0:
                errors.append(RobotDiagnostic(
                    code="NON_POSITIVE_VELOCITY_LIMIT",
                    severity="error",
                    target=f"joint:{jname}",
                    message=f"Joint '{jname}' is in 'velocity' mode but velocity limit is {joint.limit.velocity} <= 0.",
                    suggestion="Set velocity limit to a positive number (e.g. 5.0 rad/s)."
                ))
        elif mode == "torque":
            if joint.limit and joint.limit.effort <= 0:
                errors.append(RobotDiagnostic(
                    code="NON_POSITIVE_EFFORT_LIMIT",
                    severity="error",
                    target=f"joint:{jname}",
                    message=f"Joint '{jname}' is in 'torque' mode but effort limit is {joint.limit.effort} <= 0.",
                    suggestion="Set effort limit to a positive torque capacity (e.g. 50.0 Nm)."
                ))

        if joint.damping < 0:
            errors.append(RobotDiagnostic(
                code="NEGATIVE_DAMPING",
                severity="error",
                target=f"joint:{jname}",
                message=f"Joint '{jname}' has negative damping ({joint.damping}).",
                suggestion="Damping must be non-negative (>= 0.0)."
            ))
        if joint.armature < 0:
            errors.append(RobotDiagnostic(
                code="NEGATIVE_ARMATURE",
                severity="error",
                target=f"joint:{jname}",
                message=f"Joint '{jname}' has negative armature ({joint.armature}).",
                suggestion="Armature inertia must be non-negative (>= 0.0)."
            ))

    # 3. Disconnected Links & Cycle Detection via Graph Traversal
    if robot.root_link in link_names:
        visited: Set[str] = set()
        in_stack: Set[str] = set()
        has_cycle = False

        def dfs(node: str):
            nonlocal has_cycle
            visited.add(node)
            in_stack.add(node)

            for child_jnt in parent_to_joints.get(node, []):
                child = child_jnt.child_link
                if child in in_stack:
                    has_cycle = True
                    errors.append(RobotDiagnostic(
                        code="KINEMATIC_CYCLE",
                        severity="error",
                        target=f"joint:{child_jnt.name}",
                        message=f"Kinematic loop detected involving link '{child}'. MuJoCo requires open-loop trees.",
                        suggestion="Remove closing loop joint to maintain a strict kinematic tree."
                    ))
                elif child not in visited:
                    dfs(child)

            in_stack.remove(node)

        dfs(robot.root_link)

        # Disconnected links check
        disconnected = link_names - visited
        for dlink in disconnected:
            errors.append(RobotDiagnostic(
                code="DISCONNECTED_LINK",
                severity="error",
                target=f"link:{dlink}",
                message=f"Link '{dlink}' is disconnected from root link '{robot.root_link}'.",
                suggestion=f"Connect link '{dlink}' to the tree with a joint or remove it."
            ))

    # 4. Link Inertial & Geometry Validation
    for link in robot.links:
        # Inertial checks
        inertial = link.inertial
        if not inertial:
            warnings.append(RobotDiagnostic(
                code="MISSING_INERTIAL",
                severity="warning",
                target=f"link:{link.name}",
                message=f"Link '{link.name}' is missing inertial parameters.",
                suggestion="Use 'Auto-Calculate Inertial' to compute accurate inertia from geometry."
            ))
        else:
            if inertial.mass <= 0:
                errors.append(RobotDiagnostic(
                    code="NON_POSITIVE_MASS",
                    severity="error",
                    target=f"link:{link.name}",
                    message=f"Link '{link.name}' has mass {inertial.mass} <= 0.",
                    suggestion="Set link mass to a positive value (e.g. 1.0 kg)."
                ))
            if inertial.ixx <= 0 or inertial.iyy <= 0 or inertial.izz <= 0:
                errors.append(RobotDiagnostic(
                    code="NON_POSITIVE_INERTIA",
                    severity="error",
                    target=f"link:{link.name}",
                    message=f"Link '{link.name}' has non-positive principal moments of inertia.",
                    suggestion="Auto-calculate inertia or set positive diagonal values."
                ))
            else:
                # Triangle inequality check: Ixx + Iyy >= Izz, etc.
                ixx, iyy, izz = inertial.ixx, inertial.iyy, inertial.izz
                tol = 1e-7
                if (ixx + iyy + tol < izz) or (ixx + izz + tol < iyy) or (iyy + izz + tol < ixx):
                    warnings.append(RobotDiagnostic(
                        code="INERTIA_TRIANGLE_INEQUALITY",
                        severity="warning",
                        target=f"link:{link.name}",
                        message=f"Link '{link.name}' inertia tensor violates physical triangle inequalities.",
                        suggestion="Use 'Auto-Calculate Inertial' to compute geometrically valid tensor."
                    ))

        # Collision geometry checks
        if link.visual and not link.collision:
            warnings.append(RobotDiagnostic(
                code="MISSING_COLLISION_GEOM",
                severity="warning",
                target=f"link:{link.name}",
                message=f"Link '{link.name}' has visual geometry but no collision geometry; it will not generate contact forces.",
                suggestion="Add collision geometry to enable realistic physical contacts with obstacles."
            ))

        if link.collision:
            c = link.collision
            if c.type not in ("box", "sphere", "cylinder"):
                errors.append(RobotDiagnostic(
                    code="INVALID_COLLISION_TYPE",
                    severity="error",
                    target=f"link:{link.name}",
                    message=f"Link '{link.name}' collision shape '{c.type}' is unsupported.",
                    suggestion="Choose 'box', 'sphere', or 'cylinder' for collision geometry."
                ))
            if not c.size or any(s <= 0 for s in c.size):
                errors.append(RobotDiagnostic(
                    code="NON_POSITIVE_COLLISION_SIZE",
                    severity="error",
                    target=f"link:{link.name}",
                    message=f"Link '{link.name}' collision geometry has non-positive dimensions {c.size}.",
                    suggestion="All collision geometry dimensions must be strictly positive numbers."
                ))
            elif any(s > 25.0 for s in c.size):
                errors.append(RobotDiagnostic(
                    code="EXCESSIVE_COLLISION_BOUNDS",
                    severity="error",
                    target=f"link:{link.name}",
                    message=f"Link '{link.name}' collision geometry size exceeds realistic bounds (>25m).",
                    suggestion="Verify and scale down collision dimensions."
                ))

        # Visual geometry checks
        if link.visual:
            v = link.visual
            if v.type not in ("box", "sphere", "cylinder"):
                errors.append(RobotDiagnostic(
                    code="INVALID_GEOM_TYPE",
                    severity="error",
                    target=f"link:{link.name}",
                    message=f"Link '{link.name}' visual has unsupported geometry type '{v.type}'.",
                    suggestion="Choose 'box', 'sphere', or 'cylinder'."
                ))
            if not v.size or any(s <= 0 for s in v.size):
                errors.append(RobotDiagnostic(
                    code="NON_POSITIVE_GEOM_SIZE",
                    severity="error",
                    target=f"link:{link.name}",
                    message=f"Link '{link.name}' visual has non-positive dimensions {v.size}.",
                    suggestion="All visual dimensions must be strictly positive numbers."
                ))

        if not link.visual and not link.collision:
            warnings.append(RobotDiagnostic(
                code="NO_GEOMETRY",
                severity="warning",
                target=f"link:{link.name}",
                message=f"Link '{link.name}' has neither visual nor collision geometry.",
                suggestion="Add a box, cylinder, or sphere shape so the link can be visualized and collide."
            ))

    # Adjacent link collision overlap heuristic check
    for jnt in robot.joints:
        p_link = links_by_name.get(jnt.parent_link)
        c_link = links_by_name.get(jnt.child_link)
        if p_link and c_link and p_link.collision and c_link.collision:
            dist_j = math.sqrt(jnt.origin_pos[0]**2 + jnt.origin_pos[1]**2 + jnt.origin_pos[2]**2)
            if dist_j < 0.001 and p_link.collision.origin_pos == c_link.collision.origin_pos:
                warnings.append(RobotDiagnostic(
                    code="ADJACENT_COLLISION_OVERLAP",
                    severity="warning",
                    target=f"joint:{jnt.name}",
                    message=f"Joint '{jnt.name}' links '{p_link.name}' and '{c_link.name}' share identical collision origins with zero joint offset, which may cause contact jitter.",
                    suggestion="Set an origin_pos offset on the joint or adjust link collision offsets."
                ))

    # 5. MuJoCo Physics Compilation Dry-Run
    if not errors:
        try:
            # Build minimal standalone MJCF test
            xml_lines = [
                f'<mujoco model="validation_{robot.name}">',
                '  <worldbody>'
            ]

            # Construct kinematic tree bodies
            links_by_name = {l.name: l for l in robot.links}

            def build_mjcf_link(link_name: str, parent_joint: Optional[JointModel] = None) -> str:
                l = links_by_name.get(link_name)
                if not l:
                    return ""
                b_lines = [f'<body name="{l.name}" pos="0 0 0">']
                if parent_joint is None and not robot.is_fixed_base:
                    b_lines.append('<freejoint/>')
                elif parent_joint is not None and parent_joint.type != "fixed":
                    j_type = "hinge" if parent_joint.type in ("revolute", "continuous") else "slide"
                    ax = f"{parent_joint.axis[0]} {parent_joint.axis[1]} {parent_joint.axis[2]}"
                    lim_str = ""
                    if parent_joint.type in ("revolute", "prismatic") and parent_joint.limit:
                        lim_str = f' limited="true" range="{parent_joint.limit.lower} {parent_joint.limit.upper}"'
                    b_lines.append(f'<joint name="{parent_joint.name}" type="{j_type}" axis="{ax}"{lim_str}/>')

                if l.inertial:
                    ipos = f"{l.inertial.origin_pos[0]} {l.inertial.origin_pos[1]} {l.inertial.origin_pos[2]}"
                    b_lines.append(f'<inertial pos="{ipos}" mass="{l.inertial.mass}" diaginertia="{l.inertial.ixx} {l.inertial.iyy} {l.inertial.izz}"/>')

                # Support both collision and visual geoms
                if l.collision:
                    c_str = " ".join(str(x) for x in l.collision.size)
                    b_lines.append(f'<geom name="col_{l.name}" type="{l.collision.type}" size="{c_str}"/>')
                if l.visual and (not l.collision or l.visual.type != l.collision.type or l.visual.size != l.collision.size):
                    v_str = " ".join(str(x) for x in l.visual.size)
                    b_lines.append(f'<geom name="vis_{l.name}" type="{l.visual.type}" size="{v_str}" contype="0" conaffinity="0"/>')
                elif not l.collision and not l.visual:
                    pass

                for c_joint in parent_to_joints.get(link_name, []):
                    c_xml = build_mjcf_link(c_joint.child_link, c_joint)
                    if c_xml:
                        b_lines.append(c_xml)

                b_lines.append('</body>')
                return "\n".join(b_lines)

            xml_lines.append(build_mjcf_link(robot.root_link, None))
            xml_lines.append('  </worldbody>')
            xml_lines.append('</mujoco>')
            full_test_xml = "\n".join(xml_lines)

            # Test compile in MuJoCo
            mujoco.MjModel.from_xml_string(full_test_xml)
        except Exception as me:
            errors.append(RobotDiagnostic(
                code="MUJOCO_COMPILATION_ERROR",
                severity="error",
                target="robot",
                message=f"MuJoCo physics compiler rejected model: {str(me)}",
                suggestion="Review joint axes, body origins, and inertial properties."
            ))

    valid = len(errors) == 0
    if valid:
        if warnings:
            summary = f"Robot model verified with {len(warnings)} non-blocking warnings."
        else:
            summary = "Robot model passed all kinematic, structural, and physics validation checks!"
    else:
        summary = f"Robot validation failed with {len(errors)} errors and {len(warnings)} warnings."

    return RobotValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        summary=summary
    )
