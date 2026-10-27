from __future__ import annotations
import math
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple

from backend.models import (
    GenericRobotModel, LinkModel, JointModel, JointLimit,
    LinkGeom, LinkInertial, URDFImportResult
)

def _parse_vec(text: Optional[str], default: List[float], expected_len: int = 3) -> List[float]:
    if not text:
        return default
    try:
        vals = [float(x) for x in text.strip().split()]
        if len(vals) == expected_len:
            return vals
    except Exception:
        pass
    return default

def _parse_geom(geom_elem: Optional[ET.Element], origin_elem: Optional[ET.Element], color_str: str = "#4a90e2") -> Tuple[Optional[LinkGeom], List[str]]:
    warnings = []
    if geom_elem is None:
        return None, warnings
    
    pos = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin_elem is not None:
        pos = _parse_vec(origin_elem.get("xyz"), [0.0, 0.0, 0.0])
        rpy_rad = _parse_vec(origin_elem.get("rpy"), [0.0, 0.0, 0.0])
        # Convert radians to degrees for consistency
        rpy = [math.degrees(a) for a in rpy_rad]

    # Check geometry types
    box = geom_elem.find("box")
    sphere = geom_elem.find("sphere")
    cylinder = geom_elem.find("cylinder")
    mesh = geom_elem.find("mesh")

    if box is not None:
        size_full = _parse_vec(box.get("size"), [0.2, 0.2, 0.2])
        # In our models, box is half_size [hx, hy, hz]
        half_size = [s * 0.5 for s in size_full]
        return LinkGeom(type="box", size=half_size, origin_pos=pos, origin_rpy=rpy, color=color_str), warnings
    elif sphere is not None:
        r = float(sphere.get("radius", "0.1"))
        return LinkGeom(type="sphere", size=[r], origin_pos=pos, origin_rpy=rpy, color=color_str), warnings
    elif cylinder is not None:
        r = float(cylinder.get("radius", "0.05"))
        l = float(cylinder.get("length", "0.2"))
        # In our models, cylinder is [radius, half_length]
        return LinkGeom(type="cylinder", size=[r, l * 0.5], origin_pos=pos, origin_rpy=rpy, color=color_str), warnings
    elif mesh is not None:
        filename = mesh.get("filename", "unknown.dae")
        warnings.append(f"Mesh geometry '{filename}' is not natively rendered; converted to bounding box placeholder.")
        return LinkGeom(type="box", size=[0.1, 0.1, 0.1], origin_pos=pos, origin_rpy=rpy, color=color_str), warnings
    
    return None, warnings

def parse_urdf_string(urdf_content: str) -> URDFImportResult:
    """Parses a URDF XML string into a GenericRobotModel, returning validation warnings."""
    warnings: List[str] = []
    errors: List[str] = []

    try:
        root = ET.fromstring(urdf_content)
    except Exception as e:
        return URDFImportResult(
            status="error",
            robot=None,
            errors=[f"XML Parsing Failure: {str(e)}"],
            warnings=[],
            total_links=0,
            total_joints=0
        )

    if root.tag != "robot":
        return URDFImportResult(
            status="error",
            robot=None,
            errors=[f"Root tag must be <robot>, found <{root.tag}>"],
            warnings=[],
            total_links=0,
            total_joints=0
        )

    robot_name = root.get("name", "ImportedURDFRobot")

    # Unsupported top-level elements check
    for trans in root.findall("transmission"):
        tname = trans.get("name", "transmission")
        warnings.append(f"<transmission '{tname}'> construct ignored (MuJoCo manages direct joint actuation).")
    for gz in root.findall("gazebo"):
        warnings.append("<gazebo> extension tags ignored.")

    # 1. Parse Links
    links: List[LinkModel] = []
    link_names = set()

    for link_el in root.findall("link"):
        lname = link_el.get("name")
        if not lname:
            warnings.append("Encountered <link> without name attribute; skipped.")
            continue
        link_names.add(lname)

        # Parse Inertial
        inertial_model: Optional[LinkInertial] = None
        inert_el = link_el.find("inertial")
        if inert_el is not None:
            m_el = inert_el.find("mass")
            mass_val = float(m_el.get("value", "1.0")) if m_el is not None else 1.0
            
            ipos = [0.0, 0.0, 0.0]
            orig = inert_el.find("origin")
            if orig is not None:
                ipos = _parse_vec(orig.get("xyz"), [0.0, 0.0, 0.0])

            iner = inert_el.find("inertia")
            ixx = float(iner.get("ixx", "0.01")) if iner is not None else 0.01
            iyy = float(iner.get("iyy", "0.01")) if iner is not None else 0.01
            izz = float(iner.get("izz", "0.01")) if iner is not None else 0.01

            inertial_model = LinkInertial(mass=mass_val, origin_pos=ipos, ixx=ixx, iyy=iyy, izz=izz)

        # Visual Material Color
        mat_color = "#4a90e2"
        vis_el = link_el.find("visual")
        if vis_el is not None:
            mat_el = vis_el.find("material")
            if mat_el is not None:
                col_el = mat_el.find("color")
                if col_el is not None:
                    rgba = _parse_vec(col_el.get("rgba"), [0.3, 0.5, 0.8, 1.0], expected_len=4)
                    r = int(rgba[0] * 255)
                    g = int(rgba[1] * 255)
                    b = int(rgba[2] * 255)
                    mat_color = f"#{r:02x}{g:02x}{b:02x}"

        # Parse Visual & Collision
        visual_geom = None
        if vis_el is not None:
            v_orig = vis_el.find("origin")
            v_geom = vis_el.find("geometry")
            visual_geom, v_warns = _parse_geom(v_geom, v_orig, color_str=mat_color)
            warnings.extend(v_warns)

        col_el = link_el.find("collision")
        collision_geom = None
        if col_el is not None:
            c_orig = col_el.find("origin")
            c_geom = col_el.find("geometry")
            collision_geom, c_warns = _parse_geom(c_geom, c_orig, color_str=mat_color)
            warnings.extend(c_warns)

        links.append(LinkModel(
            name=lname,
            visual=visual_geom,
            collision=collision_geom,
            inertial=inertial_model
        ))

    # 2. Parse Joints
    joints: List[JointModel] = []
    child_links = set()

    for jnt_el in root.findall("joint"):
        jname = jnt_el.get("name")
        jtype = jnt_el.get("type", "revolute")

        p_el = jnt_el.find("parent")
        c_el = jnt_el.find("child")

        if not jname or p_el is None or c_el is None:
            warnings.append(f"Joint '{jname or 'unnamed'}' missing name, parent, or child link; skipped.")
            continue

        parent_name = p_el.get("link")
        child_name = c_el.get("link")

        if not parent_name or not child_name:
            continue

        if parent_name not in link_names:
            warnings.append(f"Joint '{jname}' parent link '{parent_name}' not defined in URDF.")
        if child_name not in link_names:
            warnings.append(f"Joint '{jname}' child link '{child_name}' not defined in URDF.")

        child_links.add(child_name)

        # Supported types: revolute, continuous, prismatic, fixed
        mapped_type = jtype
        if jtype not in ("revolute", "continuous", "prismatic", "fixed"):
            warnings.append(f"Joint '{jname}' type '{jtype}' unsupported; treated as 'fixed'.")
            mapped_type = "fixed"

        # Origin
        j_pos = [0.0, 0.0, 0.0]
        j_rpy = [0.0, 0.0, 0.0]
        orig = jnt_el.find("origin")
        if orig is not None:
            j_pos = _parse_vec(orig.get("xyz"), [0.0, 0.0, 0.0])
            rpy_rad = _parse_vec(orig.get("rpy"), [0.0, 0.0, 0.0])
            j_rpy = [math.degrees(a) for a in rpy_rad]

        # Axis
        axis = [0.0, 0.0, 1.0]
        axis_el = jnt_el.find("axis")
        if axis_el is not None:
            axis = _parse_vec(axis_el.get("xyz"), [0.0, 0.0, 1.0])

        # Limits
        limit_model: Optional[JointLimit] = None
        lim_el = jnt_el.find("limit")
        if lim_el is not None and mapped_type in ("revolute", "prismatic"):
            lower = float(lim_el.get("lower", "-3.14159"))
            upper = float(lim_el.get("upper", "3.14159"))
            effort = float(lim_el.get("effort", "100.0"))
            vel = float(lim_el.get("velocity", "10.0"))
            limit_model = JointLimit(lower=lower, upper=upper, effort=effort, velocity=vel)

        # Dynamics
        damping = 0.1
        dyn_el = jnt_el.find("dynamics")
        if dyn_el is not None:
            damping = float(dyn_el.get("damping", "0.1"))

        joints.append(JointModel(
            name=jname,
            type=mapped_type,  # type: ignore
            parent_link=parent_name,
            child_link=child_name,
            origin_pos=j_pos,
            origin_rpy=j_rpy,
            axis=axis,
            limit=limit_model,
            damping=damping,
            armature=0.01,
            actuator_mode="position" if mapped_type != "fixed" else None,
            kp=100.0,
            kv=10.0
        ))

    # Find root link (link that is never a child link)
    root_candidates = [lname for lname in link_names if lname not in child_links]
    root_link = root_candidates[0] if root_candidates else (links[0].name if links else "base_link")

    robot_model = GenericRobotModel(
        name=robot_name,
        root_link=root_link,
        links=links,
        joints=joints,
        base_position=[2.5, 2.5, 0.05],
        base_rpy=[0.0, 0.0, 0.0],
        is_fixed_base=True
    )

    status = "error" if errors else ("warning" if warnings else "success")
    return URDFImportResult(
        status=status,
        robot=robot_model,
        warnings=warnings,
        errors=errors,
        total_links=len(links),
        total_joints=len(joints)
    )

def export_robot_urdf(robot: GenericRobotModel) -> str:
    """Exports a GenericRobotModel as a standard, valid URDF XML string."""
    lines = [f'<robot name="{robot.name}">']

    # Links
    for link in robot.links:
        lines.append(f'  <link name="{link.name}">')
        if link.inertial:
            ip = f"{link.inertial.origin_pos[0]} {link.inertial.origin_pos[1]} {link.inertial.origin_pos[2]}"
            lines.append('    <inertial>')
            lines.append(f'      <origin xyz="{ip}" rpy="0 0 0"/>')
            lines.append(f'      <mass value="{link.inertial.mass}"/>')
            lines.append(f'      <inertia ixx="{link.inertial.ixx}" ixy="0" ixz="0" iyy="{link.inertial.iyy}" iyz="0" izz="{link.inertial.izz}"/>')
            lines.append('    </inertial>')

        for tag, geom in [("visual", link.visual), ("collision", link.collision)]:
            if not geom:
                continue
            pos_str = f"{geom.origin_pos[0]} {geom.origin_pos[1]} {geom.origin_pos[2]}"
            rpy_rad = f"{math.radians(geom.origin_rpy[0]):.4f} {math.radians(geom.origin_rpy[1]):.4f} {math.radians(geom.origin_rpy[2]):.4f}"
            lines.append(f'    <{tag}>')
            lines.append(f'      <origin xyz="{pos_str}" rpy="{rpy_rad}"/>')
            lines.append('      <geometry>')
            if geom.type == "box":
                s = f"{geom.size[0] * 2.0} {geom.size[1] * 2.0} {geom.size[2] * 2.0}"
                lines.append(f'        <box size="{s}"/>')
            elif geom.type == "sphere":
                lines.append(f'        <sphere radius="{geom.size[0]}"/>')
            elif geom.type == "cylinder":
                r = geom.size[0]
                l = geom.size[1] * 2.0 if len(geom.size) > 1 else 0.2
                lines.append(f'        <cylinder radius="{r}" length="{l}"/>')
            lines.append('      </geometry>')
            if tag == "visual" and geom.color:
                lines.append('      <material name="mat_' + link.name + '">')
                # Convert hex to rgba
                hc = geom.color.lstrip('#')
                if len(hc) == 6:
                    r = int(hc[0:2], 16) / 255.0
                    g = int(hc[2:4], 16) / 255.0
                    b = int(hc[4:6], 16) / 255.0
                    lines.append(f'        <color rgba="{r:.3f} {g:.3f} {b:.3f} 1.0"/>')
                lines.append('      </material>')
            lines.append(f'    </{tag}>')
        lines.append('  </link>')

    # Joints
    for joint in robot.joints:
        jtype = joint.type if joint.type in ("revolute", "continuous", "prismatic", "fixed") else "fixed"
        lines.append(f'  <joint name="{joint.name}" type="{jtype}">')
        lines.append(f'    <parent link="{joint.parent_link}"/>')
        lines.append(f'    <child link="{joint.child_link}"/>')
        pos_str = f"{joint.origin_pos[0]} {joint.origin_pos[1]} {joint.origin_pos[2]}"
        rpy_rad = f"{math.radians(joint.origin_rpy[0]):.4f} {math.radians(joint.origin_rpy[1]):.4f} {math.radians(joint.origin_rpy[2]):.4f}"
        lines.append(f'    <origin xyz="{pos_str}" rpy="{rpy_rad}"/>')
        lines.append(f'    <axis xyz="{joint.axis[0]} {joint.axis[1]} {joint.axis[2]}"/>')
        if joint.limit and jtype in ("revolute", "prismatic"):
            lines.append(f'    <limit lower="{joint.limit.lower}" upper="{joint.limit.upper}" effort="{joint.limit.effort}" velocity="{joint.limit.velocity}"/>')
        lines.append(f'    <dynamics damping="{joint.damping}"/>')
        lines.append('  </joint>')

    lines.append('</robot>')
    return "\n".join(lines)

