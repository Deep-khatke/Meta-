from __future__ import annotations
import math
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
import logging

from backend.models import (
    GenericRobotModel, LinkModel, LinkGeom, LinkInertial,
    JointModel, JointLimit, MJCFImportResult
)

logger = logging.getLogger("roboforge.mjcf")

def export_robot_mjcf(robot: GenericRobotModel) -> str:
    """Exports a GenericRobotModel to a clean, standard MJCF XML string compatible with MuJoCo."""
    links_by_name = {link.name: link for link in robot.links}
    joints_by_parent: Dict[str, List[JointModel]] = {}
    for j in robot.joints:
        joints_by_parent.setdefault(j.parent_link, []).append(j)

    xml_lines: List[str] = [
        f'<mujoco model="{robot.name}">',
        '  <compiler angle="radian" coordinate="local"/>',
        '  <option timestep="0.005" gravity="0 0 -9.81"/>',
        '  <worldbody>'
    ]

    def build_body(link_name: str, joint: Optional[JointModel] = None, indent: int = 4) -> None:
        link = links_by_name.get(link_name)
        if not link:
            return
        
        ind = " " * indent
        if joint is not None:
            px, py, pz = joint.origin_pos
            rx, ry, rz = [math.radians(a) for a in joint.origin_rpy]
            xml_lines.append(f'{ind}<body name="{link_name}" pos="{px:.4f} {py:.4f} {pz:.4f}">')
            
            # Joint tag
            jtype = joint.type
            if jtype == "continuous":
                jtype = "hinge"
                range_str = ""
            elif jtype == "revolute":
                jtype = "hinge"
                range_str = f' range="{joint.limit.lower:.4f} {joint.limit.upper:.4f}"' if joint.limit else ""
            elif jtype == "prismatic":
                jtype = "slide"
                range_str = f' range="{joint.limit.lower:.4f} {joint.limit.upper:.4f}"' if joint.limit else ""
            else:
                jtype = None

            if jtype:
                ax, ay, az = joint.axis
                damp = f' damping="{joint.damping:.3f}"' if joint.damping > 0 else ""
                arm = f' armature="{joint.armature:.3f}"' if joint.armature > 0 else ""
                xml_lines.append(f'{ind}  <joint name="{joint.name}" type="{jtype}" axis="{ax:.3f} {ay:.3f} {az:.3f}"{range_str}{damp}{arm}/>')
        else:
            # Root body
            bx, by, bz = robot.base_position
            xml_lines.append(f'{ind}<body name="{link_name}" pos="{bx:.4f} {by:.4f} {bz:.4f}">')
            if not robot.is_fixed_base:
                xml_lines.append(f'{ind}  <freejoint name="{robot.name}_freejoint"/>')

        # Inertial
        if link.inertial:
            ix, iy, iz = link.inertial.origin_pos
            xml_lines.append(f'{ind}  <inertial pos="{ix:.4f} {iy:.4f} {iz:.4f}" mass="{link.inertial.mass:.3f}" diaginertia="{link.inertial.ixx:.4f} {link.inertial.iyy:.4f} {link.inertial.izz:.4f}"/>')

        # Visual Geom
        if link.visual:
            v_size = " ".join(f"{s:.4f}" for s in link.visual.size)
            vx, vy, vz = link.visual.origin_pos
            xml_lines.append(f'{ind}  <geom name="{link_name}_visual" type="{link.visual.type}" size="{v_size}" pos="{vx:.4f} {vy:.4f} {vz:.4f}" contype="0" conaffinity="0" rgba="0.3 0.6 0.9 1.0"/>')

        # Collision Geom
        if link.collision:
            c_size = " ".join(f"{s:.4f}" for s in link.collision.size)
            cx, cy, cz = link.collision.origin_pos
            xml_lines.append(f'{ind}  <geom name="{link_name}_collision" type="{link.collision.type}" size="{c_size}" pos="{cx:.4f} {cy:.4f} {cz:.4f}" contype="1" conaffinity="1"/>')

        # Recurse children
        for child_joint in joints_by_parent.get(link_name, []):
            build_body(child_joint.child_link, child_joint, indent + 2)

        xml_lines.append(f'{ind}</body>')

    build_body(robot.root_link, None, 4)
    xml_lines.append('  </worldbody>')

    # Actuators section
    if robot.joints:
        xml_lines.append('  <actuator>')
        for j in robot.joints:
            if j.type == "fixed":
                continue
            mode = j.actuator_mode or "position"
            if mode == "position":
                ctrlrange = f' ctrlrange="{j.limit.lower:.4f} {j.limit.upper:.4f}"' if j.limit else ""
                xml_lines.append(f'    <position name="act_{j.name}" joint="{j.name}" kp="{j.kp:.1f}"{ctrlrange}/>')
            elif mode == "velocity":
                v_max = j.limit.velocity if j.limit else 10.0
                xml_lines.append(f'    <velocity name="act_{j.name}" joint="{j.name}" kv="{j.kv:.1f}" ctrlrange="{-v_max:.2f} {v_max:.2f}"/>')
            elif mode == "torque":
                eff = j.limit.effort if j.limit else 50.0
                xml_lines.append(f'    <motor name="act_{j.name}" joint="{j.name}" ctrlrange="{-eff:.2f} {eff:.2f}"/>')
        xml_lines.append('  </actuator>')

    xml_lines.append('</mujoco>')
    return "\n".join(xml_lines)


def parse_mjcf_string(mjcf_content: str) -> MJCFImportResult:
    """Parses an MJCF XML document into GenericRobotModel, recording warnings for unsupported features."""
    warnings: List[str] = []
    errors: List[str] = []

    try:
        root = ET.fromstring(mjcf_content)
    except ET.ParseError as pe:
        return MJCFImportResult(
            status="error",
            robot=None,
            errors=[f"Invalid XML syntax: {str(pe)}"]
        )

    if root.tag != "mujoco":
        return MJCFImportResult(
            status="error",
            robot=None,
            errors=[f"Root XML element must be <mujoco>, found <{root.tag}>"]
        )

    model_name = root.attrib.get("model", "imported_mjcf_robot")
    worldbody = root.find("worldbody")
    if worldbody is None:
        return MJCFImportResult(
            status="error",
            robot=None,
            errors=["Missing required <worldbody> tag in MJCF."]
        )

    # Check unsupported root elements
    for child in root:
        if child.tag in ("tendon", "equality", "sensor", "contact", "custom"):
            warnings.append(f"<{child.tag}> elements are not directly editable in the kinematic tree editor and were omitted.")

    links: List[LinkModel] = []
    joints: List[JointModel] = []

    top_bodies = worldbody.findall("body")
    if not top_bodies:
        return MJCFImportResult(
            status="error",
            robot=None,
            errors=["No <body> tags found under <worldbody>."]
        )

    root_body = top_bodies[0]
    root_name = root_body.attrib.get("name", "base_link")

    def parse_geom(geom_elem: ET.Element) -> Tuple[Optional[LinkGeom], bool]:
        """Parses a geom element. Returns (geom_model, is_visual_only)."""
        g_type = geom_elem.attrib.get("type", "sphere")
        if g_type not in ("box", "sphere", "cylinder"):
            warnings.append(f"Geom type '{g_type}' not natively supported in builder; mapped to box.")
            g_type = "box"

        size_str = geom_elem.attrib.get("size", "0.05")
        try:
            size_vals = [float(x) for x in size_str.split()]
        except ValueError:
            size_vals = [0.05]

        pos_str = geom_elem.attrib.get("pos", "0 0 0")
        try:
            pos_vals = [float(x) for x in pos_str.split()]
        except ValueError:
            pos_vals = [0.0, 0.0, 0.0]

        is_visual = geom_elem.attrib.get("contype", "1") == "0" and geom_elem.attrib.get("conaffinity", "1") == "0"
        
        geom = LinkGeom(
            type=g_type,  # type: ignore
            size=size_vals,
            origin_pos=pos_vals,
            color="#4a90e2"
        )
        return geom, is_visual

    def recurse_body(body_elem: ET.Element, parent_name: Optional[str] = None):
        b_name = body_elem.attrib.get("name", f"link_{len(links)}")
        
        # Geoms
        vis_geom = None
        col_geom = None
        for g in body_elem.findall("geom"):
            parsed_g, is_vis = parse_geom(g)
            if is_vis and not vis_geom:
                vis_geom = parsed_g
            elif not col_geom:
                col_geom = parsed_g

        if not vis_geom and col_geom:
            vis_geom = col_geom
        if not col_geom and vis_geom:
            col_geom = vis_geom

        # Inertial
        inertial = None
        ine_elem = body_elem.find("inertial")
        if ine_elem is not None:
            m = float(ine_elem.attrib.get("mass", "1.0"))
            ipos = [float(x) for x in ine_elem.attrib.get("pos", "0 0 0").split()]
            di = [float(x) for x in ine_elem.attrib.get("diaginertia", "0.01 0.01 0.01").split()]
            inertial = LinkInertial(
                mass=m,
                origin_pos=ipos if len(ipos) == 3 else [0, 0, 0],
                ixx=di[0] if len(di) > 0 else 0.01,
                iyy=di[1] if len(di) > 1 else 0.01,
                izz=di[2] if len(di) > 2 else 0.01
            )
        else:
            inertial = LinkInertial(mass=1.0, origin_pos=[0, 0, 0], ixx=0.01, iyy=0.01, izz=0.01)

        links.append(LinkModel(
            name=b_name,
            visual=vis_geom,
            collision=col_geom,
            inertial=inertial
        ))

        # Joint connecting to parent
        if parent_name is not None:
            j_elem = body_elem.find("joint")
            j_name = j_elem.attrib.get("name", f"joint_{b_name}") if j_elem is not None else f"joint_{b_name}"
            raw_type = j_elem.attrib.get("type", "hinge") if j_elem is not None else "hinge"
            
            # Map MJCF joint types
            if raw_type == "hinge":
                # Check if range is specified
                if j_elem is not None and "range" in j_elem.attrib:
                    j_type = "revolute"
                    rng = [float(x) for x in j_elem.attrib["range"].split()]
                    limit = JointLimit(lower=rng[0], upper=rng[1])
                else:
                    j_type = "continuous"
                    limit = None
            elif raw_type == "slide":
                j_type = "prismatic"
                if j_elem is not None and "range" in j_elem.attrib:
                    rng = [float(x) for x in j_elem.attrib["range"].split()]
                    limit = JointLimit(lower=rng[0], upper=rng[1])
                else:
                    limit = JointLimit(lower=-1.0, upper=1.0)
            else:
                j_type = "fixed"
                limit = None

            axis_str = j_elem.attrib.get("axis", "0 0 1") if j_elem is not None else "0 0 1"
            axis_vals = [float(x) for x in axis_str.split()] if len(axis_str.split()) == 3 else [0, 0, 1]

            pos_str = body_elem.attrib.get("pos", "0 0 0")
            pos_vals = [float(x) for x in pos_str.split()] if len(pos_str.split()) == 3 else [0, 0, 0]

            joints.append(JointModel(
                name=j_name,
                type=j_type,  # type: ignore
                parent_link=parent_name,
                child_link=b_name,
                origin_pos=pos_vals,
                origin_rpy=[0.0, 0.0, 0.0],
                axis=axis_vals,
                limit=limit
            ))

        for child in body_elem.findall("body"):
            recurse_body(child, b_name)

    recurse_body(root_body, None)

    robot = GenericRobotModel(
        name=model_name,
        root_link=root_name,
        links=links,
        joints=joints,
        base_position=[0.0, 0.0, 0.05],
        is_fixed_base=True
    )

    status = "warning" if warnings else "success"
    return MJCFImportResult(
        status=status,
        robot=robot,
        warnings=warnings,
        errors=[],
        total_links=len(links),
        total_joints=len(joints)
    )
