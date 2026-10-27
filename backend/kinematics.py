from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from backend.models import (
    GenericRobotModel, LinkModel, JointModel, JointLimit,
    FKResult, LinkPose, JointFrame, IKRequest, IKResult
)

# --- 3D Geometric & Transform Utilities ---

def rpy_deg_to_rot_matrix(rpy_deg: List[float]) -> np.ndarray:
    """Computes 3x3 rotation matrix from [roll, pitch, yaw] in degrees (extrinsic XYZ / intrinsic ZYX)."""
    r = math.radians(rpy_deg[0])
    p = math.radians(rpy_deg[1])
    y = math.radians(rpy_deg[2])

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr]
    ], dtype=np.float64)

    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp]
    ], dtype=np.float64)

    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Standard robotics rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

def rot_matrix_to_rpy_deg(R: np.ndarray) -> List[float]:
    """Converts a 3x3 rotation matrix to [roll, pitch, yaw] in degrees."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        r = math.atan2(R[2, 1], R[2, 2])
        p = math.atan2(-R[2, 0], sy)
        y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = math.atan2(-R[1, 2], R[1, 1])
        p = math.atan2(-R[2, 0], sy)
        y = 0.0

    return [float(math.degrees(r)), float(math.degrees(p)), float(math.degrees(y))]

def rot_matrix_to_quat(R: np.ndarray) -> List[float]:
    """Converts a 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm > 1e-8:
        return [float(w/norm), float(x/norm), float(y/norm), float(z/norm)]
    return [1.0, 0.0, 0.0, 0.0]

def rodrigues_rot_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Computes 3x3 rotation matrix for rotation of angle theta around arbitrary unit axis."""
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    u = axis / norm
    ux, uy, uz = u

    # Skew symmetric matrix
    K = np.array([
        [0.0, -uz, uy],
        [uz, 0.0, -ux],
        [-uy, ux, 0.0]
    ], dtype=np.float64)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    return np.eye(3, dtype=np.float64) + sin_t * K + (1.0 - cos_t) * (K @ K)

def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Creates a 4x4 homogeneous transformation matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


# --- Forward Kinematics Engine ---

def compute_forward_kinematics(
    robot: GenericRobotModel,
    joint_positions: Optional[Dict[str, float]] = None,
    custom_end_effector: Optional[str] = None
) -> FKResult:
    """
    Computes exact forward kinematics for a generic articulated robot model.
    Traverses the kinematic tree from root_link down, calculating:
    - Link poses in base / world frame
    - Joint frame positions and world axes
    - End-effector pose (position, orientation quaternion, and RPY)
    - Total reach envelope of the articulated chain
    """
    if not robot.links:
        return FKResult(
            success=False,
            end_effector_link="",
            end_effector_position=[0.0, 0.0, 0.0],
            end_effector_quaternion=[1.0, 0.0, 0.0, 0.0],
            end_effector_rpy_deg=[0.0, 0.0, 0.0],
            link_poses={},
            joint_frames={},
            total_reach=0.0
        )

    j_pos_map = joint_positions or {}
    links_by_name = {l.name: l for l in robot.links}

    # Index joints by parent link
    joints_by_parent: Dict[str, List[JointModel]] = {}
    joints_by_child: Dict[str, JointModel] = {}
    for j in robot.joints:
        joints_by_parent.setdefault(j.parent_link, []).append(j)
        joints_by_child[j.child_link] = j

    # Root link base transform
    base_pos = np.array(robot.base_position, dtype=np.float64)
    base_R = rpy_deg_to_rot_matrix(robot.base_rpy)
    T_root = make_transform(base_R, base_pos)

    link_transforms: Dict[str, np.ndarray] = {robot.root_link: T_root}
    joint_frames: Dict[str, JointFrame] = {}

    # Recursive traversal of the kinematic tree
    total_reach = 0.0
    leaf_links: List[str] = []

    def traverse(parent_name: str, T_parent: np.ndarray):
        children_joints = joints_by_parent.get(parent_name, [])
        if not children_joints:
            leaf_links.append(parent_name)
            return

        for joint in children_joints:
            child_name = joint.child_link
            # 1. Joint origin transform relative to parent
            j_origin_pos = np.array(joint.origin_pos, dtype=np.float64)
            j_origin_R = rpy_deg_to_rot_matrix(joint.origin_rpy)
            T_joint_origin = make_transform(j_origin_R, j_origin_pos)

            # World pose of joint origin (before joint motion)
            T_joint_world = T_parent @ T_joint_origin

            # 2. Joint motion transform
            j_val = j_pos_map.get(joint.name, 0.0)
            axis = np.array(joint.axis, dtype=np.float64)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
            else:
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            if joint.type in ("revolute", "continuous"):
                R_motion = rodrigues_rot_matrix(axis, j_val)
                p_motion = np.zeros(3, dtype=np.float64)
            elif joint.type == "prismatic":
                R_motion = np.eye(3, dtype=np.float64)
                p_motion = axis * j_val
            else:  # fixed
                R_motion = np.eye(3, dtype=np.float64)
                p_motion = np.zeros(3, dtype=np.float64)

            T_motion = make_transform(R_motion, p_motion)

            # Transform of child link in world frame
            T_child = T_joint_world @ T_motion
            link_transforms[child_name] = T_child

            # Record world axis of joint
            world_axis = (T_joint_world[:3, :3] @ axis).tolist()
            joint_frames[joint.name] = JointFrame(
                joint_name=joint.name,
                origin_position=[float(x) for x in T_joint_world[:3, 3]],
                axis_vector=[float(x) for x in world_axis],
                joint_type=joint.type,
                current_position=float(j_val),
                limits=joint.limit
            )

            # Accumulate nominal translation length for reach estimation
            nonlocal total_reach
            total_reach += float(np.linalg.norm(j_origin_pos))

            traverse(child_name, T_child)

    traverse(robot.root_link, T_root)

    # Determine end-effector link
    ee_link = custom_end_effector
    if not ee_link or ee_link not in link_transforms:
        # Prefer known naming conventions, or the deepest leaf link
        preferred_ee = [l for l in leaf_links if any(tag in l.lower() for tag in ["ee", "tool", "gripper", "end", "wrist", "tip"])]
        if preferred_ee:
            ee_link = preferred_ee[0]
        elif leaf_links:
            ee_link = leaf_links[-1]
        else:
            ee_link = robot.links[-1].name if robot.links else robot.root_link

    # Compute end-effector tip pose
    T_ee = link_transforms.get(ee_link, T_root)
    ee_pos = T_ee[:3, 3]
    ee_R = T_ee[:3, :3]

    # If the end-effector link has visual/collision geometry, offset tip to outer boundary
    ee_link_obj = links_by_name.get(ee_link)
    ee_extent = 0.0
    if ee_link_obj:
        geom = ee_link_obj.visual or ee_link_obj.collision
        if geom:
            g_pos = np.array(geom.origin_pos, dtype=np.float64)
            # Tip offset along geometry primary axis
            tip_offset = g_pos.copy()
            if geom.type == "cylinder" and len(geom.size) >= 2:
                tip_offset[2] += geom.size[1] # half height
            elif geom.type == "box" and len(geom.size) >= 3:
                # Add furthest extent
                tip_offset[0] += geom.size[0]
            tip_world = ee_pos + ee_R @ tip_offset
            # Only use if reasonable
            if not np.allclose(tip_offset, 0.0):
                ee_pos = tip_world
                ee_extent = float(np.linalg.norm(tip_offset))

    total_reach += ee_extent

    ee_quat = rot_matrix_to_quat(ee_R)
    ee_rpy = rot_matrix_to_rpy_deg(ee_R)

    # Format link poses dictionary
    link_poses: Dict[str, LinkPose] = {}
    for lname, T_mat in link_transforms.items():
        lp_pos = [float(x) for x in T_mat[:3, 3]]
        lp_quat = rot_matrix_to_quat(T_mat[:3, :3])
        lp_rpy = rot_matrix_to_rpy_deg(T_mat[:3, :3])
        parent = joints_by_child[lname].parent_link if lname in joints_by_child else None
        link_poses[lname] = LinkPose(
            link_name=lname,
            position=lp_pos,
            quaternion=lp_quat,
            rpy_deg=lp_rpy,
            parent_link=parent
        )

    return FKResult(
        success=True,
        end_effector_link=ee_link,
        end_effector_position=[float(x) for x in ee_pos],
        end_effector_quaternion=[float(x) for x in ee_quat],
        end_effector_rpy_deg=[float(x) for x in ee_rpy],
        link_poses=link_poses,
        joint_frames=joint_frames,
        total_reach=float(max(total_reach, 0.2))
    )


# --- Inverse Kinematics Engine (Damped Least Squares with Limit Clamping) ---

def extract_kinematic_chain(
    robot: GenericRobotModel,
    end_effector_link: str
) -> List[JointModel]:
    """Finds the ordered list of joints from root_link to end_effector_link."""
    joints_by_child = {j.child_link: j for j in robot.joints}
    chain: List[JointModel] = []
    curr = end_effector_link

    visited = set()
    while curr in joints_by_child and curr not in visited:
        visited.add(curr)
        joint = joints_by_child[curr]
        chain.append(joint)
        curr = joint.parent_link
        if curr == robot.root_link:
            break

    chain.reverse()
    # Filter to only actuated joints (revolute, continuous, prismatic)
    actuated_chain = [j for j in chain if j.type != "fixed"]
    return actuated_chain


def solve_inverse_kinematics(req: IKRequest) -> IKResult:
    """
    Practical, robust Inverse Kinematics solver for generic articulated chains.
    Uses Damped Least Squares (Levenberg-Marquardt Jacobian pseudo-inverse)
    with joint limit saturation detection and reach verification.

    CRITICAL RULE: Never silently claims success if error exceeds tolerance.
    Reports clear status:
    - SUCCESS: Target reached within tolerance.
    - TARGET_OUT_OF_REACH: Target distance strictly exceeds physical reach envelope.
    - JOINT_LIMITS_BLOCKED: Joint limits prevent the robot from reaching the target.
    - CONVERGENCE_FAILED: Solver timed out or reached local minimum without reaching target.
    """
    robot = req.robot
    if not robot.links or not robot.joints:
        return IKResult(
            success=False,
            status="ERROR",
            iterations=0,
            error_distance=999.0,
            joint_solution={},
            achieved_position=[0.0, 0.0, 0.0],
            achieved_rpy_deg=[0.0, 0.0, 0.0],
            message="Robot model has no links or joints defined."
        )

    # 1. Determine EE link and extract active kinematic chain
    fk_nominal = compute_forward_kinematics(robot, req.initial_joint_positions, req.end_effector_link)
    ee_link = req.end_effector_link or fk_nominal.end_effector_link
    active_chain = extract_kinematic_chain(robot, ee_link)

    if not active_chain:
        return IKResult(
            success=False,
            status="ERROR",
            iterations=0,
            error_distance=float(np.linalg.norm(np.array(fk_nominal.end_effector_position) - np.array(req.target_position))),
            joint_solution={},
            achieved_position=fk_nominal.end_effector_position,
            achieved_rpy_deg=fk_nominal.end_effector_rpy_deg,
            message=f"No movable joints found in kinematic chain to link '{ee_link}'."
        )

    target_pos = np.array(req.target_position, dtype=np.float64)
    base_pos = np.array(robot.base_position, dtype=np.float64)

    # 2. Check theoretical physical workspace reach
    dist_from_base = float(np.linalg.norm(target_pos - base_pos))
    max_reach = fk_nominal.total_reach + 0.08  # Include geometry tip margin

    if dist_from_base > max_reach:
        exceed_dist = dist_from_base - max_reach
        return IKResult(
            success=False,
            status="TARGET_OUT_OF_REACH",
            iterations=0,
            error_distance=float(exceed_dist),
            joint_solution=req.initial_joint_positions or {},
            achieved_position=fk_nominal.end_effector_position,
            achieved_rpy_deg=fk_nominal.end_effector_rpy_deg,
            message=(
                f"Target position at distance {dist_from_base:.2f}m is outside maximum theoretical "
                f"workspace reach of {max_reach:.2f}m (exceeds by {exceed_dist:.2f}m)."
            )
        )

    # 3. Setup solver state
    current_q: Dict[str, float] = {}
    if req.initial_joint_positions:
        current_q.update(req.initial_joint_positions)
    for j in active_chain:
        if j.name not in current_q:
            # Default to mid-range of limits or 0
            if j.limit:
                current_q[j.name] = (j.limit.lower + j.limit.upper) * 0.5
            else:
                current_q[j.name] = 0.0

    num_joints = len(active_chain)
    q_vec = np.array([current_q[j.name] for j in active_chain], dtype=np.float64)

    tolerance = max(req.tolerance, 0.001)
    max_iter = max(req.max_iterations, 30)
    damping = 0.05

    best_q_vec = q_vec.copy()
    best_error = float("inf")
    best_ee_pos = np.array(fk_nominal.end_effector_position)
    best_ee_rpy = fk_nominal.end_effector_rpy_deg

    iteration = 0
    limits_saturated = False

    for it in range(max_iter):
        iteration = it + 1
        # Compute current FK
        joint_dict = {active_chain[i].name: float(q_vec[i]) for i in range(num_joints)}
        fk = compute_forward_kinematics(robot, joint_dict, ee_link)
        cur_pos = np.array(fk.end_effector_position, dtype=np.float64)

        err_vec = target_pos - cur_pos
        err_dist = float(np.linalg.norm(err_vec))

        if err_dist < best_error:
            best_error = err_dist
            best_q_vec = q_vec.copy()
            best_ee_pos = cur_pos
            best_ee_rpy = fk.end_effector_rpy_deg

        if err_dist <= tolerance:
            # Converged successfully!
            final_solution = {active_chain[i].name: float(q_vec[i]) for i in range(num_joints)}
            return IKResult(
                success=True,
                status="SUCCESS",
                iterations=iteration,
                error_distance=float(err_dist),
                joint_solution=final_solution,
                achieved_position=[float(x) for x in cur_pos],
                achieved_rpy_deg=fk.end_effector_rpy_deg,
                message=f"IK solved successfully in {iteration} iterations (residual error: {err_dist*1000.0:.2f} mm)."
            )

        # Compute Geometric Jacobian J (3 x M)
        J = np.zeros((3, num_joints), dtype=np.float64)
        for i, jnt in enumerate(active_chain):
            j_frame = fk.joint_frames.get(jnt.name)
            if not j_frame:
                continue
            z_axis = np.array(j_frame.axis_vector, dtype=np.float64)
            p_joint = np.array(j_frame.origin_position, dtype=np.float64)

            if jnt.type in ("revolute", "continuous"):
                # Column is z_i x (p_ee - p_joint)
                J[:, i] = np.cross(z_axis, cur_pos - p_joint)
            elif jnt.type == "prismatic":
                J[:, i] = z_axis

        # Damped Least Squares: delta_q = J^T * (J * J^T + lambda^2 * I)^(-1) * err_vec
        JJt = J @ J.T
        damped_inv = np.linalg.inv(JJt + (damping ** 2) * np.eye(3))
        delta_q = J.T @ (damped_inv @ err_vec)

        # Scale step size to prevent overshoot
        max_step = 0.25 # max 0.25 rad (~14 deg) per iteration
        step_norm = float(np.linalg.norm(delta_q))
        if step_norm > max_step:
            delta_q = delta_q * (max_step / step_norm)

        # Candidate update
        q_candidate = q_vec + delta_q

        # Enforce joint limits
        clamped_count = 0
        for i, jnt in enumerate(active_chain):
            if jnt.type in ("revolute", "prismatic") and jnt.limit:
                lower = jnt.limit.lower
                upper = jnt.limit.upper
                if q_candidate[i] <= lower:
                    q_candidate[i] = lower
                    clamped_count += 1
                elif q_candidate[i] >= upper:
                    q_candidate[i] = upper
                    clamped_count += 1
            elif jnt.type == "continuous":
                # Normalize continuous joints to [-pi, pi]
                q_candidate[i] = (q_candidate[i] + math.pi) % (2.0 * math.pi) - math.pi

        if clamped_count >= max(1, num_joints - 1):
            limits_saturated = True

        q_vec = q_candidate

    # If loop ended without reaching tolerance, report honest result
    final_solution = {active_chain[i].name: float(best_q_vec[i]) for i in range(num_joints)}
    status_msg = "JOINT_LIMITS_BLOCKED" if limits_saturated else "CONVERGENCE_FAILED"
    diag = (
        f"Joint limits prevented reaching target. Closest distance: {best_error:.3f}m"
        if limits_saturated else
        f"Solver reached limit of {max_iter} iterations. Closest distance: {best_error:.3f}m"
    )

    return IKResult(
        success=False,
        status=status_msg,  # type: ignore
        iterations=iteration,
        error_distance=float(best_error),
        joint_solution=final_solution,
        achieved_position=[float(x) for x in best_ee_pos],
        achieved_rpy_deg=best_ee_rpy,
        message=f"{diag} (residual error: {best_error*1000.0:.1f} mm)."
    )
