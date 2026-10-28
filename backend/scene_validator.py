from __future__ import annotations
import math
from typing import List, Dict, Optional, Tuple, Literal
from pydantic import BaseModel, Field

from backend.models import Project, SceneObject, RobotConfig, GenericRobotModel

class SceneIssue(BaseModel):
    id: str
    severity: Literal["error", "warning", "info"]
    category: Literal["overlap", "floor_penetration", "out_of_bounds", "robot_clearance", "duplicate_id", "invalid_transform"]
    target_ids: List[str]
    message: str
    suggestion: Optional[str] = None

class SceneValidationReport(BaseModel):
    is_valid: bool
    errors: List[SceneIssue] = Field(default_factory=list)
    warnings: List[SceneIssue] = Field(default_factory=list)
    infos: List[SceneIssue] = Field(default_factory=list)
    total_objects: int = 0
    overlapping_pairs_count: int = 0


def get_object_aabb(obj: SceneObject) -> Tuple[List[float], List[float]]:
    """Calculates axis-aligned bounding box (min_xyz, max_xyz) for primitive objects in meters."""
    px, py, pz = obj.position
    if obj.type == "box":
        hx, hy, hz = obj.size[0], obj.size[1], obj.size[2]
        return [px - hx, py - hy, pz - hz], [px + hx, py + hy, pz + hz]
    elif obj.type == "sphere":
        r = obj.size[0]
        return [px - r, py - r, pz - r], [px + r, py + r, pz + r]
    elif obj.type == "cylinder":
        r, hh = obj.size[0], obj.size[1]
        return [px - r, py - r, pz - hh], [px + r, py + r, pz + hh]
    else:
        # Default fallback
        d = 0.2
        return [px - d, py - d, pz - d], [px + d, py + d, pz + d]


def check_aabb_intersection(min1: List[float], max1: List[float], min2: List[float], max2: List[float], margin: float = 0.005) -> bool:
    """Returns True if two AABBs overlap with a small contact tolerance margin."""
    return (
        (min1[0] + margin) < (max2[0] - margin) and (max1[0] - margin) > (min2[0] + margin) and
        (min1[1] + margin) < (max2[1] - margin) and (max1[1] - margin) > (min2[1] + margin) and
        (min1[2] + margin) < (max2[2] - margin) and (max1[2] - margin) > (min2[2] + margin)
    )


def validate_scene(project: Project) -> SceneValidationReport:
    """Comprehensive scene validation: collision overlaps, floor penetration, boundaries, and clearances."""
    errors: List[SceneIssue] = []
    warnings: List[SceneIssue] = []
    infos: List[SceneIssue] = []

    objects = project.objects
    bounds = project.settings.world_boundaries if project.settings.enable_boundaries else [-100, 100, -100, 100, 0, 50]
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    seen_ids: Dict[str, str] = {}
    overlapping_pairs = 0

    # 1. Check duplicate IDs and invalid transforms
    for obj in objects:
        if obj.id in seen_ids:
            errors.append(SceneIssue(
                id=f"dup_{obj.id}",
                severity="error",
                category="duplicate_id",
                target_ids=[obj.id, seen_ids[obj.id]],
                message=f"Duplicate object ID '{obj.id}' detected on '{obj.name}'",
                suggestion="Re-generate a unique UUID for this object."
            ))
        else:
            seen_ids[obj.id] = obj.name

        # Transform finite checks
        if any(math.isnan(v) or math.isinf(v) for v in obj.position):
            errors.append(SceneIssue(
                id=f"nan_pos_{obj.id}",
                severity="error",
                category="invalid_transform",
                target_ids=[obj.id],
                message=f"Object '{obj.name}' has NaN or infinite position coordinates.",
                suggestion="Reset object position to valid numbers."
            ))

    # 2. Check floor penetration and world bounds for each object
    for obj in objects:
        min_pt, max_pt = get_object_aabb(obj)

        # Floor penetration (zmin is usually 0.0)
        if min_pt[2] < (zmin - 0.01):
            penetration_depth = round(zmin - min_pt[2], 3)
            warnings.append(SceneIssue(
                id=f"floor_pen_{obj.id}",
                severity="warning",
                category="floor_penetration",
                target_ids=[obj.id],
                message=f"Object '{obj.name}' penetrates the ground plane by {penetration_depth}m (min z={min_pt[2]:.3f})",
                suggestion=f"Raise z position by at least {penetration_depth}m to rest atop the floor."
            ))

        # World boundary check
        if project.settings.enable_boundaries:
            if min_pt[0] < xmin or max_pt[0] > xmax or min_pt[1] < ymin or max_pt[1] > ymax or max_pt[2] > zmax:
                warnings.append(SceneIssue(
                    id=f"oob_{obj.id}",
                    severity="warning",
                    category="out_of_bounds",
                    target_ids=[obj.id],
                    message=f"Object '{obj.name}' extends outside the world boundaries [{xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax}].",
                    suggestion="Move the object inside the boundary walls or expand the boundaries."
                ))

    # 3. Check pairwise AABB overlaps between objects
    n_objs = len(objects)
    for i in range(n_objs):
        obj_a = objects[i]
        min_a, max_a = get_object_aabb(obj_a)
        for j in range(i + 1, n_objs):
            obj_b = objects[j]
            min_b, max_b = get_object_aabb(obj_b)
            if check_aabb_intersection(min_a, max_a, min_b, max_b):
                overlapping_pairs += 1
                warnings.append(SceneIssue(
                    id=f"overlap_{obj_a.id}_{obj_b.id}",
                    severity="warning",
                    category="overlap",
                    target_ids=[obj_a.id, obj_b.id],
                    message=f"Collision overlap detected between '{obj_a.name}' and '{obj_b.name}'.",
                    suggestion="Separate these objects to prevent explosive contact forces upon simulation start."
                ))

    # 4. Check Robot clearance against obstacles at spawn position
    rx, ry, rz = project.robot.initial_position
    r_radius = max(project.robot.chassis_size[0], project.robot.chassis_size[1]) * 1.2
    for obj in objects:
        ox, oy, _ = obj.position
        dist_2d = math.sqrt((ox - rx)**2 + (oy - ry)**2)
        obj_radius = max(obj.size[0], obj.size[1]) if len(obj.size) >= 2 else obj.size[0]
        if dist_2d < (r_radius + obj_radius):
            errors.append(SceneIssue(
                id=f"robot_collision_{obj.id}",
                severity="error",
                category="robot_clearance",
                target_ids=[obj.id, project.robot.id],
                message=f"Robot spawn position ({rx:.2f}, {ry:.2f}) intersects with obstacle '{obj.name}' (distance {dist_2d:.2f}m < clearance {r_radius + obj_radius:.2f}m).",
                suggestion=f"Relocate '{obj.name}' or adjust robot initial_position."
            ))

    is_valid = len(errors) == 0

    return SceneValidationReport(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        infos=infos,
        total_objects=n_objs,
        overlapping_pairs_count=overlapping_pairs
    )
