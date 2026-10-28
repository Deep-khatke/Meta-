from __future__ import annotations
import math
import random
import copy
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from backend.models import Project, SceneObject
from backend.simulation import MuJoCoEngine

class BoundingBox2D(BaseModel):
    object_id: str
    label: str
    bbox: List[float] # [x_min, y_min, width, height] in pixel coordinates
    area: float
    confidence: float = 1.0

class SyntheticFrame(BaseModel):
    frame_id: int
    image_data_url: str
    width: int
    height: int
    camera_name: str
    annotations: List[BoundingBox2D] = Field(default_factory=list)

class DomainRandomizationConfig(BaseModel):
    randomize_object_positions: bool = True
    position_jitter_m: float = 0.3
    randomize_object_colors: bool = True
    randomize_lighting: bool = False
    randomize_friction: bool = False

class COCODataset(BaseModel):
    info: Dict[str, Any] = Field(default_factory=lambda: {"description": "RoboForge Synthetic Perception Dataset", "version": "1.0"})
    categories: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    annotations: List[Dict[str, Any]] = Field(default_factory=list)


def project_3d_box_to_2d(
    box_center_world: List[float],
    box_half_sizes: List[float],
    cam_pos: List[float],
    cam_yaw_rad: float,
    cam_pitch_rad: float,
    width: int,
    height: int,
    fov_deg: float
) -> Optional[BoundingBox2D]:
    """Projects 3D bounding box corners into 2D camera pixel coordinates using pinhole camera geometry."""
    cx, cy, cz = cam_pos
    hx, hy, hz = box_half_sizes

    # 8 corners of the 3D bounding box
    corners_world = [
        [box_center_world[0] + dx, box_center_world[1] + dy, box_center_world[2] + dz]
        for dx in (-hx, hx) for dy in (-hy, hy) for dz in (-hz, hz)
    ]

    # Camera rotation matrix: camera looks along +X in robot frame, pitch down/up, yaw
    # Standard robotics camera frame: Z forward, X right, Y down
    # MuJoCo free / forward camera: +X right, +Y up, -Z into scene or +Z up, +X forward
    focal_length = (width * 0.5) / math.tan(math.radians(fov_deg * 0.5))
    u0 = width * 0.5
    v0 = height * 0.5

    cos_y = math.cos(-cam_yaw_rad)
    sin_y = math.sin(-cam_yaw_rad)

    pixels_u = []
    pixels_v = []
    points_in_front = 0

    for pt in corners_world:
        # Translate relative to camera
        dx = pt[0] - cx
        dy = pt[1] - cy
        dz = pt[2] - cz

        # Rotate around camera yaw
        x_cam = dx * cos_y - dy * sin_y
        y_cam = dx * sin_y + dy * cos_y
        z_cam = dz

        # Longitudinal distance along camera optical axis (assumed forward +X)
        if x_cam > 0.05:
            points_in_front += 1
            # Pinhole projection: horizontal is -y_cam, vertical is z_cam
            u = u0 - (focal_length * y_cam / x_cam)
            v = v0 - (focal_length * z_cam / x_cam)
            pixels_u.append(u)
            pixels_v.append(v)

    if points_in_front < 4 or not pixels_u:
        return None

    u_min = max(0.0, min(pixels_u))
    u_max = min(float(width), max(pixels_u))
    v_min = max(0.0, min(pixels_v))
    v_max = min(float(height), max(pixels_v))

    w_box = u_max - u_min
    h_box = v_max - v_min

    if w_box < 4 or h_box < 4:
        return None

    return BoundingBox2D(
        object_id="",
        label="",
        bbox=[round(u_min, 1), round(v_min, 1), round(w_box, 1), round(h_box, 1)],
        area=round(w_box * h_box, 1)
    )


class SyntheticDataGenerator:
    """Generates synthetic multi-modal perception datasets with ground-truth 2D bounding boxes and domain randomization."""
    def __init__(self, project: Project):
        self.project = project

    def randomize_environment(self, config: DomainRandomizationConfig) -> Project:
        """Creates a randomized clone of the project without modifying the original."""
        proj = copy.deepcopy(self.project)
        palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]

        for obj in proj.objects:
            if obj.is_static and not config.randomize_object_positions:
                continue
            if config.randomize_object_positions:
                jitter_x = random.uniform(-config.position_jitter_m, config.position_jitter_m)
                jitter_y = random.uniform(-config.position_jitter_m, config.position_jitter_m)
                obj.position[0] += jitter_x
                obj.position[1] += jitter_y
            if config.randomize_object_colors:
                obj.color = random.choice(palette)
            if config.randomize_friction:
                obj.friction[0] = round(random.uniform(0.3, 1.5), 3)

        return proj

    def capture_labeled_frame(self, frame_id: int = 1, width: int = 320, height: int = 240) -> SyntheticFrame:
        """Steps MuJoCo offscreen camera and computes verified 2D bounding box annotations."""
        engine = MuJoCoEngine(self.project)
        camera_meta = engine.capture_camera_frame(width=width, height=height)

        cam_cfg = self.project.sensors.camera
        rx, ry, rz = self.project.robot.initial_position
        cam_world_pos = [rx + cam_cfg.mount_pos[0], ry + cam_cfg.mount_pos[1], rz + cam_cfg.mount_pos[2]]
        cam_yaw = math.radians(self.project.robot.initial_yaw)

        annotations: List[BoundingBox2D] = []
        for obj in self.project.objects:
            half_sizes = [obj.size[0], obj.size[1], obj.size[2]] if obj.type == "box" else [obj.size[0], obj.size[0], obj.size[0]]
            bbox = project_3d_box_to_2d(
                box_center_world=obj.position,
                box_half_sizes=half_sizes,
                cam_pos=cam_world_pos,
                cam_yaw_rad=cam_yaw,
                cam_pitch_rad=0.0,
                width=width,
                height=height,
                fov_deg=cam_cfg.fov_deg
            )
            if bbox:
                bbox.object_id = obj.id
                bbox.label = obj.name
                annotations.append(bbox)

        return SyntheticFrame(
            frame_id=frame_id,
            image_data_url=camera_meta["image_data_url"],
            width=width,
            height=height,
            camera_name="robot_onboard_camera",
            annotations=annotations
        )

    def generate_coco_dataset(self, num_samples: int = 5, width: int = 320, height: int = 240) -> COCODataset:
        """Exports a full batch of randomized frames into standard COCO object detection format."""
        categories_map: Dict[str, int] = {}
        coco_categories: List[Dict[str, Any]] = []
        coco_images: List[Dict[str, Any]] = []
        coco_annotations: List[Dict[str, Any]] = []

        ann_id = 1
        config = DomainRandomizationConfig()

        for s_idx in range(1, num_samples + 1):
            randomized_proj = self.randomize_environment(config)
            gen = SyntheticDataGenerator(randomized_proj)
            frame = gen.capture_labeled_frame(frame_id=s_idx, width=width, height=height)

            coco_images.append({
                "id": s_idx,
                "file_name": f"synthetic_frame_{s_idx:04d}.png",
                "width": width,
                "height": height
            })

            for ann in frame.annotations:
                cat_name = ann.label
                if cat_name not in categories_map:
                    cat_id = len(categories_map) + 1
                    categories_map[cat_name] = cat_id
                    coco_categories.append({
                        "id": cat_id,
                        "name": cat_name,
                        "supercategory": "object"
                    })
                cat_id = categories_map[cat_name]

                coco_annotations.append({
                    "id": ann_id,
                    "image_id": s_idx,
                    "category_id": cat_id,
                    "bbox": ann.bbox,
                    "area": ann.area,
                    "iscrowd": 0
                })
                ann_id += 1

        return COCODataset(
            categories=coco_categories,
            images=coco_images,
            annotations=coco_annotations
        )
