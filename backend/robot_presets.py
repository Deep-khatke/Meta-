"""Articulated humanoid and quadruped (robot dog) presets.

Both are fixed-base `GenericRobotModel`s: the base is mounted at its natural standing height, every limb is a
chain of revolute joints driven by MuJoCo position actuators, and the viewport draws them with the designed
white-and-grey link meshes (frontend `robots/presetVisuals.ts`). Link origins here must match the origin
tables in `humanoidModel.ts` / `dogModel.ts`, because each link's mesh is authored relative to its origin.

Joint sign conventions (right-handed, +Z up, +X forward, +Y left):
  * pitch joints rotate about +Y: a hanging limb swings BACKWARD for positive angles.
  * roll joints rotate about +X: a hanging limb on the left side (y > 0) swings OUTWARD for positive angles.
"""
from typing import Dict, List, Optional, Tuple

from backend.models import GenericRobotModel, JointLimit, JointModel, LinkGeom, LinkInertial, LinkModel

V3 = Tuple[float, float, float]
_GREY = "#9a9a9a"
_X, _Y, _Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

# link name -> (origin in the standing pose, mass kg, collision geom type, size, geom centre relative to origin)
LinkSpec = Dict[str, Tuple[V3, float, str, Tuple[float, ...], V3]]
# (joint name, parent link, child link, axis, lower, upper, kp, kv)
JointSpec = List[Tuple[str, str, str, V3, float, float, float, float]]


def _link(name: str, mass: float, gtype: str, size: Tuple[float, ...], gpos: V3) -> LinkModel:
    def geom() -> LinkGeom:
        return LinkGeom(type=gtype, size=list(size), origin_pos=list(gpos), color=_GREY)

    i = max(1e-3, mass * (max(size) ** 2) * 0.4)
    return LinkModel(
        name=name, visual=geom(), collision=geom(),
        inertial=LinkInertial(mass=mass, origin_pos=list(gpos), ixx=i, iyy=i, izz=i),
    )


def _build(name: str, root: str, links: LinkSpec, joints: JointSpec, fixed: bool = True) -> GenericRobotModel:
    origins = {k: v[0] for k, v in links.items()}
    jm: List[JointModel] = []
    for jname, parent, child, axis, lo, hi, kp, kv in joints:
        po, co = origins[parent], origins[child]
        jm.append(JointModel(
            name=jname, type="revolute", parent_link=parent, child_link=child,
            origin_pos=[round(co[i] - po[i], 5) for i in range(3)], axis=list(axis),
            limit=JointLimit(lower=lo, upper=hi, effort=120.0, velocity=6.0),
            damping=2.0, armature=0.02, kp=kp, kv=kv,
        ))
    return GenericRobotModel(
        name=name, root_link=root, base_position=list(origins[root]), base_rpy=[0.0, 0.0, 0.0],
        is_fixed_base=fixed,
        links=[_link(k, v[1], v[2], v[3], v[4]) for k, v in links.items()], joints=jm,
    )


def build_humanoid(free: bool = False) -> GenericRobotModel:
    links: LinkSpec = {
        "pelvis": ((0, 0, 0.68), 5.0, "box", (0.085, 0.13, 0.055), (0, 0, 0)),
        "chest": ((0, 0, 0.74), 8.0, "box", (0.11, 0.17, 0.2), (0, 0, 0.23)),
        "head": ((0, 0, 1.12), 2.0, "sphere", (0.09,), (0.01, 0, 0.13)),
    }
    joints: JointSpec = [
        ("waist_yaw", "pelvis", "chest", _Z, -1.0, 1.0, 300, 15),
        ("neck_yaw", "chest", "head", _Z, -1.2, 1.2, 80, 5),
    ]
    for side, s in (("l", 1), ("r", -1)):
        links[f"{side}_shoulder"] = ((0, s * 0.205, 1.07), 0.6, "sphere", (0.05,), (0, 0, 0))
        links[f"{side}_upper_arm"] = ((0, s * 0.205, 1.07), 1.2, "cylinder", (0.04, 0.135), (0, 0, -0.135))
        links[f"{side}_forearm"] = ((-0.01, s * 0.228, 0.8), 1.0, "cylinder", (0.03, 0.1), (0, 0, -0.1))
        links[f"{side}_thigh"] = ((0, s * 0.095, 0.64), 3.2, "cylinder", (0.055, 0.145), (0, 0, -0.145))
        links[f"{side}_shin"] = ((0.015, s * 0.1, 0.35), 2.4, "cylinder", (0.042, 0.13), (0, 0, -0.13))
        links[f"{side}_foot"] = ((-0.01, s * 0.1, 0.085), 0.8, "box", (0.11, 0.05, 0.025), (0.055, 0, -0.06))
        roll_lo, roll_hi = (-0.3, 3.0) if s > 0 else (-3.0, 0.3)
        joints += [
            (f"{side}_shoulder_roll", "chest", f"{side}_shoulder", _X, roll_lo, roll_hi, 120, 8),
            (f"{side}_shoulder_pitch", f"{side}_shoulder", f"{side}_upper_arm", _Y, -2.6, 1.2, 120, 8),
            (f"{side}_elbow", f"{side}_upper_arm", f"{side}_forearm", _Y, -2.4, 0.0, 90, 6),
            (f"{side}_hip_pitch", "pelvis", f"{side}_thigh", _Y, -1.6, 0.8, 500, 25),
            (f"{side}_knee", f"{side}_thigh", f"{side}_shin", _Y, 0.0, 2.4, 400, 20),
            (f"{side}_ankle", f"{side}_shin", f"{side}_foot", _Y, -0.7, 0.7, 150, 8),
        ]
    return _build("ForgeBot Humanoid", "pelvis", links, joints, fixed=not free)


_BZ = 0.44


def build_dog(free: bool = False) -> GenericRobotModel:
    links: LinkSpec = {"body": ((0, 0, _BZ), 8.0, "box", (0.26, 0.11, 0.075), (0, 0, 0))}
    joints: JointSpec = []
    for fr, hx in (("f", 0.22), ("r", -0.22)):
        for side, s in (("l", 1), ("r", -1)):
            p = f"{fr}{side}"
            links[f"{p}_hip_link"] = ((hx, s * 0.12, _BZ - 0.01), 0.5, "sphere", (0.04,), (0, 0, 0))
            links[f"{p}_thigh"] = ((hx, s * 0.148, _BZ - 0.01), 0.9, "cylinder", (0.035, 0.105), (-0.032, 0, -0.105))
            links[f"{p}_shin"] = ((hx - 0.065, s * 0.148, 0.22), 0.5, "sphere", (0.024,), (0.08, 0, -0.196))
            joints += [
                (f"{p}_abd", "body", f"{p}_hip_link", _X, -0.5, 0.5, 250, 12),
                (f"{p}_hip", f"{p}_hip_link", f"{p}_thigh", _Y, -1.0, 1.0, 400, 20),
                (f"{p}_knee", f"{p}_thigh", f"{p}_shin", _Y, -1.2, 1.2, 300, 15),
            ]
    return _build("ForgeBot Dog", "body", links, joints, fixed=not free)


def build_preset(kind: str, free: bool = False) -> Optional[GenericRobotModel]:
    k = (kind or "").lower()
    if k in ("humanoid", "biped"):
        return build_humanoid(free)
    if k in ("dog", "quadruped", "robot_dog"):
        return build_dog(free)
    return None


def _zeros(jnames: List[str]) -> Dict[str, float]:
    return {n: 0.0 for n in jnames}


def preset_motion(robot_name: str, motion: Optional[str], jnames: List[str]) -> Optional[dict]:
    """Authored motions for the presets. Returns {"keyframes": [...], "loop": bool} or None."""
    n = (robot_name or "").lower()
    m = (motion or "wave").lower().strip()
    have = set(jnames)

    def kf(targets: Dict[str, float], dur: float) -> dict:
        return {"targets": {k: v for k, v in targets.items() if k in have}, "duration": dur}

    if "humanoid" in n:
        if m in ("home", "rest", "zero", "stand"):
            return {"keyframes": [kf(_zeros(jnames), 1.2)], "loop": False}
        if m in ("walk", "march", "step"):
            a = dict(l_hip_pitch=-0.4, l_knee=0.8, l_ankle=0.25, r_hip_pitch=0.25, r_knee=0.1, r_ankle=-0.15,
                     l_shoulder_pitch=0.5, r_shoulder_pitch=-0.5, l_elbow=-0.3, r_elbow=-0.7, waist_yaw=0.12)
            b = dict(r_hip_pitch=-0.4, r_knee=0.8, r_ankle=0.25, l_hip_pitch=0.25, l_knee=0.1, l_ankle=-0.15,
                     r_shoulder_pitch=0.5, l_shoulder_pitch=-0.5, r_elbow=-0.3, l_elbow=-0.7, waist_yaw=-0.12)
            return {"keyframes": [kf(a, 0.5), kf(b, 0.5)], "loop": True}
        # wave (default): raise the right arm out to the side and wave the forearm
        return {"keyframes": [
            kf(dict(r_shoulder_roll=-2.5, r_elbow=-0.6), 1.0),
            kf(dict(r_elbow=-1.2), 0.4), kf(dict(r_elbow=-0.2), 0.4),
            kf(dict(r_elbow=-1.2), 0.4), kf(dict(r_elbow=-0.2), 0.4),
            kf(_zeros(jnames), 1.0),
        ], "loop": False}

    if "dog" in n or "quadruped" in n:
        if m in ("home", "rest", "zero", "stand"):
            return {"keyframes": [kf(_zeros(jnames), 1.0)], "loop": False}
        if m in ("sit", "crouch"):
            t = {}
            for p in ("fl", "fr"):
                t.update({f"{p}_hip": 0.5, f"{p}_knee": 0.5})
            for p in ("rl", "rr"):
                t.update({f"{p}_hip": -0.7, f"{p}_knee": -0.9})
            return {"keyframes": [kf(t, 1.2)], "loop": False}
        # walk / trot in place: diagonal pairs alternate between swing (forward, foot lifted) and stance
        def pose(swing: List[str], stance: List[str]) -> Dict[str, float]:
            t: Dict[str, float] = {}
            for p in swing:
                t.update({f"{p}_hip": -0.35, f"{p}_knee": -0.55})
            for p in stance:
                t.update({f"{p}_hip": 0.3, f"{p}_knee": 0.2})
            return t
        return {"keyframes": [kf(pose(["fl", "rr"], ["fr", "rl"]), 0.32), kf(pose(["fr", "rl"], ["fl", "rr"]), 0.32)], "loop": True}
    return None
