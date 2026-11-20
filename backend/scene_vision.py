"""Photo -> simulated environment.

A user-supplied photo is analysed by Gemini (multimodal) into a structured scene description (stairs, ramps, boxes,
cylinders, balls, with metric sizes and positions). `apply_scene` turns that description into real MuJoCo scene
objects (stairs become stacked static boxes, ramps become tilted boxes), and stores a text context that the chat
assistant receives with every request so it knows what the photo contains and what can be trained.

Sizes are estimates from a single photo (no depth sensor). The user may pass a hint with real measurements
("each step is 17 cm high") and those override the estimate.
"""
import base64
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

import json
import urllib.error
import urllib.request

logger = logging.getLogger("roboforge.scene_vision")

CTX: Dict[str, Any] = {"active": False, "description": "", "objects": [], "targets": {}, "stairs": None}

PROMPT = """You are the perception module of a robotics simulator. Look at the photo and describe the physical
environment a robot would have to operate in, as simple primitives, in metres. Use a frame where the robot stands
at the origin facing forward: 'forward' = distance ahead of the robot, 'left' = lateral offset (negative = right).
Estimate real-world sizes from everyday references (stair step ~0.17 m high, door ~2 m, mug ~0.1 m).
Return ONLY JSON:
{"description": "<one or two plain sentences>",
 "objects": [
  {"label": "<short noun, e.g. staircase, red cup, table, box>",
   "kind": "stairs" | "ramp" | "box" | "cylinder" | "sphere",
   "forward": <m>, "left": <m>,
   "width": <m, across left-right>, "depth": <m, along forward>, "height": <m>,
   "steps": <int, stairs only>, "step_height": <m, stairs only>, "step_depth": <m, stairs only>,
   "rise": <m, ramps only: total height gained over depth>,
   "color": "#rrggbb", "movable": <true for small loose things a robot could push or grab>}
 ],
 "suggested_tasks": ["<e.g. climb the stairs>", "<reach the red cup>"]}
Rules: stairs go UP away from the robot. Put the nearest object at least 0.7 m ahead. Max 14 objects, ignore
clutter that doesn't matter. Round cylinders/bottles/cups use kind 'cylinder' (width = diameter), balls use 'sphere'."""


def _call_gemini(img_b64: str, mime: str, hint: str) -> Dict[str, Any]:
    from backend.ai_reasoning_service import _gemini_request, _gemini_text, _parse_json_lenient
    body = {
        "contents": [{"role": "user", "parts": [
            {"text": PROMPT + (f"\nUser measurements/notes (trust these over your estimates): {hint}" if hint else "")},
            {"inline_data": {"mime_type": mime, "data": img_b64}}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    tiers = [
        (os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"), os.getenv("GEMINI_ENDPOINT")),
        (os.getenv("GEMINI_BACKUP_API_KEY", ""), os.getenv("GEMINI_BACKUP_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
         os.getenv("GEMINI_BACKUP_ENDPOINT", "vertex")),
    ] + [
        (os.getenv(f"GEMINI_BACKUP{n}_API_KEY", ""), os.getenv(f"GEMINI_BACKUP{n}_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
         os.getenv(f"GEMINI_BACKUP{n}_ENDPOINT", "developer")) for n in range(2, 10)
    ]
    last = "no Gemini key configured"
    for key, model, ep in tiers:
        if not key.strip():
            continue
        for attempt in range(3):
            if attempt:
                time.sleep(2.0 * attempt)
            try:
                url, headers = _gemini_request(model, key.strip(), ep)
                req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=90) as r:
                    return _parse_json_lenient(_gemini_text(json.loads(r.read().decode())))
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
            except Exception as e:  # noqa: BLE001
                last = str(e)
    raise RuntimeError(f"Image analysis failed ({last}).")


def _f(v, d, lo=0.01, hi=30.0) -> float:
    try:
        return float(min(hi, max(lo, float(v))))
    except Exception:
        return d


def build_objects(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the analysed description to SceneObject dicts + target positions (world frame, z up)."""
    objs: List[Dict[str, Any]] = []
    targets: Dict[str, List[float]] = {}
    stairs = None
    used = set()
    for i, o in enumerate((spec.get("objects") or [])[:14]):
        kind = str(o.get("kind", "box")).lower()
        label = str(o.get("label") or kind)[:30]
        fwd, left = _f(o.get("forward"), 1.0, -30, 30), _f(o.get("left"), 0.0, -30, 30)
        w, d, h = _f(o.get("width"), 0.4), _f(o.get("depth"), 0.4), _f(o.get("height"), 0.3)
        color = o.get("color") if re.fullmatch(r"#[0-9a-fA-F]{6}", str(o.get("color", ""))) else "#8892b0"
        base = re.sub(r"[^A-Za-z0-9_ -]", "", label) or kind
        name = base
        while name in used:
            name = f"{base} {len(used)}"
        used.add(name)
        common = dict(name=name[:50], color=color, is_static=True, mass=5.0)
        if kind == "stairs":
            n = int(_f(o.get("steps"), 4, 1, 20))
            sh = _f(o.get("step_height"), 0.17, 0.02, 0.4)
            sd = _f(o.get("step_depth"), 0.28, 0.08, 1.0)
            for k in range(n):
                objs.append(dict(common, name=f"{name} step {k + 1}"[:50], type="box",
                                 position=[fwd + (k + 0.5) * sd, left, (k + 1) * sh / 2],
                                 size=[sd / 2, w / 2, (k + 1) * sh / 2]))
            top = n * sh
            stairs = dict(label=name, steps=n, step_height=sh, step_depth=sd, width=w, forward=fwd, left=left, top=top)
            targets[name.lower()] = [fwd + (n - 0.5) * sd, left, top]
        elif kind == "ramp":
            rise = _f(o.get("rise"), h, 0.02, 3.0)
            ang = math.atan2(rise, d)
            L = math.hypot(d, rise)
            objs.append(dict(common, type="box", position=[fwd + d / 2, left, rise / 2],
                             rotation=[0.0, -math.degrees(ang), 0.0], size=[L / 2, w / 2, 0.03]))
            targets[name.lower()] = [fwd + d, left, rise]
        elif kind == "sphere":
            r = max(0.02, min(w, d, h) / 2) if o.get("height") else max(0.02, w / 2)
            objs.append(dict(common, type="sphere", position=[fwd, left, r], size=[r], is_static=not o.get("movable")))
            targets[name.lower()] = [fwd, left, r]
        elif kind == "cylinder":
            r = max(0.02, w / 2)
            objs.append(dict(common, type="cylinder", position=[fwd, left, h / 2], size=[r, h / 2],
                             is_static=not o.get("movable")))
            targets[name.lower()] = [fwd, left, h / 2]
        else:
            objs.append(dict(common, type="box", position=[fwd, left, h / 2], size=[d / 2, w / 2, h / 2],
                             is_static=not o.get("movable")))
            targets[name.lower()] = [fwd, left, h / 2]
    return {"objects": objs, "targets": targets, "stairs": stairs}


def context_block() -> str:
    if not CTX["active"]:
        return ""
    lines = [f"\n\nUser-uploaded photo, converted into the simulated scene: {CTX['description']}"]
    for lab, p in CTX["targets"].items():
        lines.append(f"  - {lab} at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m")
    st = CTX.get("stairs")
    if st:
        lines.append(f"  Stairs: {st['steps']} steps, {st['step_height'] * 100:.0f} cm high, {st['step_depth'] * 100:.0f} cm deep.")
    lines.append("To train: use tool 'train_scene_task' {task:'climb'|'go_to'|'reach', target:'<label from above>', timesteps?}. "
                 "'climb'/'go_to' train a walking robot (spawned automatically) to reach the target on real physics; "
                 "'reach' trains the robot arm to touch the target object.")
    return "\n".join(lines)


def apply_scene(spec: Dict[str, Any], pm, engine, name: str = "photo") -> Dict[str, Any]:
    from backend.models import SceneObject
    built = build_objects(spec)
    if not built["objects"]:
        raise RuntimeError("No usable objects were found in the image.")
    proj = pm.get_current()
    proj.objects = [SceneObject(**o) for o in built["objects"]]
    proj.robot, proj.robot_active = None, False
    pm.update_project(proj)
    engine.project = proj
    engine.balance_controller = None
    engine.rebuild_model(preserve_state=False)
    engine.play()
    CTX.update(active=True, description=str(spec.get("description", ""))[:400], objects=built["objects"],
               targets=built["targets"], stairs=built["stairs"], suggested=spec.get("suggested_tasks") or [], source=name)
    return {"status": "success", "description": CTX["description"], "object_count": len(built["objects"]),
            "targets": built["targets"], "stairs": built["stairs"], "suggested_tasks": CTX["suggested"]}


def from_image(data: str, mime: Optional[str], hint: str, pm, engine, name: str = "photo") -> Dict[str, Any]:
    if data.startswith("data:"):
        head, data = data.split(",", 1)
        mime = mime or head[5:].split(";")[0]
    base64.b64decode(data, validate=False)  # reject garbage early
    if len(data) > 14_000_000:
        raise RuntimeError("Image is too large (max ~10 MB).")
    spec = _call_gemini(data, mime or "image/jpeg", hint or "")
    return apply_scene(spec, pm, engine, name)
