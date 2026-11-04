"""General shape/path -> waypoint generator for the rover.

This turns ANY motion command into an ordered list of (x, y) world-frame waypoints that the
existing ContinuousTrajectoryController can follow. It is deliberately parametric and
data-driven: there is no per-shape *control* code anywhere. A square, a triangle, a heptagon,
a figure-8, a zig-zag, a spiral, a star, a straight line, or a raw list of points all collapse
to the same thing — a waypoint list — so the rover follows any of them with one mechanism.

Design:
- Named shapes map to parametric generators (circle, polygon-by-N-sides, star, figure-8,
  zig-zag, spiral, line, wave).
- "square"/"triangle"/"pentagon"/... are just regular polygons with the right side count, so
  an unbounded family of shapes works without new code.
- Freeform: an explicit `points` list is accepted verbatim.
- Everything is centered on the robot's current position (or an explicit center) and sized by a
  `size` (radius / half-extent in meters), clamped to stay inside the arena.

Returned waypoints are dense enough that the PD waypoint-follower traces smooth curves.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple, Dict, Any


# Word -> number of polygon sides. Lets "pentagon", "hexagon", "octagon", etc. work directly.
_POLYGON_SIDES = {
    "triangle": 3, "trigon": 3,
    "square": 4, "rectangle": 4, "diamond": 4, "box": 4, "quad": 4, "rhombus": 4,
    "pentagon": 5,
    "hexagon": 6,
    "heptagon": 7, "septagon": 7,
    "octagon": 8,
    "nonagon": 9, "enneagon": 9,
    "decagon": 10,
    "hendecagon": 11, "undecagon": 11,
    "dodecagon": 12,
}

# Ordinal words -> N, so "a 5 sided shape" / "five-sided polygon" / "7-gon" all resolve.
_NUMBER_WORDS = {
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def _clamp_point(x: float, y: float, bound: float) -> Tuple[float, float]:
    b = max(0.5, bound - 0.5)  # keep a margin from the wall
    return max(-b, min(b, x)), max(-b, min(b, y))


def _regular_polygon(cx: float, cy: float, radius: float, sides: int,
                     points_per_edge: int = 8, rotation: float = math.pi / 2.0) -> List[List[float]]:
    """Vertices of a regular N-gon with interpolated points along each edge so the follower
    drives straight edges and turns crisp corners. `rotation` orients the first vertex."""
    sides = max(3, int(sides))
    verts = []
    for i in range(sides):
        ang = rotation + (2.0 * math.pi * i / sides)
        verts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
    verts.append(verts[0])  # close the loop

    pts: List[List[float]] = []
    for i in range(len(verts) - 1):
        ax, ay = verts[i]
        bx, by = verts[i + 1]
        for k in range(points_per_edge):
            t = k / float(points_per_edge)
            pts.append([ax + (bx - ax) * t, ay + (by - ay) * t])
    pts.append([verts[0][0], verts[0][1]])
    return pts


def _circle(cx: float, cy: float, radius: float, n: int = 48) -> List[List[float]]:
    return [
        [cx + radius * math.cos(2.0 * math.pi * i / n),
         cy + radius * math.sin(2.0 * math.pi * i / n)]
        for i in range(n + 1)
    ]


def _figure_eight(cx: float, cy: float, radius: float, n: int = 64) -> List[List[float]]:
    # Lemniscate-ish: two lobes. x = r*sin(t), y = r*sin(t)*cos(t) scaled.
    pts = []
    for i in range(n + 1):
        t = 2.0 * math.pi * i / n
        pts.append([cx + radius * math.sin(t), cy + radius * math.sin(t) * math.cos(t)])
    return pts


def _star(cx: float, cy: float, radius: float, points: int = 5,
          inner_ratio: float = 0.45, points_per_edge: int = 5) -> List[List[float]]:
    points = max(3, int(points))
    verts = []
    for i in range(points * 2):
        r = radius if i % 2 == 0 else radius * inner_ratio
        ang = math.pi / 2.0 + (math.pi * i / points)
        verts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    verts.append(verts[0])
    pts: List[List[float]] = []
    for i in range(len(verts) - 1):
        ax, ay = verts[i]
        bx, by = verts[i + 1]
        for k in range(points_per_edge):
            t = k / float(points_per_edge)
            pts.append([ax + (bx - ax) * t, ay + (by - ay) * t])
    pts.append([verts[0][0], verts[0][1]])
    return pts


def _zigzag(cx: float, cy: float, size: float, teeth: int = 4, n_per: int = 2) -> List[List[float]]:
    # A forward sweep that alternates +/- in Y: a zig-zag / slalom.
    teeth = max(2, int(teeth))
    span = 2.0 * size
    pts: List[List[float]] = []
    x0 = cx - size
    for i in range(teeth + 1):
        x = x0 + span * i / teeth
        y = cy + (size * 0.6 if i % 2 == 0 else -size * 0.6)
        pts.append([x, y])
    # densify
    dense: List[List[float]] = []
    for i in range(len(pts) - 1):
        for k in range(n_per):
            t = k / float(n_per)
            dense.append([pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t,
                          pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t])
    dense.append(pts[-1])
    return dense


def _spiral(cx: float, cy: float, radius: float, turns: float = 2.0, n: int = 72) -> List[List[float]]:
    pts = []
    total = turns * 2.0 * math.pi
    for i in range(n + 1):
        t = total * i / n
        r = radius * (i / n)
        pts.append([cx + r * math.cos(t), cy + r * math.sin(t)])
    return pts


def _line(cx: float, cy: float, size: float, n: int = 10) -> List[List[float]]:
    return [[cx - size + (2.0 * size) * i / n, cy] for i in range(n + 1)]


def _wave(cx: float, cy: float, size: float, cycles: float = 2.0, n: int = 48) -> List[List[float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        x = cx - size + (2.0 * size) * t
        y = cy + size * 0.5 * math.sin(2.0 * math.pi * cycles * t)
        pts.append([x, y])
    return pts


def parse_number(text: str) -> Optional[int]:
    """Extract a side/point count from text: '7', 'seven', '5-sided', '7-gon'."""
    m = re.search(r"(\d+)\s*(?:-?\s*sided|-?\s*gon|\s*sides|\s*points?)", text)
    if m:
        return int(m.group(1))
    for w, n in _NUMBER_WORDS.items():
        if re.search(rf"\b{w}\b", text):
            return n
    return None


def generate_path(
    description: str = "",
    center: Optional[Sequence[float]] = None,
    size: float = 1.6,
    points: Optional[List[List[float]]] = None,
    arena_bound: float = 6.0,
    closed: Optional[bool] = None,
) -> Dict[str, Any]:
    """Resolve ANY shape/path request into waypoints.

    Returns {waypoints, loop, shape, closed}. `loop` means the follower should cycle the path
    continuously (closed shapes like circle/square), otherwise it drives it once and stops.
    """
    cx, cy = (float(center[0]), float(center[1])) if center and len(center) >= 2 else (0.0, 0.0)
    t = (description or "").lower().strip()
    size = max(0.4, min(float(size), arena_bound - 0.6))

    shape = "custom"
    loop = False
    wps: List[List[float]]

    if points:
        wps = [[float(p[0]), float(p[1])] for p in points if len(p) >= 2]
        shape = "freeform"
        # A freeform path loops if its ends nearly coincide.
        if len(wps) >= 3:
            loop = math.hypot(wps[0][0] - wps[-1][0], wps[0][1] - wps[-1][1]) < 0.4
    elif any(w in t for w in ["figure-8", "figure 8", "figure8", "eight", "infinity", "lemniscate"]) and "sided" not in t:
        wps = _figure_eight(cx, cy, size); shape = "figure-8"; loop = True
    elif any(w in t for w in ["circle", "circular", "loop", "orbit", "ring", "round"]):
        wps = _circle(cx, cy, size); shape = "circle"; loop = True
    elif any(w in t for w in ["star"]):
        npts = parse_number(t) or 5
        wps = _star(cx, cy, size, points=npts); shape = f"{npts}-point star"; loop = True
    elif any(w in t for w in ["zig", "zag", "zigzag", "slalom", "weave", "snake"]):
        wps = _zigzag(cx, cy, size); shape = "zig-zag"; loop = False
    elif any(w in t for w in ["spiral", "coil", "whirl"]):
        wps = _spiral(cx, cy, size); shape = "spiral"; loop = False
    elif any(w in t for w in ["wave", "sine", "sinusoid", "s-curve", "wavy"]):
        wps = _wave(cx, cy, size); shape = "wave"; loop = False
    elif any(w in t for w in ["straight", "line", "forward", "ahead"]):
        wps = _line(cx, cy, size); shape = "line"; loop = False
    else:
        # Polygon family: named (square/triangle/pentagon/...) or N-sided.
        sides = None
        for name, n in _POLYGON_SIDES.items():
            if name in t:
                sides = n
                break
        if sides is None:
            sides = parse_number(t)
        if sides is not None:
            sides = max(3, min(int(sides), 20))
            wps = _regular_polygon(cx, cy, size, sides)
            names = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}
            shape = names.get(sides, f"{sides}-gon")
            loop = True
        else:
            # Unknown command -> default to a circle (a sensible, visible default) rather than fail.
            wps = _circle(cx, cy, size); shape = "circle (default)"; loop = True

    if closed is not None:
        loop = bool(closed)

    wps = [list(_clamp_point(p[0], p[1], arena_bound)) for p in wps]
    return {"waypoints": wps, "loop": loop, "shape": shape, "closed": loop}
