from __future__ import annotations
import math
import heapq
from typing import List, Tuple, Dict, Set, Optional, Literal
from pydantic import BaseModel, Field

from backend.models import Project, SceneObject

class PathPlanningRequest(BaseModel):
    start_pos: List[float] = Field(..., description="[x, y] or [x, y, z] start coordinates")
    goal_pos: List[float] = Field(..., description="[x, y] or [x, y, z] goal coordinates")
    grid_resolution: float = Field(default=0.15, ge=0.05, le=1.0, description="Grid cell size in meters")
    robot_radius: float = Field(default=0.35, ge=0.05, le=2.0, description="Robot inflation clearance radius in meters")
    smooth_path: bool = Field(default=True, description="Apply line-of-sight shortcutting smoothing")

class PathPlanningResult(BaseModel):
    success: bool
    path: List[List[float]] = Field(default_factory=list, description="Waypoints in world coordinates [[x, y], ...]")
    path_length: float = 0.0
    nodes_expanded: int = 0
    message: str = ""


class GridAStarPlanner:
    def __init__(self, project: Project, resolution: float = 0.15, robot_radius: float = 0.35):
        self.project = project
        self.res = resolution
        self.robot_radius = robot_radius

        bounds = project.settings.world_boundaries if project.settings.enable_boundaries else [-15.0, 15.0, -15.0, 15.0, 0.0, 8.0]
        self.xmin, self.xmax, self.ymin, self.ymax = bounds[0], bounds[1], bounds[2], bounds[3]

        self.cols = int(math.ceil((self.xmax - self.xmin) / self.res))
        self.rows = int(math.ceil((self.ymax - self.ymin) / self.res))

        self.occupied_cells: Set[Tuple[int, int]] = set()
        self._rasterize_obstacles()

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(math.floor((x - self.xmin) / self.res))
        gy = int(math.floor((y - self.ymin) / self.res))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = self.xmin + (gx + 0.5) * self.res
        y = self.ymin + (gy + 0.5) * self.res
        return x, y

    def is_in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.cols and 0 <= gy < self.rows

    def _rasterize_obstacles(self):
        """Marks grid cells covered by obstacles inflated by robot radius."""
        inflation_cells = int(math.ceil(self.robot_radius / self.res))

        for obj in self.project.objects:
            ox, oy, _ = obj.position
            # Estimate 2D bounding radius of obstacle
            if obj.type == "box":
                rad = math.sqrt(obj.size[0]**2 + obj.size[1]**2)
            else:
                rad = obj.size[0]
            
            total_rad = rad + self.robot_radius
            cgx, cgy = self.world_to_grid(ox, oy)
            r_cells = int(math.ceil(total_rad / self.res))

            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    gx = cgx + dx
                    gy = cgy + dy
                    if self.is_in_bounds(gx, gy):
                        wx, wy = self.grid_to_world(gx, gy)
                        dist = math.sqrt((wx - ox)**2 + (wy - oy)**2)
                        if dist <= total_rad:
                            self.occupied_cells.add((gx, gy))

    def is_collision_free(self, gx: int, gy: int) -> bool:
        return self.is_in_bounds(gx, gy) and (gx, gy) not in self.occupied_cells

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], smooth: bool = True) -> PathPlanningResult:
        sgx, sgy = self.world_to_grid(start[0], start[1])
        ggx, ggy = self.world_to_grid(goal[0], goal[1])

        if not self.is_in_bounds(sgx, sgy):
            return PathPlanningResult(success=False, message="Start position outside environment bounds.")
        if not self.is_in_bounds(ggx, ggy):
            return PathPlanningResult(success=False, message="Goal position outside environment bounds.")
        if (ggx, ggy) in self.occupied_cells:
            return PathPlanningResult(success=False, message="Goal position is inside an inflated obstacle.")

        # 8-connectivity motions: (dx, dy, cost)
        sqrt2 = math.sqrt(2.0)
        motions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, sqrt2), (-1, 1, sqrt2), (1, -1, sqrt2), (-1, -1, sqrt2)
        ]

        def heuristic(ax: int, ay: int, bx: int, by: int) -> float:
            return math.sqrt((ax - bx)**2 + (ay - by)**2)

        start_node = (sgx, sgy)
        goal_node = (ggx, ggy)

        # Priority queue stores: (f_score, h_score, (gx, gy))
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (heuristic(sgx, sgy, ggx, ggy), 0.0, start_node))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_node: 0.0}
        closed_set: Set[Tuple[int, int]] = set()

        nodes_expanded = 0

        while open_set:
            f, current_g, current = heapq.heappop(open_set)

            if current == goal_node:
                # Reconstruct path
                grid_path = [current]
                while current in came_from:
                    current = came_from[current]
                    grid_path.append(current)
                grid_path.reverse()

                # Convert to world coordinates
                world_path = [list(self.grid_to_world(gx, gy)) for gx, gy in grid_path]
                world_path[0] = [start[0], start[1]]
                world_path[-1] = [goal[0], goal[1]]

                # Line-of-sight path smoothing
                if smooth and len(world_path) > 2:
                    world_path = self._smooth_path(world_path)

                total_dist = 0.0
                for i in range(len(world_path) - 1):
                    dx = world_path[i+1][0] - world_path[i][0]
                    dy = world_path[i+1][1] - world_path[i][1]
                    total_dist += math.sqrt(dx*dx + dy*dy)

                return PathPlanningResult(
                    success=True,
                    path=world_path,
                    path_length=round(total_dist, 3),
                    nodes_expanded=nodes_expanded,
                    message=f"Path found with {len(world_path)} waypoints."
                )

            closed_set.add(current)
            nodes_expanded += 1

            for dx, dy, cost in motions:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in closed_set:
                    continue
                if not self.is_collision_free(neighbor[0], neighbor[1]):
                    continue
                # Prevent diagonal corner cutting
                if dx != 0 and dy != 0:
                    if not self.is_collision_free(current[0] + dx, current[1]) or not self.is_collision_free(current[0], current[1] + dy):
                        continue

                tentative_g = g_score[current] + cost * self.res
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = heuristic(neighbor[0], neighbor[1], ggx, ggy) * self.res
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, neighbor))

        return PathPlanningResult(
            success=False,
            path=[],
            path_length=0.0,
            nodes_expanded=nodes_expanded,
            message="No collision-free path found to goal."
        )

    def _line_of_sight(self, p1: List[float], p2: List[float]) -> bool:
        """Checks if a straight line between two world points is free of obstacles."""
        dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        steps = int(math.ceil(dist / (self.res * 0.5)))
        if steps == 0:
            return True
        for s in range(steps + 1):
            alpha = s / steps
            x = p1[0] + alpha * (p2[0] - p1[0])
            y = p1[1] + alpha * (p2[1] - p1[1])
            gx, gy = self.world_to_grid(x, y)
            if not self.is_collision_free(gx, gy):
                return False
        return True

    def _smooth_path(self, path: List[List[float]]) -> List[List[float]]:
        """Greedy raycast shortcutting to eliminate unnecessary grid zig-zags."""
        smoothed = [path[0]]
        current_idx = 0
        n = len(path)
        while current_idx < n - 1:
            furthest_visible = current_idx + 1
            for next_idx in range(n - 1, current_idx, -1):
                if self._line_of_sight(path[current_idx], path[next_idx]):
                    furthest_visible = next_idx
                    break
            smoothed.append(path[furthest_visible])
            current_idx = furthest_visible
        return smoothed


def plan_collision_free_path(project: Project, request: PathPlanningRequest) -> PathPlanningResult:
    planner = GridAStarPlanner(project, resolution=request.grid_resolution, robot_radius=request.robot_radius)
    start_pt = (request.start_pos[0], request.start_pos[1])
    goal_pt = (request.goal_pos[0], request.goal_pos[1])
    return planner.plan(start_pt, goal_pt, smooth=request.smooth_path)


class TrajectoryExecutionState(BaseModel):
    trajectory_id: str
    status: Literal["idle", "planning", "running", "completed", "cancelled", "failed"] = "idle"
    start_pos: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    goal_pos: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    waypoints: List[List[float]] = Field(default_factory=list)
    current_waypoint_index: int = 0
    total_waypoints: int = 0
    distance_to_goal: float = 0.0
    progress_pct: float = 0.0
    collision_detected: bool = False
    status_message: str = "Ready"
    elapsed_time: float = 0.0


class ContinuousTrajectoryController:
    """Continuously tracks waypoints in the authoritative MuJoCo physics loop without stopping between waypoints."""
    def __init__(
        self,
        waypoints: List[List[float]],
        v_max: float = 1.2,
        w_max: float = 2.5,
        waypoint_tolerance: float = 0.28,
        goal_tolerance: float = 0.15
    ):
        self.waypoints = waypoints
        self.v_max = v_max
        self.w_max = w_max
        self.waypoint_tolerance = waypoint_tolerance
        self.goal_tolerance = goal_tolerance
        
        self.current_index: int = 0
        self.is_completed: bool = False
        self.is_failed: bool = False
        self.failure_reason: str = ""

        # Internal heading PID
        self.kp_w = 3.2
        self.kd_w = 0.2
        self._prev_e_theta: float = 0.0
        self._has_prev: bool = False

    def update(self, rx: float, ry: float, ryaw: float, dt: float) -> Tuple[float, float]:
        """Calculates continuous (v, omega) commands at the current simulation step."""
        if self.is_completed or self.is_failed or not self.waypoints:
            return 0.0, 0.0

        n = len(self.waypoints)
        cur_wp = self.waypoints[self.current_index]
        dx = cur_wp[0] - rx
        dy = cur_wp[1] - ry
        dist_to_wp = math.sqrt(dx * dx + dy * dy)

        final_wp = self.waypoints[-1]
        dist_to_goal = math.sqrt((final_wp[0] - rx)**2 + (final_wp[1] - ry)**2)

        # Final goal reached
        if dist_to_goal <= self.goal_tolerance:
            self.is_completed = True
            return 0.0, 0.0

        # Advance to next waypoint without halting if within tolerance
        while dist_to_wp <= self.waypoint_tolerance and self.current_index < n - 1:
            self.current_index += 1
            cur_wp = self.waypoints[self.current_index]
            dx = cur_wp[0] - rx
            dy = cur_wp[1] - ry
            dist_to_wp = math.sqrt(dx * dx + dy * dy)

        # Target heading towards current waypoint
        target_yaw = math.atan2(dy, dx)
        e_theta = target_yaw - ryaw
        while e_theta > math.pi:
            e_theta -= 2.0 * math.pi
        while e_theta < -math.pi:
            e_theta += 2.0 * math.pi

        # Angular PD steering
        de = (e_theta - self._prev_e_theta) / max(1e-4, dt) if self._has_prev else 0.0
        self._prev_e_theta = e_theta
        self._has_prev = True

        raw_w = self.kp_w * e_theta + self.kd_w * de
        cmd_w = max(-self.w_max, min(self.w_max, raw_w))

        # Forward velocity: smooth alignment scaling + decel near final goal
        heading_alignment = max(0.0, math.cos(e_theta))
        if abs(e_theta) > math.radians(70):
            # Heavy turn, pivot in place without racing forward
            cmd_v = 0.05
        else:
            speed_factor = heading_alignment
            # Gentle slowdown only near final goal
            if dist_to_goal < 0.6:
                speed_factor *= max(0.25, dist_to_goal / 0.6)
            cmd_v = float(self.v_max * speed_factor)

        return cmd_v, cmd_w
