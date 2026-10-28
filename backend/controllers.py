from __future__ import annotations
import time
import math
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

class PIDGains(BaseModel):
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    min_output: float = -100.0
    max_output: float = 100.0
    integral_limit: float = 50.0

class PIDController:
    """Discrete-time PID feedback controller with anti-windup clamping and derivative filtering."""
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, min_output: float = -100.0, max_output: float = 100.0, integral_limit: float = 50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.integral_limit = integral_limit

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._has_prev_error: bool = False

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._has_prev_error = False

    def compute(self, target: float, current: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        error = target - current
        
        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup clamping
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        i_term = self.ki * self._integral

        # Derivative term
        if self._has_prev_error:
            d_error = (error - self._prev_error) / dt
        else:
            d_error = 0.0
            self._has_prev_error = True
        self._prev_error = error
        d_term = self.kd * d_error

        output = p_term + i_term + d_term
        return max(self.min_output, min(self.max_output, output))


class TrajectoryPoint(BaseModel):
    time_from_start: float # in seconds
    positions: Dict[str, float]
    velocities: Optional[Dict[str, float]] = None

class TrajectoryTracker:
    """Interpolates waypoints over time for smooth multi-joint or mobile base trajectory execution."""
    def __init__(self, points: List[TrajectoryPoint]):
        self.points = sorted(points, key=lambda p: p.time_from_start)
        self.total_duration = self.points[-1].time_from_start if self.points else 0.0

    def sample(self, current_time: float) -> Dict[str, float]:
        """Linearly interpolates joint target values at a given time offset."""
        if not self.points:
            return {}
        if current_time <= self.points[0].time_from_start:
            return dict(self.points[0].positions)
        if current_time >= self.points[-1].time_from_start:
            return dict(self.points[-1].positions)

        # Find enclosing interval [p_prev, p_next]
        for i in range(len(self.points) - 1):
            p0 = self.points[i]
            p1 = self.points[i + 1]
            if p0.time_from_start <= current_time <= p1.time_from_start:
                span = p1.time_from_start - p0.time_from_start
                alpha = (current_time - p0.time_from_start) / span if span > 1e-6 else 0.0
                interpolated: Dict[str, float] = {}
                for jname, v0 in p0.positions.items():
                    v1 = p1.positions.get(jname, v0)
                    interpolated[jname] = v0 + alpha * (v1 - v0)
                return interpolated
        return dict(self.points[-1].positions)
