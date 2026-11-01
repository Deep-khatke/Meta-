"""
Real Gaussian-Process Bayesian Optimization for RoboForge Scientific Intelligence.

This module implements genuine Bayesian optimization -- a Gaussian Process surrogate with an
RBF (squared-exponential) kernel plus the analytic Expected Improvement acquisition function --
in pure NumPy, with no SciPy / scikit-optimize / BoTorch dependency.

It replaces the previous "closeness-to-proposed-value" heuristic that was labeled as
"Expected Improvement / acquisition function" but contained no posterior. Here the EI is the
true closed-form expression derived from the GP posterior predictive mean and variance:

    EI(x) = (mu_improve) * Phi(z) + sigma * phi(z)

where mu_improve = (f_best - mu(x)) for minimization, z = mu_improve / sigma, Phi is the
standard-normal CDF and phi its PDF.

The optimizer operates over a normalized [0, 1]^d box defined by physical parameter bounds, so
a reviewer can confirm it never proposes out-of-bounds controls. It supports arbitrary
dimensionality, which lets the agent tune coupled multi-parameter bottlenecks (the exact case
where 1D coordinate search and random search fail).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


@dataclass
class GPParameterSpace:
    """Named continuous parameters with physical lower/upper bounds."""
    names: List[str]
    lows: List[float]
    highs: List[float]

    @property
    def dim(self) -> int:
        return len(self.names)

    def normalize(self, point: Dict[str, float]) -> np.ndarray:
        out = np.zeros(self.dim)
        for i, name in enumerate(self.names):
            lo, hi = self.lows[i], self.highs[i]
            span = (hi - lo) or 1.0
            out[i] = (float(point.get(name, (lo + hi) / 2.0)) - lo) / span
        return np.clip(out, 0.0, 1.0)

    def denormalize(self, x: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, name in enumerate(self.names):
            lo, hi = self.lows[i], self.highs[i]
            out[name] = round(float(lo + x[i] * (hi - lo)), 4)
        return out


class GaussianProcessRegressor:
    """
    Minimal GP regression with an RBF kernel and homoscedastic noise.

    k(x, x') = signal_var * exp(-||x - x'||^2 / (2 * length_scale^2))

    Uses a Cholesky solve of (K + noise^2 I) for numerically stable posterior inference.
    """

    def __init__(self, length_scale: float = 0.25, signal_var: float = 1.0, noise: float = 1e-2):
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise = noise
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Squared euclidean distance matrix
        A2 = np.sum(A * A, axis=1)[:, None]
        B2 = np.sum(B * B, axis=1)[None, :]
        sqdist = A2 + B2 - 2.0 * A @ B.T
        sqdist = np.maximum(sqdist, 0.0)
        return self.signal_var * np.exp(-0.5 * sqdist / (self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        self._X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        self._y = (y - self._y_mean) / self._y_std

        K = self._kernel(self._X, self._X) + (self.noise ** 2) * np.eye(len(self._X))
        # Jitter for numerical stability
        K += 1e-8 * np.eye(len(self._X))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, self._y))
        return self

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) of the posterior predictive in the original y-units."""
        if self._X is None or self._alpha is None or self._L is None:
            n = len(X_star)
            return np.zeros(n), np.ones(n) * self._y_std
        K_star = self._kernel(self._X, X_star)
        mean = K_star.T @ self._alpha
        v = np.linalg.solve(self._L, K_star)
        K_ss = self._kernel(X_star, X_star)
        var = np.diag(K_ss) - np.sum(v * v, axis=0)
        var = np.maximum(var, 1e-10)
        mean = mean * self._y_std + self._y_mean
        std = np.sqrt(var) * self._y_std
        return mean, std


@dataclass
class BOObservation:
    params: Dict[str, float]
    objective: float          # Lower is better (the optimizer always minimizes).
    raw_metrics: Dict[str, float] = field(default_factory=dict)


class BayesianOptimizer:
    """
    GP-based Bayesian optimizer with Expected Improvement, minimizing a scalar objective.

    The agent plugs in an objective that encodes the research goal (e.g. collisions + time
    penalties - success bonus); lower is better. `suggest()` returns the next parameter dict to
    evaluate, trading off exploitation (low predicted objective) and exploration (high variance).
    """

    def __init__(
        self,
        space: GPParameterSpace,
        length_scale: float = 0.3,
        noise: float = 0.05,
        xi: float = 0.01,
        seed: int = 7,
    ):
        self.space = space
        self.gp = GaussianProcessRegressor(length_scale=length_scale, noise=noise)
        self.xi = xi
        self.rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
        self.observations: List[BOObservation] = []

    def register(self, params: Dict[str, float], objective: float, raw_metrics: Optional[Dict[str, float]] = None) -> None:
        self.observations.append(BOObservation(params=dict(params), objective=float(objective), raw_metrics=raw_metrics or {}))

    def _candidate_pool(self, n: int) -> np.ndarray:
        """Latin-hypercube-ish random pool over the normalized box for acquisition search."""
        return self._np_rng.uniform(0.0, 1.0, size=(n, self.space.dim))

    def expected_improvement(self, X_norm: np.ndarray) -> np.ndarray:
        """Analytic EI from the fitted GP posterior (minimization)."""
        mean, std = self.gp.predict(X_norm)
        f_best = min(o.objective for o in self.observations)
        ei = np.zeros(len(X_norm))
        for i in range(len(X_norm)):
            s = std[i]
            if s < 1e-9:
                ei[i] = 0.0
                continue
            improve = f_best - mean[i] - self.xi
            z = improve / s
            ei[i] = improve * _norm_cdf(z) + s * _norm_pdf(z)
        return ei

    def _sobol_like_seed(self, idx: int) -> np.ndarray:
        """Deterministic space-filling (corners + center) cold-start design."""
        d = self.space.dim
        # First points: center, then a coarse low-discrepancy spread.
        if idx == 0:
            return np.full(d, 0.5)
        # Use a scrambled Halton-ish sequence for the next few points.
        bases = [2, 3, 5, 7, 11, 13][:d] or [2]
        pt = np.zeros(d)
        for j in range(d):
            base = bases[j % len(bases)]
            f, i, r = 1.0, idx + 1, 0.0
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            pt[j] = r
        return pt

    def suggest(self, pool_size: int = 4096) -> Dict[str, float]:
        """Proposes the next parameter set to evaluate by maximizing Expected Improvement."""
        # Cold start: deterministic space-filling design until we can fit a GP.
        n_seed = max(3, self.space.dim + 1)
        if len(self.observations) < n_seed:
            x = self._sobol_like_seed(len(self.observations))
            return self.space.denormalize(np.clip(x, 0.0, 1.0))

        X = np.array([self.space.normalize(o.params) for o in self.observations])
        y = np.array([o.objective for o in self.observations])
        self.gp.fit(X, y)

        pool = self._candidate_pool(pool_size)
        ei = self.expected_improvement(pool)
        best_idx = int(np.argmax(ei))

        # If EI is flat (fully explored), fall back to the most uncertain point.
        if ei[best_idx] <= 1e-12:
            _, std = self.gp.predict(pool)
            best_idx = int(np.argmax(std))

        return self.space.denormalize(pool[best_idx])

    def best(self) -> Optional[BOObservation]:
        if not self.observations:
            return None
        return min(self.observations, key=lambda o: o.objective)

    def posterior_at_best(self) -> Dict[str, float]:
        """Posterior mean/std at the current best point, for honest reporting of confidence."""
        best = self.best()
        if not best or len(self.observations) < 2:
            return {"predicted_objective": best.objective if best else 0.0, "posterior_std": 0.0}
        X = np.array([self.space.normalize(o.params) for o in self.observations])
        y = np.array([o.objective for o in self.observations])
        self.gp.fit(X, y)
        xb = self.space.normalize(best.params).reshape(1, -1)
        mean, std = self.gp.predict(xb)
        return {"predicted_objective": round(float(mean[0]), 4), "posterior_std": round(float(std[0]), 4)}
