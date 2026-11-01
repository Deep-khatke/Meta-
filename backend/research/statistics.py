"""
Rigorous, dependency-free statistical inference for RoboForge Scientific Intelligence.

Every empirical claim the research agent makes ("improved", "degraded", "distinguishable
from baseline") is backed by a real statistical procedure implemented here rather than a
hand-tuned threshold. Pure Python / NumPy only -- no SciPy requirement -- so it runs in the
same minimal environment as the rest of the backend.

Procedures implemented:
  * Welch's two-sample t-test (unequal variances) with a Welch-Satterthwaite dof and a
    numerically-stable Student-t survival function via a regularized incomplete beta function.
  * Bootstrap confidence intervals (percentile method) for an arbitrary statistic.
  * Two-proportion z-test for success-rate comparisons.
  * Cohen's d effect size for magnitude reporting.

The goal is that a skeptical reviewer can read a single file and verify that "statistically
significant" means an actual p-value, not `delta < -0.3`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Special functions (regularized incomplete beta -> Student-t / F distributions)
# ---------------------------------------------------------------------------

def _log_gamma(x: float) -> float:
    """Lanczos approximation of ln(Gamma(x))."""
    g = 7
    c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    if x < 0.5:
        # Reflection formula
        return math.log(math.pi / math.sin(math.pi * x)) - _log_gamma(1.0 - x)
    x -= 1.0
    a = c[0]
    t = x + g + 0.5
    for i in range(1, g + 2):
        a += c[i] / (x + i)
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    MAXIT = 200
    EPS = 3.0e-12
    FPMIN = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) -- regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, dof: float) -> float:
    """Two-sided survival probability for the Student-t distribution, P(|T| > |t|)."""
    if dof <= 0:
        return 1.0
    x = dof / (dof + t * t)
    # P(|T| > |t|) = I_x(dof/2, 1/2)
    return regularized_incomplete_beta(dof / 2.0, 0.5, x)


def normal_sf(z: float) -> float:
    """Two-sided survival probability for the standard normal, P(|Z| > |z|)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TTestResult:
    mean_a: float
    mean_b: float
    diff: float
    t_statistic: float
    dof: float
    p_value: float
    significant: bool
    effect_size_d: float
    alpha: float = 0.05

    def summary(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (
            f"diff={self.diff:+.3f}, t={self.t_statistic:.3f}, dof={self.dof:.1f}, "
            f"p={self.p_value:.4f} ({sig} at alpha={self.alpha}), Cohen's d={self.effect_size_d:.2f}"
        )


@dataclass
class ConfidenceInterval:
    point_estimate: float
    low: float
    high: float
    confidence: float = 0.95

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def summary(self) -> str:
        return f"{self.point_estimate:.3f} [{self.low:.3f}, {self.high:.3f}] ({int(self.confidence*100)}% CI)"


@dataclass
class ProportionTestResult:
    p_a: float
    p_b: float
    diff: float
    z_statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05


# ---------------------------------------------------------------------------
# Core procedures
# ---------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _variance(xs: Sequence[float], ddof: int = 1) -> float:
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def welch_t_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    alpha: float = 0.05,
) -> TTestResult:
    """
    Welch's unequal-variance two-sample t-test.

    sample_a is the candidate, sample_b is the baseline. A negative diff means the candidate
    has a lower mean than the baseline (good when minimizing collisions / time).
    """
    na, nb = len(sample_a), len(sample_b)
    mean_a, mean_b = _mean(sample_a), _mean(sample_b)
    diff = mean_a - mean_b

    var_a = _variance(sample_a)
    var_b = _variance(sample_b)

    # Pooled std for Cohen's d
    pooled_sd = math.sqrt(((var_a + var_b) / 2.0)) if (var_a + var_b) > 0 else 0.0
    effect_d = (diff / pooled_sd) if pooled_sd > 1e-12 else 0.0

    se_sq = (var_a / na if na > 0 else 0.0) + (var_b / nb if nb > 0 else 0.0)

    if se_sq <= 1e-15:
        # Zero variance in both groups: distinguishable iff the means genuinely differ.
        significant = abs(diff) > 1e-9
        return TTestResult(
            mean_a=mean_a, mean_b=mean_b, diff=diff,
            t_statistic=math.inf if significant else 0.0,
            dof=float(max(1, na + nb - 2)),
            p_value=0.0 if significant else 1.0,
            significant=significant, effect_size_d=effect_d, alpha=alpha,
        )

    t_stat = diff / math.sqrt(se_sq)

    # Welch-Satterthwaite degrees of freedom
    num = se_sq ** 2
    den = 0.0
    if na > 1:
        den += (var_a / na) ** 2 / (na - 1)
    if nb > 1:
        den += (var_b / nb) ** 2 / (nb - 1)
    dof = (num / den) if den > 1e-15 else float(max(1, na + nb - 2))

    p_value = student_t_sf(t_stat, dof)
    return TTestResult(
        mean_a=mean_a, mean_b=mean_b, diff=diff,
        t_statistic=t_stat, dof=dof, p_value=p_value,
        significant=p_value < alpha, effect_size_d=effect_d, alpha=alpha,
    )


def bootstrap_ci(
    sample: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = _mean,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 12345,
) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval for an arbitrary statistic."""
    if not sample:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence)
    if len(sample) == 1:
        v = float(sample[0])
        return ConfidenceInterval(v, v, v, confidence)

    rng = random.Random(seed)
    n = len(sample)
    estimates: List[float] = []
    for _ in range(n_resamples):
        resample = [sample[rng.randrange(n)] for _ in range(n)]
        estimates.append(statistic(resample))
    estimates.sort()

    lo_idx = int((1.0 - confidence) / 2.0 * n_resamples)
    hi_idx = int((1.0 + confidence) / 2.0 * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))

    return ConfidenceInterval(
        point_estimate=statistic(sample),
        low=estimates[lo_idx],
        high=estimates[hi_idx],
        confidence=confidence,
    )


def bootstrap_diff_ci(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 12345,
) -> ConfidenceInterval:
    """Bootstrap CI for the difference of means (candidate minus baseline)."""
    if not sample_a or not sample_b:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence)
    rng = random.Random(seed)
    na, nb = len(sample_a), len(sample_b)
    diffs: List[float] = []
    for _ in range(n_resamples):
        ra = [sample_a[rng.randrange(na)] for _ in range(na)]
        rb = [sample_b[rng.randrange(nb)] for _ in range(nb)]
        diffs.append(_mean(ra) - _mean(rb))
    diffs.sort()
    lo_idx = max(0, int((1.0 - confidence) / 2.0 * n_resamples))
    hi_idx = min(n_resamples - 1, int((1.0 + confidence) / 2.0 * n_resamples) - 1)
    return ConfidenceInterval(
        point_estimate=_mean(sample_a) - _mean(sample_b),
        low=diffs[lo_idx],
        high=diffs[hi_idx],
        confidence=confidence,
    )


def two_proportion_z_test(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    alpha: float = 0.05,
) -> ProportionTestResult:
    """Two-proportion z-test (candidate A vs baseline B), used for success-rate comparisons."""
    if n_a == 0 or n_b == 0:
        return ProportionTestResult(0.0, 0.0, 0.0, 0.0, 1.0, False, alpha)
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se <= 1e-15:
        significant = abs(p_a - p_b) > 1e-9
        return ProportionTestResult(
            p_a, p_b, p_a - p_b,
            math.inf if significant else 0.0,
            0.0 if significant else 1.0, significant, alpha,
        )
    z = (p_a - p_b) / se
    p_value = normal_sf(z)
    return ProportionTestResult(p_a, p_b, p_a - p_b, z, p_value, p_value < alpha, alpha)


def summarize_sample(sample: Sequence[float]) -> dict:
    """Mean, std, standard error, min, max for reporting."""
    n = len(sample)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "min": 0.0, "max": 0.0}
    m = _mean(sample)
    sd = math.sqrt(_variance(sample)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(m, 4),
        "std": round(sd, 4),
        "sem": round(sd / math.sqrt(n), 4) if n > 0 else 0.0,
        "min": round(min(sample), 4),
        "max": round(max(sample), 4),
    }
