"""Design quantities for supervision validity: the generator health statistic of Section 2,
the recomputation rule and the information in a five-level preference of Section 3, the
leakage curve of Section 4 and the economics of Batch Zero in Section 5.

Every figure is closed-form; the generator statistic is also evaluated on the reference
run of the constrained generator in constraints.py.
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
from statistics import NormalDist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import constraints as c  # noqa: E402

PHI = NormalDist().cdf
PHI_INV = NormalDist().inv_cdf


def pdf(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


# -------------------------------------------------------- Section 2: generator health statistic
def _gram_inverse():
    sigma = c.covariance()
    at = c.transpose(c.A)
    return sigma, c.inverse(c.matmul(c.matmul(c.A, sigma), at))


def noncentrality(mean) -> float:
    """delta = (b - A mu)^T (A Sigma A^T)^-1 (b - A mu) for an upstream mean `mean`."""
    _, gram_inv = _gram_inverse()
    gap = [c.B_VEC[r] - sum(c.A[r][j] * mean[j] for j in range(len(mean))) for r in range(len(c.A))]
    return sum(gap[r] * gram_inv[r][s] * gap[s] for r in range(len(gap)) for s in range(len(gap)))


def displacement_mean(draws: int = 200_000, seed: int = 3, mean=None) -> float:
    """Average of T = (b - A Z)^T (A Sigma A^T)^-1 (b - A Z) over a run of the generator."""
    mean = c.MEAN if mean is None else mean
    normal = c.normal_stream(random.Random(seed))
    sigma, gram_inv = _gram_inverse()
    low = c.cholesky(sigma)
    d, m = len(mean), len(c.A)
    total = 0.0
    for _ in range(draws):
        e = [next(normal) for _ in range(d)]
        z = [mean[i] + sum(low[i][k] * e[k] for k in range(i + 1)) for i in range(d)]
        gap = [c.B_VEC[r] - sum(c.A[r][j] * z[j] for j in range(d)) for r in range(m)]
        total += sum(gap[r] * gram_inv[r][s] * gap[s] for r in range(m) for s in range(m))
    return total / draws


def drift_signal(delta: float, m: int, batch: int) -> float:
    """Standard errors by which a drift of noncentrality delta moves the batch mean of T."""
    return delta / math.sqrt(2 * (m + 2 * delta) / batch)


# ------------------------------------------------------------ Section 3: recomputation rule
def stale_rate(ratio: float) -> float:
    return PHI(-ratio)


def recompute_threshold(alpha: float) -> float:
    """Variants with margin-to-noise ratio below this are recomputed; the rest inherit their
    label with error at most alpha."""
    return PHI_INV(1 - alpha)


# ---------------------------------------------------------------- Section 4: leakage curve
def ou_midpoint_ratio(kappa_h: float) -> float:
    """Var(X_t | F_s, X_T) / Var(X_t | F_s) at the midpoint for an Ornstein-Uhlenbeck path,
    with kappa_h = mean-reversion speed times half the interval."""
    return 1 / (1 + math.exp(-2 * kappa_h))


def ou_midpoint_ratio_from_covariance(kappa: float, h: float) -> float:
    def cov(u, v):
        return (math.exp(-kappa * abs(u - v)) - math.exp(-kappa * (u + v))) / (2 * kappa)

    t, big_t = h, 2 * h
    return (cov(t, t) - cov(t, big_t) ** 2 / cov(big_t, big_t)) / cov(t, t)


def rmse_drop(ratio: float) -> float:
    return 1 - math.sqrt(ratio)


# ----------------------------------------------- Section 3: information in a graded preference
def information(cuts, gap: float) -> float:
    """Fisher information about the latent quality gap carried by one ordered response
    with the given cutpoints (in units of judgment noise)."""
    edges = [-math.inf, *cuts, math.inf]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        p = PHI(b - gap) - PHI(a - gap)
        dp = (pdf(a - gap) if math.isfinite(a) else 0.0) - (pdf(b - gap) if math.isfinite(b) else 0.0)
        if p > 1e-15:
            total += dp * dp / p
    return total


def best_five_level_cuts(step: float = 0.01):
    """Symmetric cutpoints (-c2, -c1, c1, c2) that maximize information at a tie."""
    best = (0.0, None)
    c1 = step
    while c1 < 1.5:
        c2 = c1 + step
        while c2 < 3.0:
            value = information((-c2, -c1, c1, c2), 0.0)
            if value > best[0]:
                best = (value, (round(c1, 2), round(c2, 2)))
            c2 += step
        c1 += step
    return best


# -------------------------------------------------------------- Section 5: Batch Zero economics
def wrong_records_per_family(k: int, eps: float, detect: float, resolves: int) -> float:
    return k * eps * (1 - detect) ** resolves


def best_variants(cost_ratio: float, rho: float) -> float:
    return math.sqrt(cost_ratio * (1 - rho) / rho)


def variance_at_budget(k: float, cost_ratio: float, rho: float) -> float:
    """Variance of the family-clustered mean at a fixed budget, up to a constant:
    (c_G + k c_v) [1 + (k - 1) rho] / k with c_v = 1 and c_G = cost_ratio."""
    return (cost_ratio + k) * (1 + (k - 1) * rho) / k


def run() -> dict:
    m = len(c.A)
    drifted = list(c.MEAN)
    drifted[1] *= 1.10                      # upstream receivables mean drifts by 10%
    delta0, delta1 = noncentrality(c.MEAN), noncentrality(drifted)
    best_info, best_cuts = best_five_level_cuts()
    return {
        "displacement": {
            "constraints": m,
            "noncentrality_reference": delta0,
            "mean_on_reference_run": displacement_mean(),
            "noncentrality_after_10pct_receivables_drift": delta1,
            "mean_after_drift": displacement_mean(mean=drifted),
            "standard_errors_at_batch_1000": drift_signal(delta1 - delta0, m, 1_000),
        },
        "recompute": {
            "stale_rate": {r: stale_rate(r) for r in (0.25, 1.0, 2.0, 3.0)},
            "threshold": {a: recompute_threshold(a) for a in (0.05, 0.01)},
            "two_covenant_bound": stale_rate(1.5) + stale_rate(2.0),
        },
        "leakage": {
            kh: {"ratio": ou_midpoint_ratio(kh), "rmse_drop": rmse_drop(ou_midpoint_ratio(kh)),
                 "check": ou_midpoint_ratio_from_covariance(kh if kh > 0 else 1e-6, 1.0)}
            for kh in (0.0, 0.25, 0.5, 1.0, 2.0)
        },
        "information": {
            gap: {"binary": information((0.0,), gap),
                  "five_level": information((-1.5, -0.5, 0.5, 1.5), gap)}
            for gap in (0.0, 0.5, 1.0)
        },
        "best_five_level": {"cuts": best_cuts, "information_at_tie": best_info},
        "batch_zero": {
            "wrong_records": [wrong_records_per_family(20, 0.15, 0.8, r) for r in (0, 1, 2)],
            "best_variants": {f"cost ratio {cr}, rho {rho}": best_variants(cr, rho)
                              for cr, rho in ((20, 0.5), (6, 0.5), (20, 0.2))},
            "variance_k20_over_best": variance_at_budget(20, 20, 0.5) / variance_at_budget(best_variants(20, 0.5), 20, 0.5),
            "variance_at_k": {k: variance_at_budget(k, 20, 0.5) for k in (4, 5, 20)},
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
