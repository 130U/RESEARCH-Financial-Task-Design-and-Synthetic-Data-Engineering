"""Sections 1-2: structural validity under linear accounting identities.

1. The W2 / total-variation counterexample: off-manifold noise of size sigma makes W2 -> 0
   while no generated record satisfies the identities exactly.
2. Rejection sampling: acceptance alpha_m(eps) = (2 Phi(eps) - 1)^m for m standardized residuals.
3. Covariance-weighted (Mahalanobis) projection onto {x : A x = b}: every draw satisfies the
   identities to rounding error and the draws have the exact conditional Gaussian law.

The reference run of Section 2 is defined here: twelve balance-sheet and income fields under
four identities, 200,000 draws. Normal draws come from the Box-Muller transform of
random.random(), whose sequence is fixed across Python versions for a given seed, so the
run gives the same figures everywhere.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

PHI = NormalDist().cdf

# ------------------------------------------------------------------ small linear algebra


def matmul(a, b):
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def transpose(a):
    return [list(r) for r in zip(*a)]


def inverse(a):
    n = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        piv = m[c][c]
        m[c] = [v / piv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [row[n:] for row in m]


def cholesky(a):
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            low[i][j] = math.sqrt(s) if i == j else s / low[j][j]
    return low


def frobenius(a):
    return math.sqrt(sum(v * v for row in a for v in row))


# ------------------------------------------------------------------ 1. W2 vs TV


def w2_tv_counterexample(sigmas=(1e-1, 1e-3, 1e-6), m=4, draws=10_000, seed=1):
    """For X on M and X_sigma = X + sigma N eps, the coupling bound W2 <= sigma sqrt(m) and the
    fraction of generated records lying exactly on M (Q_sigma(M), which is 0)."""
    rng = random.Random(seed)
    out = []
    for s in sigmas:
        on_manifold = 0
        sq = 0.0
        for _ in range(draws):
            eps = [rng.gauss(0.0, 1.0) for _ in range(m)]
            normal_offset = [s * e for e in eps]          # coordinates in the normal space
            sq += sum(v * v for v in normal_offset)
            on_manifold += all(v == 0.0 for v in normal_offset)
        out.append({"sigma": s, "coupling_rms": math.sqrt(sq / draws), "bound": s * math.sqrt(m),
                    "share_exactly_on_M": on_manifold / draws})
    return out


# ------------------------------------------------------------------ 2. rejection sampling


def rejection_table(eps=0.05, ms=(1, 2, 4)):
    rows = []
    for m in ms:
        alpha = (2 * PHI(eps) - 1) ** m
        rows.append({"m": m, "acceptance": alpha, "expected_draws": 1 / alpha,
                     "small_eps_approx": (math.sqrt(2 / math.pi) * eps) ** m})
    return rows


def rejection_by_sampling(eps=0.05, m=2, draws=2_000_000, seed=2):
    rng = random.Random(seed)
    accepted = 0
    for _ in range(draws):
        if all(abs(rng.gauss(0.0, 1.0)) <= eps for _ in range(m)):
            accepted += 1
    return accepted / draws


# ------------------------------------------------------------------ 3. the reference run
#
# Twelve balance-sheet and income fields, in units of 100 million:
#   0 cash  1 receivables  2 inventory  3 PP&E  4 intangibles  5 total assets
#   6 payables  7 short-term debt  8 long-term debt  9 total liabilities  10 equity  11 revenue
MEAN = [1.2, 1.6, 1.1, 4.0, 0.9, 8.8, 1.3, 0.9, 2.6, 4.8, 4.0, 9.7]
SCALE = [0.4, 0.5, 0.4, 1.2, 0.4, 2.0, 0.4, 0.3, 0.9, 1.3, 1.2, 2.5]
# Two-factor loadings (size, leverage) give a realistic positive-definite correlation.
LOAD = [(0.7, -0.1), (0.8, 0.0), (0.7, 0.1), (0.8, 0.3), (0.5, 0.2), (0.9, 0.2),
        (0.7, 0.2), (0.5, 0.6), (0.6, 0.6), (0.7, 0.6), (0.7, -0.5), (0.85, 0.0)]
DSO_DAYS = 60
A = [
    [1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0],             # assets add up
    [0, 0, 0, 0, 0, 0, 1, 1, 1, -1, 0, 0],             # liabilities add up
    [0, 0, 0, 0, 0, 1, 0, 0, 0, -1, -1, 0],            # assets = liabilities + equity
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -DSO_DAYS / 365],  # receivables = 60 days of revenue
]
B_VEC = [0.0, 0.0, 0.0, 0.0]


def covariance():
    d = len(MEAN)
    corr = [[sum(LOAD[i][k] * LOAD[j][k] for k in range(2)) for j in range(d)] for i in range(d)]
    for i in range(d):
        corr[i][i] = 1.0
    return [[SCALE[i] * SCALE[j] * corr[i][j] for j in range(d)] for i in range(d)]


def normal_stream(rng):
    """Standard normal draws by the Box-Muller transform of rng.random()."""
    while True:
        u1 = 1.0 - rng.random()
        u2 = rng.random()
        radius = math.sqrt(-2.0 * math.log(u1))
        yield radius * math.cos(2.0 * math.pi * u2)
        yield radius * math.sin(2.0 * math.pi * u2)


def reference_run(draws=200_000, seed=3):
    rng = random.Random(seed)
    normal = normal_stream(rng)
    sigma = covariance()
    d, m = len(MEAN), len(A)
    low = cholesky(sigma)
    at = transpose(A)
    k_gain = matmul(matmul(sigma, at), inverse(matmul(matmul(A, sigma), at)))  # Sigma A^T (A Sigma A^T)^-1
    # theoretical conditional law of Z | A Z = b
    ka_sigma = matmul(matmul(k_gain, A), sigma)
    cond_cov = [[sigma[i][j] - ka_sigma[i][j] for j in range(d)] for i in range(d)]
    a_mu = [sum(A[r][j] * MEAN[j] for j in range(d)) for r in range(m)]
    cond_mean = [MEAN[i] + sum(k_gain[i][r] * (B_VEC[r] - a_mu[r]) for r in range(m)) for i in range(d)]

    total = [0.0] * d
    cross = [[0.0] * d for _ in range(d)]
    max_resid = 0.0
    max_abs_x = 0.0
    for _ in range(draws):
        e = [next(normal) for _ in range(d)]
        z = [MEAN[i] + sum(low[i][k] * e[k] for k in range(i + 1)) for i in range(d)]
        gap = [B_VEC[r] - sum(A[r][j] * z[j] for j in range(d)) for r in range(m)]
        x = [z[i] + sum(k_gain[i][r] * gap[r] for r in range(m)) for i in range(d)]
        for r in range(m):
            res = abs(sum(A[r][j] * x[j] for j in range(d)) - B_VEC[r])
            if res > max_resid:
                max_resid = res
        max_abs_x = max(max_abs_x, max(abs(v) for v in x))
        for i in range(d):
            xi = x[i]
            total[i] += xi
            row = cross[i]
            for j in range(i, d):
                row[j] += xi * x[j]
    mean = [t / draws for t in total]
    emp = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            v = (cross[i][j] - draws * mean[i] * mean[j]) / (draws - 1)
            emp[i][j] = emp[j][i] = v
    diff = [[emp[i][j] - cond_cov[i][j] for j in range(d)] for i in range(d)]
    mean_err = max(abs(a - b) for a, b in zip(mean, cond_mean))
    return {
        "draws": draws,
        "fields": d,
        "constraints": m,
        "max_abs_constraint_residual": max_resid,
        "max_abs_field_value": max_abs_x,
        "relative_frobenius_cov_error_percent": 100 * frobenius(diff) / frobenius(cond_cov),
        "max_abs_mean_error": mean_err,
        "conditional_cov_rank_deficit": m,
    }


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps({"rejection": rejection_table(), "w2_tv": w2_tv_counterexample(),
                      "projection": reference_run(int(sys.argv[1]) if len(sys.argv) > 1 else 20_000)},
                     indent=2))
