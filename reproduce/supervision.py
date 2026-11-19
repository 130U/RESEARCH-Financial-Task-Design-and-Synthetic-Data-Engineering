"""Sections 3-5: when supervision goes stale, leaks, or overstates its sample size.

1. Stale labels near a decision boundary: P{d(x + Delta) < 0} = Phi(-gamma / s).
   Checked with a three-dimensional Gaussian perturbation of a linear decision rule.
2. Brownian bridge leakage: conditioning on X_T halves the midpoint variance, an apparent
   29.29% RMSE reduction that comes only from future information.
3. Family clustering: G families of k correlated variants carry Gk / (1 + (k - 1) rho)
   effective observations.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

PHI = NormalDist().cdf


# ------------------------------------------------------------------ 1. stale labels


def stale_label_rates(ratios=(2.0, 1.0, 0.25), draws=1_000_000, seed=4):
    """Linear rule d(x) = w.x - c with margin gamma at x, perturbation Delta ~ N(0, Omega).

    Omega is diagonal with unequal variances so that s^2 = w' Omega w mixes several directions;
    gamma is set to ratio * s. The label flips when w.Delta < -gamma.
    """
    rng = random.Random(seed)
    w = (0.8, -0.5, 0.3)              # e.g. weights on EBITDA, net debt, working capital
    omega = (0.04, 0.09, 0.25)        # perturbation variances on each input
    s = math.sqrt(sum(wi * wi * oi for wi, oi in zip(w, omega)))
    sd = [math.sqrt(o) for o in omega]
    out = []
    for ratio in ratios:
        gamma = ratio * s
        flips = 0
        for _ in range(draws):
            if sum(wi * rng.gauss(0.0, si) for wi, si in zip(w, sd)) < -gamma:
                flips += 1
        out.append({"gamma_over_s": ratio, "exact": PHI(-ratio), "sampled": flips / draws})
    return out


def lipschitz_certificate_holds(trials=100_000, seed=5):
    """|d(x)| > L ||Delta|| implies the sign is unchanged, for d(x) = w.x - c (L = ||w||)."""
    rng = random.Random(seed)
    w = (0.8, -0.5, 0.3)
    lip = math.sqrt(sum(v * v for v in w))
    for _ in range(trials):
        x = [rng.uniform(-2, 2) for _ in w]
        delta = [rng.gauss(0.0, 0.3) for _ in w]
        c = rng.uniform(-1, 1)
        d0 = sum(a * b for a, b in zip(w, x)) - c
        d1 = sum(a * (b + e) for a, b, e in zip(w, x, delta)) - c
        if abs(d0) > lip * math.sqrt(sum(e * e for e in delta)) and (d0 > 0) != (d1 > 0):
            return False
    return True


# ------------------------------------------------------------------ 2. temporal leakage


def bridge_leakage(s=0.0, t=0.5, T=1.0, sigma=1.0, draws=1_000_000, seed=6):
    """Compare the optimal forecast of X_t from F_s with the one that also sees X_T."""
    rng = random.Random(seed)
    sse_hist = sse_leak = 0.0
    for _ in range(draws):
        xs = 0.0
        xt = xs + sigma * math.sqrt(t - s) * rng.gauss(0.0, 1.0)
        xT = xt + sigma * math.sqrt(T - t) * rng.gauss(0.0, 1.0)
        pred_hist = xs                                         # E[X_t | F_s]
        pred_leak = xs + (t - s) / (T - s) * (xT - xs)         # E[X_t | F_s, X_T]
        sse_hist += (xt - pred_hist) ** 2
        sse_leak += (xt - pred_leak) ** 2
    exact_ratio = (T - t) / (T - s)
    return {
        "exact_variance_ratio": exact_ratio,
        "exact_rmse_reduction": 1 - math.sqrt(exact_ratio),
        "sampled_rmse_reduction": 1 - math.sqrt(sse_leak / sse_hist),
        "sampled_var_hist": sse_hist / draws,
        "exact_var_hist": sigma**2 * (t - s),
    }


# ------------------------------------------------------------------ 3. family clustering


def effective_sample_size(families=100, variants=20, rho=0.5):
    nominal = families * variants
    deff = 1 + (variants - 1) * rho
    return {"nominal": nominal, "design_effect": deff, "effective": nominal / deff,
            "se_understatement": math.sqrt(deff)}


def clustering_by_sampling(families=100, variants=20, rho=0.5, reps=20_000, seed=7):
    """Random-effects model Y = a_g + e_gi with Var(a) = rho, Var(e) = 1 - rho.

    Returns the sampled variance of the grand mean divided by the naive i.i.d. variance
    1 / (G k); its expectation is the design effect.
    """
    rng = random.Random(seed)
    sa, se = math.sqrt(rho), math.sqrt(1 - rho)
    n = families * variants
    means = []
    for _ in range(reps):
        total = 0.0
        for _ in range(families):
            a = rng.gauss(0.0, sa)
            # sum of `variants` independent e's equals sqrt(variants) * se * N(0, 1)
            total += variants * a + math.sqrt(variants) * se * rng.gauss(0.0, 1.0)
        means.append(total / n)
    mu = sum(means) / reps
    var = sum((m - mu) ** 2 for m in means) / (reps - 1)
    return var / (1 / n)


if __name__ == "__main__":
    import json

    print(json.dumps({"stale": stale_label_rates(draws=200_000), "bridge": bridge_leakage(draws=200_000),
                      "ess": effective_sample_size(), "ess_sampled": clustering_by_sampling(reps=5_000)}, indent=2))
