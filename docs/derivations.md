# Derivations

Full proofs of the article's results, in its notation. Section numbers (§) refer to the
[article](../manuscript/financial-task-design-and-synthetic-data-engineering.md); every figure quoted here is recomputed in
[results](../reproduce/results.md).

| | Result | Section |
| --- | --- | --- |
| [D1](#d1-distributional-fit-does-not-imply-structural-validity) | Distributional fit does not imply structural validity | §1 |
| [D2](#d2-rejection-sampling-decays-geometrically-in-the-number-of-identities) | Rejection sampling decays geometrically in the number of identities | §2 |
| [D3](#d3-the-covariance-weighted-projection-samples-the-exact-conditional-law) | The covariance-weighted projection samples the exact conditional law | §2 |
| [D4](#d4-the-displacement-is-a-chi-square-monitor) | The displacement is a chi-square monitor | §2 |
| [D5](#d5-a-sufficient-condition-for-keeping-a-reference-decision) | A sufficient condition for keeping a reference decision | §3 |
| [D6](#d6-stale-label-risk-covenants-and-the-recompute-rule) | Stale-label risk, covenants and the recompute rule | §3 |
| [D7](#d7-the-information-in-a-graded-preference) | The information in a graded preference | §3 |
| [D8](#d8-endpoint-conditioning-and-what-it-leaks) | Endpoint conditioning and what it leaks | §4 |
| [D9](#d9-effective-sample-size-of-clustered-variants) | Effective sample size of clustered variants | §5 |
| [D10](#d10-seed-leverage-and-the-optimal-number-of-variants) | Seed leverage and the optimal number of variants | §5 |
| [D11](#d11-coverage-and-how-to-spend-on-it) | Coverage and how to spend on it | §6 |

---

## D1. Distributional fit does not imply structural validity

**Setup.** Valid inputs satisfy $`Ax = b`$ with $`A \in ℝ^{m\times d}`$ of full row rank, so the valid law
$`P`$ lives on the affine subspace $`M`$. $`N \in ℝ^{d\times m}`$ has orthonormal columns spanning the normal
space $`M^\perp = \mathrm{range}(A^\top)`$. The generator outputs $`X_\sigma = X + \sigma N\varepsilon`$ with
$`\varepsilon \sim 𝒩(0, I_m)`$ independent of $`X \sim P`$.

1. **$`W_2 \to 0`$.** $`(X, X_\sigma)`$ is a coupling of $`P`$ and $`Q_\sigma`$, so
   $`W_2(P,Q_\sigma)^2 \le E\|X_\sigma - X\|^2 = \sigma^2 E\|N\varepsilon\|^2 = \sigma^2\mathrm{tr}(N^\top N) = \sigma^2 m`$.
2. **$`Q_\sigma(M) = 0`$.** $`AX_\sigma - b = \sigma AN\varepsilon`$. The columns of $`N`$ span
   $`\mathrm{range}(A^\top)`$, on which $`A`$ is injective, so $`AN`$ is an invertible $`m\times m`$ matrix. Then
   $`AX_\sigma = b`$ if and only if $`\varepsilon = 0`$, an event of probability zero for a continuous Gaussian.
3. **$`\|P - Q_\sigma\|_{\text{TV}} = 1`$.** Take the Borel set $`B = M`$: $`\left\vert P(M) - Q_\sigma(M)\right\vert = 1`$, and total variation
   never exceeds 1. ∎

A generator can be arbitrarily close in every transport metric and still produce no record that balances. Structure
has to be imposed by construction.

## D2. Rejection sampling decays geometrically in the number of identities

If the $`m`$ standardized residuals are independent $`𝒩(0,1)`$, each falls in $`[-\epsilon, \epsilon]`$
with probability $`2\Phi(\epsilon) - 1`$, so all do with probability
$`\alpha_m(\epsilon) = [2\Phi(\epsilon) - 1]^m`$. The number of candidates until the first acceptance is geometric
with mean $`1/\alpha_m`$. As $`\epsilon \to 0^+`$,
$`2\Phi(\epsilon) - 1 = 2\int_0^\epsilon \varphi(u)\,du = \sqrt{2/\pi}\,\epsilon + O(\epsilon^3)`$. ∎ At
$`\epsilon = 0.05`$ the acceptance rate is 3.9878%, 0.1590% and 0.0002529% for $`m = 1, 2, 4`$, and the expected
number of draws per valid input goes from 25.08 to 395,442.59.

## D3. The covariance-weighted projection samples the exact conditional law

**Closed form.** Minimize $`\tfrac12(x - Z)^\top\Sigma^{-1}(x - Z)`$ subject to $`Ax = b`$. Stationarity of the
Lagrangian gives $`\Sigma^{-1}(x - Z) + A^\top\lambda = 0`$, so $`x = Z - \Sigma A^\top\lambda`$. Imposing $`Ax = b`$
gives $`\lambda = -(A\Sigma A^\top)^{-1}(b - AZ)`$, hence

```math
X = Z + K(b - AZ), \qquad K = \Sigma A^\top(A\Sigma A^\top)^{-1}, \qquad AK = I_m . \qquad (D3.1)
```

$`A\Sigma A^\top`$ is invertible because $`\Sigma \succ 0`$ and $`A`$ has full row rank. Since
$`AX = AZ + (b - AZ) = b`$, every draw satisfies the identities exactly.

**Moments.** $`X = (I - KA)Z + Kb`$ is affine in $`Z`$, so it is Gaussian with mean $`\mu + K(b - A\mu)`$ and
covariance $`(I-KA)\Sigma(I-KA)^\top`$. Expanding, with $`KA\Sigma = \Sigma A^\top(A\Sigma A^\top)^{-1}A\Sigma`$
symmetric:

```math
(I-KA)\Sigma(I-KA)^\top = \Sigma - 2KA\Sigma + KA\Sigma A^\top K^\top = \Sigma - \Sigma A^\top(A\Sigma A^\top)^{-1}A\Sigma . \qquad (D3.2)
```

**Why these are the conditional moments.** $`\mathrm{Cov}(X, AZ) = (I-KA)\Sigma A^\top = 0`$. $`X`$ and $`AZ`$
are jointly Gaussian and uncorrelated, so they are independent. On the event $`AZ = b`$ we have $`X = Z`$. The law
of $`Z`$ given $`AZ = b`$ therefore equals the law of $`X`$ given $`AZ = b`$, which by independence is the
unconditional law of $`X`$. ∎

**Why the weighting matters.** The Euclidean projection $`K_I = A^\top(AA^\top)^{-1}`$ also enforces $`AX = b`$, but
$`\mathrm{Cov}(X, AZ) = (I - K_I A)\Sigma A^\top \ne 0`$ unless $`\Sigma A^\top`$ lies in
$`\mathrm{range}(A^\top)`$. It satisfies the accounting and distorts the joint distribution of the fields.

**Where exactness holds.** For a Gaussian copula the argument applies to constraints that are affine in the latent
Gaussian coordinates. Bounds and orderings are handled by bijections into unconstrained coordinates ($`\log`$ for
positive fields, $`t_k = t_{k-1} + e^{\xi_k}`$ for dates), and fields fully determined by others, such as totals, are
computed and never sampled.

## D4. The displacement is a chi-square monitor

The gap $`b - AZ`$ is Gaussian with mean $`b - A\mu`$ and covariance $`A\Sigma A^\top`$. Whitening by
$`(A\Sigma A^\top)^{-1/2}`$ gives $`m`$ independent unit-variance normals with mean vector of squared length
$`\delta`$, so

```math
T = (b - AZ)^\top(A\Sigma A^\top)^{-1}(b - AZ) \sim \chi^2_m(\delta), \qquad
\delta = (b - A\mu)^\top(A\Sigma A^\top)^{-1}(b - A\mu), \qquad (D4.1)
```

with $`E[T] = m + \delta`$ and $`\mathrm{Var}(T) = 2(m + 2\delta)`$. ∎ $`T`$ is the squared Mahalanobis distance
the projection moves the draw: $`X - Z = K(b - AZ)`$ and
$`K^\top\Sigma^{-1}K = (A\Sigma A^\top)^{-1}`$, so $`(X - Z)^\top\Sigma^{-1}(X - Z) = T`$. It is computed in the
correction step at no further cost.

**Drift signal.** Over a batch of $`n`$ draws the mean of $`T`$ shifts by $`\delta`$ with standard error
$`\sqrt{2(m + 2\delta)/n}`$. On the reference run the mean is 4.01 against $`m = 4`$. A 10% drift in the upstream
mean of receivables gives $`\delta = 0.197`$, an expected mean of 4.20, and a signal of 2.10 standard errors at
$`n = 1{,}000`$ and 6.65 at $`n = 10{,}000`$.

## D5. A sufficient condition for keeping a reference decision

If $`d`$ is $`L`$-Lipschitz on a neighborhood containing $`x`$ and $`x + \Delta`$, then
$`d(x+\Delta) \ge d(x) - L\|\Delta\|`$, and $`d(x) > L\|\Delta\|`$ gives $`d(x+\Delta) > 0`$. The negative case is
symmetric. ∎ The certificate covers the sign of the decision. The reference calculation, the evidence cited for it
and the preference between two responses can each change while the sign is fixed, so each is checked separately.

## D6. Stale-label risk, covenants and the recompute rule

**Linear rule.** With $`d(x) = w^\top x - c`$, margin $`\gamma = d(x) > 0`$ and
$`\Delta \sim 𝒩(0, \Omega)`$: $`w^\top\Delta \sim 𝒩(0, s^2)`$ with $`s^2 = w^\top\Omega w`$. The
label flips if and only if $`w^\top\Delta < -\gamma`$, so $`\Pr(\text{flip}) = \Phi(-\gamma/s)`$. ∎ The rate is 2.28%
at $`\gamma/s = 2`$, 15.87% at 1 and 40.13% at 0.25.

**Covenants.** An interest-coverage test $`E/I \ge c`$ with $`I > 0`$ is the same event as $`E - cI \ge 0`$, a
half-space in the accounting levels with $`w = (1, -c)`$; a leverage test $`D/E \le c`$ with $`E > 0`$ is
$`cE - D \ge 0`$. The rule is linear in the levels, so $`\Phi(-\gamma/s)`$ is the exact flip rate under Gaussian
perturbations of the levels, with no linearization. When several covenants apply, the case flips if any one does,
and by the union bound the rate is at most $`\sum_k \Phi(-\gamma_k/s_k)`$: 6.68% + 2.28% = 8.96% for two covenants
at 1.5 and 2.

**Recompute rule.** $`\Phi(-\gamma/s)`$ is decreasing in $`\gamma/s`$. Inheriting the label exactly when
$`\gamma/s \ge z_{1-\alpha} = \Phi^{-1}(1-\alpha)`$ therefore keeps the stale-label probability of every inherited
variant at or below $`\Phi(-z_{1-\alpha}) = \alpha`$. ∎ The thresholds are 1.645 at 5% and 2.326 at 1%. Diagnostic
cases are built near the boundary on purpose, so they fall below any such threshold and are always recomputed.

## D7. The information in a graded preference

Model the expert's judgment as an ordered response to the latent quality gap: $`U = g + \eta`$ with
$`\eta \sim 𝒩(0,1)`$, and the response is level $`k`$ when $`c_{k-1} < U \le c_k`$, with
$`c_0 = -\infty`$ and $`c_K = +\infty`$. Then $`p_k(g) = \Phi(c_k - g) - \Phi(c_{k-1} - g)`$ and
$`\partial_g p_k = \varphi(c_{k-1} - g) - \varphi(c_k - g)`$. For a multinomial observation the Fisher information
about $`g`$ is

```math
I(g) = \sum_k \frac{(\partial_g p_k(g))^2}{p_k(g)} . \qquad (D7.1)
```

A continuous report of $`U`$ would carry $`I = 1`$, which is the ceiling. A binary judgment has one cutpoint at
zero and $`I(g) = \varphi(g)^2/[\Phi(g)(1 - \Phi(g))]`$, equal to $`2/\pi = 0.637`$ at a tie. With cutpoints at
$`\pm 0.5`$ and $`\pm 1.5`$ the five-level judgment carries 0.912 at a tie, 0.902 at $`g = 0.5`$ and 0.863 at
$`g = 1`$, against 0.637, 0.581 and 0.439 for the binary judgment. ∎ The ratio rises from 1.43 to 1.97 because the
binary response saturates as the gap grows while the outer cutpoints keep resolving it. Maximizing $`I(0)`$ over
symmetric cutpoints gives $`\pm 0.38`$ and $`\pm 1.24`$, where the information is 0.920.

## D8. Endpoint conditioning and what it leaks

**Brownian paths.** For $`X_u = \sigma W_u`$ and $`s < t < T`$, given $`ℱ_s`$ the pair $`(X_t, X_T)`$ is
Gaussian with $`\mathrm{Var}(X_t) = \sigma^2(t-s)`$, $`\mathrm{Var}(X_T) = \sigma^2(T-s)`$ and
$`\mathrm{Cov}(X_t, X_T) = \sigma^2(t-s)`$. Gaussian conditioning gives

```math
\mathrm{Var}(X_t \mid ℱ_s, X_T) = \sigma^2(t-s) - \frac{\sigma^4(t-s)^2}{\sigma^2(T-s)} = \sigma^2(t-s)\,\frac{T-t}{T-s} . \qquad (D8.1)
```

At the midpoint the ratio is $`1/2`$, so the RMSE falls by the factor $`1/\sqrt2`$, an apparent improvement of
29.29%. ∎

**Mean-reverting paths.** For $`dX = -\kappa X\,du + \sigma\,dW`$ and $`t - s = T - t = h`$, write
$`v(u) = \sigma^2(1 - e^{-2\kappa u})/(2\kappa)`$. Given $`ℱ_s`$, $`\mathrm{Var}(X_t) = v(h)`$,
$`\mathrm{Var}(X_T) = v(2h)`$ and $`\mathrm{Cov}(X_t, X_T) = e^{-\kappa h}v(h)`$. Then

```math
\frac{\mathrm{Var}(X_t \mid ℱ_s, X_T)}{\mathrm{Var}(X_t \mid ℱ_s)}
= 1 - \frac{e^{-2\kappa h}\,v(h)}{v(2h)}
= 1 - \frac{e^{-2\kappa h}(1 - e^{-2\kappa h})}{1 - e^{-4\kappa h}}
= \frac{1}{1 + e^{-2\kappa h}}, \qquad (D8.2)
```

using $`1 - e^{-4\kappa h} = (1 - e^{-2\kappa h})(1 + e^{-2\kappa h})`$. ∎ The apparent RMSE reduction is
$`1 - (1 + e^{-2\kappa h})^{-1/2}`$: 29.29% as $`\kappa h \to 0`$, then 21.10%, 14.50%, 6.15% and 0.90% at
$`\kappa h = 0.25, 0.5, 1, 2`$. The code checks the closed form against direct conditioning of the covariance
matrix.

The conditional mean is the bridge interpolant. It is the right object when the task discloses $`X_T`$
(interpolation, backcasting) and it is leakage when the task is a forecast from $`ℱ_s`$.

## D9. Effective sample size of clustered variants

For $`G`$ independent families of $`k`$ variants with common variance $`\sigma_Y^2`$ and within-family correlation
$`\rho`$, $`\mathrm{Var}(\sum Y) = G[k\sigma_Y^2 + k(k-1)\rho\sigma_Y^2]`$. Dividing by $`(Gk)^2`$:

```math
\mathrm{Var}(\bar Y) = \frac{\sigma_Y^2}{Gk}[1 + (k-1)\rho], \qquad N_{\text{eff}} = \frac{Gk}{1 + (k-1)\rho} . \quad ∎ \qquad (D9.1)
```

For $`G = 100`$, $`k = 20`$, $`\rho = 0.5`$ the design effect is 10.5, $`N_{\text{eff}}`$ is 190, and naive standard
errors are too small by $`\sqrt{10.5} = 3.24`$.

**Unequal family sizes.** With family sizes $`k_g`$ the same computation gives a design effect of
$`1 + (\bar k_w - 1)\rho`$, where $`\bar k_w = \sum_g k_g^2 / \sum_g k_g`$ is the size-weighted mean family size. A
few prolific templates cost more effective sample than their share of records suggests.

## D10. Seed leverage and the optimal number of variants

**Leverage.** A seed whose reference is wrong with probability $`\varepsilon`$ and that yields $`k`$ variants
carries $`k\varepsilon`$ wrong records in expectation. If each independent re-solve catches a wrong reference with
probability $`d`$, the reference survives $`r`$ re-solves undetected with probability $`(1-d)^r`$, leaving
$`k\varepsilon(1-d)^r`$. ∎ With $`k = 20`$, $`\varepsilon = 0.15`$ and $`d = 0.8`$: 3.0, 0.6 and 0.12 for
$`r = 0, 1, 2`$.

**Optimal variants per seed.** A family costs $`c_G`$ and each variant $`c_v`$, so a budget $`B`$ buys
$`G = B/(c_G + kc_v)`$ families. By D9 the variance of the mean is proportional to

```math
f(k) = \frac{(c_G + kc_v)\,[1 + (k-1)\rho]}{k} = \frac{c_G(1-\rho)}{k} + c_G\rho + c_v(1-\rho) + c_v\rho\,k . \qquad (D10.1)
```

$`f`$ is convex in $`k`$, and $`f'(k) = -c_G(1-\rho)/k^2 + c_v\rho = 0`$ gives

```math
k^* = \sqrt{\frac{c_G}{c_v}\cdot\frac{1-\rho}{\rho}} . \quad ∎ \qquad (D10.2)
```

With $`c_G/c_v = 20`$ and $`\rho = 0.5`$, $`k^* = 4.5`$. In units of $`c_v`$, $`f(20) = 21`$ and
$`f(4) = f(5) = 15`$: twenty variants per family carry 40% more variance than four or five at the same budget, and
moving to the optimum cuts the variance by 29%.

## D11. Coverage and how to spend on it

$`\text{Coverage} = \sum_j w_j\min(1, n_j/n_j^*)`$ is concave, nondecreasing and piecewise linear in the counts.
Each valid family in cell $`j`$ adds $`w_j/n_j^*`$ until the cell is full, and nothing after. If one more valid
family in cell $`j`$ costs $`c_j`$ in expectation (generation, review, rework and adjudication, divided by the
cell's yield of valid families), maximizing coverage under a total budget is a fractional knapsack problem: fill
cells in decreasing order of $`w_j/(n_j^* c_j)`$, each up to its target. Greedy filling is optimal for the
continuous relaxation, and with integer counts only the last cell is partly filled, so it loses at most one
family's contribution, $`\max_j w_j/n_j^*`$. ∎ More variants of an easy template stop counting once its cell is
full, and a cell with a low yield of valid families is expensive even when its raw generation is cheap.
