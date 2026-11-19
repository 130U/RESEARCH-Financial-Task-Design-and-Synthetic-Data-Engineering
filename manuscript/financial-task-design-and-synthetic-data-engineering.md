# Financial Task Design and Synthetic Data Engineering

**Duke University × Leading Foundation Model Developer**  
Letao (Theodore) Ouyang · Researcher · Durham, North Carolina; Palo Alto, California · September 2023 – September 2026

**Confidentiality and disclosure.** I conducted this project through a Duke University collaboration with a leading foundation model developer from September 2023 to September 2026. The partner's name and proprietary system identifiers are withheld under a nondisclosure agreement. The project title, my role, locations, dates, system architecture and mathematical formulations are cleared for public disclosure. Production tasks and operating figures are withheld; the numerical results are closed-form values, a reference run of the constrained generator, and worked examples with their inputs stated.

## Summary

When a synthetic financial case changes one number, its reference answer, the evidence behind that answer and the expert's preference can all go stale together. Synthetic data is judged on fidelity, utility and privacy, and none of the three asks whether the label is still right; at a margin of a quarter of the perturbation scale, an inherited label is wrong 40% of the time. I built the rules that decide when each object has to be recomputed, and the generators that keep accounting identities exact as the data scales. I created Batch Zero to test those rules on domain experts before a task family scaled, and the partner company subsequently adopted the protocol across other post-training programs.

## Problem and Approach

In late 2023, frontier foundation models could reproduce corporate finance definitions and still made unpredictable errors in financial workflows, where a small numerical change or an accounting footnote could reverse an investment recommendation. Adding synthetic data helps only if the supervision moves with the scenario. I built the data strategy around that requirement, connecting task specifications, synthesis methods and expert calibration to the capability objectives of post-training research teams and to two problems:
1. **Diagnosing reasoning limits through task design.** I designed prompts around subtle business failure modes, including covenant breaches, off-balance-sheet adjustments and valuation differences close to a decision boundary, because simple prompts may give little basis for telling candidate responses apart. Paired with the five-level preference protocol, these prompts carry a domain judgment such as *"Response A is better because it isolates the working-capital constraint and cites Footnote 4"* into structured preference supervision for post-training.
2. **Matching the generator to the financial structure.** Synthetic financial data has to satisfy structural constraints, stay consistent with reference judgments and respect the task's information boundaries. A corporate table that violates Total Assets $`=`$ Total Liabilities $`+`$ Equity, even by a fraction of a percentage point, introduces a structural error. An interest-rate path that exposes a future terminal rate to a historical forecasting task introduces temporal leakage.

I treat the result as a fourth axis of synthetic-data quality, *supervision validity*: the share of generated cases whose scenario, reference conclusion, evidence and preference are all still correct for the case as generated. The system that maintains it works like an incremental build. The four objects form a dependency graph, a change to the scenario marks everything downstream as stale, and a certificate computed from the decision margin says which labels may be inherited without rework. Identities hold by construction in the generator, each temporal task states which information it may see, and Batch Zero checks before scale that experts who did not write a specification can execute it. Sections 1 through 6 give the conditions, analytical results and designs for each part.

The team averaged roughly 500 completed financial tasks a week. Each task combined synthetic financial data with a reference solution and expert review. A typical financial task took about three specialist-hours across drafting, three reviews and revision.

## Key Contributions

* **I framed the scaling of financial training data as a problem of keeping its supervision valid.** When a financial scenario changes, its reference conclusion, supporting evidence and expert preference can all stop being correct together. From late 2023 I treated those four objects and the dependencies among them as a single design problem, and I refined the framework through the programs that followed. For a frontier foundation model program I specified the capability taxonomy, primary evidence standards, input schemas, decisive variables and acceptance criteria. The diagnostic cases targeted high-stakes business decision boundaries where foundation models and practitioners repeatedly make errors, and they translated doctoral or CFA-level domain judgment into testable task requirements.
* **I tied reference recomputation to the decision margin of each synthetic variant.** Margin conditions certify when the decision label remains unchanged; reference calculations, supporting evidence and pairwise preferences are checked separately. Gaussian margin-to-noise analysis ($`\gamma/s`$) measures the label-flip risk, and for covenant tests, which are ratios of accounting levels, the closed form is exact. Variants below a margin-to-noise threshold are recomputed and the rest inherit their labels with a stated error bound. When a perturbation approaches or crosses a decision boundary, the workflow recomputes the reference solution and collects new five-level preference judgments on fresh model reasoning.
* **I designed a five-level pairwise preference protocol for RLHF and DPO supervision.** It records why an expert prefers one response to another. Each judgment has to cite verifiable evidence, the relevant accounting rules and the decisive variables, so the preference reaches post-training with its reasons attached and helps train models to assess whether a financial judgment has sufficient support. One graded judgment carries the information of 1.4 to 2 binary judgments about the quality gap between two responses.
* **I created Batch Zero to establish and refine these standards, and the partner company subsequently adopted the protocol across other post-training programs.** Starting from the capability objectives and data requirements of post-training research teams, I established a protocol for independent execution and expert calibration that a task family goes through before it enters large-scale data construction. Doctoral-level domain experts independently construct tasks, generate synthetic variants, solve reference cases and assign five-level pairwise preference scores. The disagreements and failure modes that appear in execution inform revisions to task specifications, boundary cases and **Batch One** data acceptance criteria. An error caught at the seed is caught once for every variant the seed would have produced.
* **I developed financial synthetic data methods for four settings.** For each task I selected a mathematical generator and adapted it to the task's structural invariants, data types and permitted information:
  * *Cross-sectional corporate fundamentals.* I used Gaussian and vine copulas with conditional sampling for company size, operating history and financial ratios. These methods preserve joint dependencies across columns and avoid the unrealistic combinations that sampling each column independently produces.
  * *Accounting identities, ratio bounds and chronological order.* Constrained sampling, variable transformations and covariance-weighted affine projections enforce balance-sheet and cash-flow identities, ratio ranges and date ordering. On a reference run of 200,000 twelve-dimensional draws under four identities, the maximum constraint residual was **$`3.0 \times 10^{-15}`$**; rejection sampling under four constraints at a tolerance of 0.05 standard deviations accepts $`0.0002529\%`$ of candidates. The distance each draw is moved is a chi-square statistic that monitors the upstream generator at no cost.
  * *Multimodal records with numerical and categorical fields.* Deep tabular generators model the multimodal empirical distributions, and business-rule checks are mandatory after generation.
  * *Continuous-time price and interest-rate paths.* SDEs, neural SDEs, Brownian bridges and stochastic interpolation generate paths for temporal financial tasks, and each method is matched to the task's permitted filtration ($`ℱ_s`$) to keep future information out of evaluation.
* **I quantified stale-label errors, temporal leakage and task-family clustering in closed form.** When labels are not recomputed, perturbations near a decision boundary produce **40.13%** inherited-label error at $`\gamma/s = 0.25`$. Undisclosed Brownian bridge endpoint conditioning creates an apparent **29.29% reduction in RMSE** at the interval midpoint through future-information leakage, and for mean-reverting rates the leak shrinks to $`1 - (1 + e^{-2\kappa h})^{-1/2}`$. With within-family correlation $`\rho = 0.5`$, 2,000 nominal synthetic variants from 100 task families amount to about **190** effective independent observations, and the same budget spent at the optimal number of variants per family has 29% less variance.

## Manuscript

### 1. Distributional fit and task validity

Modern tabular generators address different distributional problems. [CTGAN](https://arxiv.org/abs/1907.00503) handles mixed discrete and continuous data, multimodality and class imbalance through mode-specific normalization and conditional vectors, and [TabDDPM](https://proceedings.mlr.press/v202/kotelnikov23a.html) runs Gaussian diffusion on numerical fields and multinomial diffusion on categorical ones. Both learn empirical distributions effectively. A financial task can match its distribution and still be invalid, because part of what makes it valid is structural. Accounting identities have to hold exactly, fiscal periods have to run in chronological order, covenant applicability conditions have to be respected, and the numerical inputs have to agree logically with the reference judgment. Global distributional similarity metrics enforce none of this.

A simple construction shows how far apart the two can drift. Suppose valid financial inputs $`x \in ℝ^d`$ have to satisfy $`m`$ linear accounting or ratio identities $`Ax = b`$, where $`A \in ℝ^{m \times d}`$ has full row rank, so that the valid population distribution $`P`$ is supported entirely on the affine subspace $`M = \{x \in ℝ^d : Ax = b\}`$. Assume this distribution has finite second moments, and let the columns of $`N \in ℝ^{d \times m}`$ form an orthonormal basis for the normal space orthogonal to $`M`$. Now take a generator that captures $`P`$ along the manifold but adds a small amount of independent noise in the normal directions: $`X_\sigma = X + \sigma N \varepsilon`$, where $`\varepsilon \sim 𝒩(0, I_m)`$ is independent of $`X \sim P`$. Writing $`Q_\sigma`$ for the generated distribution, I obtain, for every $`\sigma > 0`$,

```math
W_2(P, Q_\sigma) \le \sqrt{E\|X_\sigma - X\|^2} = \sigma \sqrt{m} \longrightarrow 0 \quad (\text{as } \sigma \to 0^+) \qquad (1.1)
```

```math
P(M) = 1, \qquad Q_\sigma(M) = 0 \qquad (1.2)
```

```math
\|P - Q_\sigma\|_{\text{TV}} := \sup_B \left\vert P(B) - Q_\sigma(B)\right\vert = 1 \qquad (1.3)
```

The first line comes from coupling $`(X, X_\sigma)`$ on the same probability space. The second holds because $`AN`$ is invertible and a continuous $`m`$-dimensional Gaussian vector equals zero with probability zero, and taking the Borel set $`B = M`$ then gives the third. So the 2-Wasserstein distance $`W_2(P, Q_\sigma)`$ can tend to zero while the acceptance rate under exact business constraints stays at zero ($`Q_\sigma(M) = 0`$) and the total variation distance stays at its maximum value of $`1`$. With a numerical tolerance the acceptance rate depends on the ratio of tolerance to off-manifold noise, and Section 2 shows that at a fixed ratio it falls geometrically with each added constraint.

The construction is one instance of a general gap. Synthetic data is evaluated on three axes, fidelity to the source distribution, utility for a downstream model and privacy, and the sample-level refinements of those metrics still score the inputs ([Alaa et al.](https://proceedings.mlr.press/v162/alaa22a.html)). None of the three has a place for the question a training team has to answer about supervision: is the label attached to this generated case still the right label? Data augmentation has the same blind spot from the other side. Counterfactual augmentation and contrast sets showed that a minimal edit can flip the gold answer, and they restore validity by having a person relabel every edited example ([Kaushik et al.](https://arxiv.org/abs/1909.12434); [Gardner et al.](https://aclanthology.org/2020.findings-emnlp.117/)), which does not scale to generated variants. I add the missing axis, supervision validity, and the remaining sections build the machinery that keeps it high at scale.

The counterexample also sets how I divided responsibilities in the synthetic data architecture ([SDV constraint-augmented generation](https://docs.sdv.dev/sdv/modeling/constraint-augmented-generation-cag)):
- Copula models and conditional sampling capture cross-sectional rank and tail dependencies among continuous corporate indicators, including company size, operating history, leverage and profit margins. They preserve the dependencies that independent column generation would discard.
- [CTGAN](https://arxiv.org/abs/1907.00503) and [TabDDPM](https://proceedings.mlr.press/v202/kotelnikov23a.html) model mixed-type and multimodal empirical distributions, and explicit business-rule checks follow them.
- Deterministic variable transformations and manifold projections enforce accounting equalities, ratio bounds and chronological order.

### 2. Preserving the conditional distribution under business constraints

Rejection sampling generates unconstrained candidates and discards those that violate the constraints, and as several financial constraints tighten its computational cost becomes prohibitive. Suppose a candidate draw produces $`m`$ standardized constraint residuals that behave as independent standard normal variables, each of which has to lie within $`[-\epsilon, \epsilon]`$. The acceptance probability and the expected number of candidate draws per valid input are

```math
\alpha_m(\epsilon) = [2\Phi(\epsilon) - 1]^m \sim \left( \sqrt{\frac{2}{\pi}} \epsilon \right)^m \quad (\epsilon \to 0^+), \qquad E[N_{\text{attempts}}] = \frac{1}{\alpha_m(\epsilon)} \qquad (2.1)
```

so each additional accounting or ratio constraint multiplies the expected candidate count. With a tolerance of $`\epsilon = 0.05`$ residual standard deviations, the exact analytical values are:

**Table 1. Acceptance rate of rejection sampling under simultaneous constraints.**

| Simultaneous constraints ($`m`$) | Rejection-sampling acceptance rate $`\alpha_m(\epsilon)`$ | Expected candidate draws per valid input |
| --- | --- | --- |
| **1** | 3.9878% | 25.08 |
| **2** | 0.1590% | 628.84 |
| **4** | **0.0002529%** | **395,442.59** |

The alternative is to condition instead of reject: draw once, then correct the draw onto the constraint. In the linear Gaussian setting the covariance-weighted correction returns a sample from the exact conditional distribution. For a Gaussian copula the result holds for constraints that are affine in the latent Gaussian coordinates, since nonlinear marginal transformations do not preserve affine constraints in the observed financial variables. Let the initial unconstrained draw be $`Z \sim 𝒩(\mu, \Sigma)`$, with $`\Sigma \succ 0`$. I apply a covariance-weighted Mahalanobis projection:

```math
X = \arg\min_{x : Ax = b} \frac{1}{2}(x - Z)^\top \Sigma^{-1}(x - Z) = Z + K(b - AZ), \qquad K = \Sigma A^\top (A \Sigma A^\top)^{-1} \qquad (2.2)
```

The first-order Lagrangian stationarity condition is $`\Sigma^{-1}(X - Z) + A^\top \lambda = 0`$, and substituting $`AX = b`$ and eliminating the multiplier $`\lambda`$ gives the closed-form update. Since $`AK = I_m`$, every draw satisfies $`AX = b`$ exactly. Its mean and covariance are

```math
E[X] = \mu + K(b - A\mu) \qquad (2.3)
```

```math
\mathrm{Cov}(X) = (I - KA)\Sigma(I - KA)^\top = \Sigma - \Sigma A^\top (A \Sigma A^\top)^{-1} A\Sigma \qquad (2.4)
```

Affine transformations preserve Gaussianity, so these moments identify the exact regular conditional Gaussian distribution of $`Z \mid AZ = b`$ ([Majumdar and Majumdar](https://arxiv.org/abs/1710.09285)). The update has a long lineage. It is the sampler for a normal distribution truncated on hyperplanes ([Cong, Chen and Zhou](https://doi.org/10.1214/17-BA1052)) and Matheron's rule for pathwise conditioning on a noise-free linear observation ([Wilson et al.](https://arxiv.org/abs/2011.04026)). What I did with it is put accounting identities in the place of the hyperplanes and state where exactness holds, in the latent coordinates of the copula. That also marks the division of labor with constraint layers for deep tabular generators, which enforce linear inequalities in any generator and make no statement about the law of the corrected sample ([Stoian et al.](https://arxiv.org/abs/2402.04823)). Inequalities such as ratio ranges go to transformations and layers; identities go to the projection, which returns the right distribution as well as a feasible point.

On a reference run of the constrained generator, **200,000 twelve-dimensional balance-sheet and income vectors subject to 4 simultaneous linear identities** (assets sum to total assets, liabilities sum to total liabilities, assets equal liabilities plus equity, and receivables stand at 60 days of revenue), the maximum absolute constraint residual was **$`3.0 \times 10^{-15}`$**, and the sample covariance matched the theoretical conditional covariance to **0.359%** in relative Frobenius norm. Every draw is used; the expected 395,442.59 candidate draws per valid input in Table 1 are what rejection would have spent. Fields that accounting definitions fully determine are computed analytically. Range bounds and date ordering are enforced in the latent generation space when they admit bijective transformations, such as logarithmic transforms or positive increments $`t_{k} = t_{k-1} + \exp(\xi_k)`$.

**The correction is also a monitor.** The squared Mahalanobis distance by which the projection moves a draw has a known law:

```math
T = (b - AZ)^\top (A \Sigma A^\top)^{-1} (b - AZ) \ \sim\ \chi^2_m(\delta), \qquad \delta = (b - A\mu)^\top (A \Sigma A^\top)^{-1} (b - A\mu) \qquad (2.5)
```

When the upstream generator is consistent with the identities, $`\delta = 0`$ and $`T`$ averages $`m`$. When the upstream model drifts, $`\delta`$ grows and the average of $`T`$ rises with it. Every batch therefore tests the fit of its own generator while it is being corrected. On the reference run the average of $`T`$ was 4.01 against $`m = 4`$. A 10% drift in the upstream mean of receivables raises $`\delta`$ to 0.197 and the average to 4.20, which is two standard errors in a batch of 1,000 draws and more than six in a batch of 10,000.

### 3. Using decision margins to determine when reference judgments need recomputation

A synthetic variant of a financial reference case often keeps the business narrative and changes decisive financial figures, covenant thresholds or macroeconomic inputs. If the original reference answer stays in place after those values change, systematic label errors enter post-training, even when all accounting constraints hold. [FinQA](https://aclanthology.org/2021.emnlp-main.300/) pairs financial questions with executable reasoning programs, and synthetic preference data similarly require a maintained link between inputs, reference conclusions and the expert's explanation of a preference.

Neighboring work links data generation to feedback in other ways. [WizardMath](https://arxiv.org/pdf/2308.09583v1) couples the evolution of mathematical instructions with reinforcement feedback, and [FinGPT](https://arxiv.org/pdf/2307.10485v1) fine-tunes financial language models on feedback from the market. Templated reasoning sets recompute an answer from the template's variables for every instance ([GSM-Symbolic](https://arxiv.org/abs/2410.05229)), which works when the answer is one formula and leaves evidence and preference untouched. I make the dependencies among a financial scenario, its reference conclusion, the evidence behind that conclusion and the expert preference built on it the object of design, so that a change to the scenario reaches each of them, and only the ones it has to.

Consider a decision rule that chooses between two corporate actions, and let $`d(x)`$ denote the net valuation or metric advantage of action A over action B, so that the correct reference decision is $`\mathrm{sign}\,d(x)`$. If $`d`$ is $`L`$-Lipschitz on a feasible neighborhood, a sufficient condition for retaining the original reference decision is that the perturbation $`\Delta`$ stays strictly within the decision margin:

```math
\left\vert d(x + \Delta) - d(x)\right\vert \le L\|\Delta\|, \qquad \left\vert d(x)\right\vert > L\|\Delta\| \implies \mathrm{sign}\,d(x + \Delta) = \mathrm{sign}\,d(x) \qquad (3.1)
```

This condition certifies the decision label. A loan can stay approved while the calculation behind it, the supporting evidence and the relative quality of two responses all change, so I check reference calculations, evidence and pairwise preferences separately.

My diagnostic tasks focus on contested decision boundaries where models are prone to error, which means $`\left\vert d(x)\right\vert`$ is deliberately kept small. For a linear comparison rule $`d(x) = w^\top x - c`$ with initial margin $`\gamma = d(x) > 0`$, the stale-label error rate has an exact closed form. Let the perturbation within the feasible manifold satisfy $`\Delta \sim 𝒩(0, \Omega)`$, with projected variance $`s^2 = w^\top \Omega w > 0`$. The probability that the correct label flips while the original decision is retained is

```math
\Pr\{d(x + \Delta) < 0\} = \Pr\{w^\top \Delta < -\gamma\} = \Phi\left(-\frac{\gamma}{s}\right) \qquad (3.2)
```

The closed form gives the following inherited-label error rates:
- At a margin-to-noise ratio of $`\gamma / s = 2`$, the error rate is **2.28%**.
- At $`\gamma / s = 1`$, it is **15.87%**.
- Near a diagnostic decision boundary, at $`\gamma / s = 0.25`$, it is **40.13%**.

**The formula is exact for covenants.** The linear rule is the common case in finance. A covenant test is a ratio of accounting levels against a threshold, interest coverage $`E / I \ge c`$ or leverage $`D / E \le c`$, and with a positive denominator it is the half-space $`E - cI \ge 0`$ in the levels themselves. For the covenant family the flip probability $`\Phi(-\gamma / s)`$ is the exact rate under Gaussian perturbations of the levels, with $`w = (1, -c)`$. When several covenants bind at once, the case flips if any of them does, and the rate is bounded by the sum of the individual rates: two covenants at $`\gamma / s`$ of 1.5 and 2 give at most 8.96%.

**The certificate is a cache-hit test.** Treat the four objects as nodes of a dependency graph. A perturbed scenario marks its reference conclusion, evidence and preference as stale, and the margin decides which of them a builder may reuse. With a tolerated error $`\alpha`$ on inherited labels:

```math
\text{recompute the variant when } \frac{\gamma}{s} < z_{1 - \alpha}, \qquad \text{inherit otherwise, with } \Pr(\text{stale label}) \le \alpha \qquad (3.3)
```

The thresholds are 1.645 at $`\alpha = 5\%`$ and 2.326 at $`\alpha = 1\%`$, and a variant at $`\gamma / s = 3`$ inherits its label with an error of 0.13%. Variants far from the boundary cost no expert time, and the diagnostic variants near it, which are the ones the task family exists for, are always recomputed.

I therefore tie case expansion to reference-answer recomputation and renewed five-level pairwise preference grading when perturbations approach or cross a decision boundary. In that event the workflow recomputes the reference solution and samples new candidate model reasoning trajectories. Domain experts then assess the pair on the five-level preference scale and identify which trajectory recognized the decisive variable whose change reversed the decision and supported its judgment with primary evidence. [HelpSteer2-Preference](https://arxiv.org/abs/2410.01257) gives a comparison point from the methods literature.

**What a graded preference is worth.** A five-level judgment takes an expert a few seconds longer than a binary one, on a task that has already cost about three specialist-hours. Model the judgment as an ordered response to the latent quality gap $`g`$ between two responses, measured in units of judgment noise, with cutpoints $`c_1 < \dots < c_4`$, $`c_0 = -\infty`$ and $`c_5 = +\infty`$. The Fisher information one response carries about the gap is

```math
I(g) = \sum_{k} \frac{\big( \partial_g\, p_k(g) \big)^2}{p_k(g)}, \qquad p_k(g) = \Phi(c_k - g) - \Phi(c_{k-1} - g) \qquad (3.4)
```

**Table 2. Fisher information of a binary and a five-level judgment by quality gap.**

| Quality gap $`g`$ | Binary judgment | Five-level judgment (cutpoints at $`\pm 0.5`$, $`\pm 1.5`$) | Ratio |
| --- | --- | --- | --- |
| 0 (a tie) | 0.637 | 0.912 | 1.43 |
| 0.5 | 0.581 | 0.902 | 1.55 |
| 1.0 | 0.439 | 0.863 | 1.97 |

One graded judgment is worth 1.4 to 2 binary judgments, and the advantage grows with the gap, where a binary label saturates. The same expression designs the scale: at a tie the information is largest with cutpoints at $`\pm 0.38`$ and $`\pm 1.24`$, where it reaches 0.920 of the information in a continuous response. The protocol delivers three signals that a training pipeline can consume separately: the direction of the preference, its strength, and the cited evidence and decisive variable.

### 4. Specifying the information available to each temporal task

A cross-sectional company comparison does not require continuous-time stochastic processes. Dynamic tasks such as derivatives valuation, interest-rate term-structure interpolation and liquidity stress tests over multiple periods do require path generators. For those I use SDEs and neural SDEs to model path distributions beyond single-time marginals ([Neural SDEs as Infinite-Dimensional GANs](https://proceedings.mlr.press/v139/kidger21b.html)). Brownian bridges and stochastic interpolation support conditioning on known intermediate observations or endpoints, following the constructions in [Glasserman, *Monte Carlo Methods in Financial Engineering*, Chapter 3](https://doi.org/10.1007/978-0-387-21617-1). A bridge is the conditioning of Section 2 in one dimension: draw the free path, then correct it onto the known endpoint.

That conditioning can introduce temporal leakage into evaluation, and I derive the size of the effect in closed form. Consider scaled Brownian motion $`X_u = \sigma W_u`$, where $`s < t < T`$. Under squared-error loss, the optimal predictor at time $`s`$, given only the historical filtration $`ℱ_s`$, has conditional mean squared error $`\sigma^2(t - s)`$. If the generator conditions paths on the future endpoint $`X_T`$ and inadvertently exposes that conditioning to a forecasting task, Gaussian conditioning gives ([Pitman on the Brownian bridge](https://www.stat.berkeley.edu/~pitman/s205s03/b_bridge.pdf)):

```math
\mathrm{Var}(X_t \mid ℱ_s) = \sigma^2(t - s) \qquad (4.1)
```

```math
\mathrm{Var}(X_t \mid ℱ_s, X_T) = \sigma^2(t - s) - \frac{[\sigma^2(t - s)]^2}{\sigma^2(T - s)} = \sigma^2(t - s)\frac{T - t}{T - s} \qquad (4.2)
```

At the midpoint $`t = (s + T)/2`$, conditioning on $`X_T`$ halves the conditional variance. The resulting **29.29% apparent reduction in RMSE** ($`1 - 1/\sqrt{2}`$) comes entirely from the additional endpoint information in this construction, and whether that counts as a flaw depends on the task. It is temporal leakage if the forecasting task does not permit access to that information. If instead the task gives the model both endpoints $`X_s`$ and $`X_T`$ and asks it to audit an interpolation or backcast an intervening accounting period, Brownian bridge conditioning is consistent with the specified information set.

Interest rates and spreads revert to a mean, and for them the leak has its own closed form. For an Ornstein-Uhlenbeck path with mean-reversion speed $`\kappa`$, at the midpoint of an interval of half-length $`h`$:

```math
\frac{\mathrm{Var}(X_t \mid ℱ_s, X_T)}{\mathrm{Var}(X_t \mid ℱ_s)} = \frac{1}{1 + e^{-2\kappa h}}, \qquad \text{apparent RMSE reduction} = 1 - \big( 1 + e^{-2\kappa h} \big)^{-1/2} \qquad (4.3)
```

**Table 3. Apparent RMSE reduction from endpoint leakage under mean reversion.**

| $`\kappa h`$ | 0 (Brownian) | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
| Apparent RMSE reduction | 29.29% | 21.10% | 14.50% | 6.15% | 0.90% |

Mean reversion discounts the leak, because the endpoint tells less about the middle of a path that forgets. The financial reading is direct: endpoint leakage is most damaging for series close to a unit root, prices and long rates, and nearly harmless for fast-reverting spreads. The information-set rule is therefore enforced with a severity that follows the persistence of the series.

The task specification keeps three information sets apart:
1. **Construction information ($`ℱ_{\text{gen}}`$):** the complete path, endpoint or latent parameters used to generate a scenario.
2. **Permitted predictor filtration ($`ℱ_s`$):** the historical information the task allows the model to observe at decision time $`s`$.
3. **Evaluation and oracle information ($`ℱ_{\text{eval}}`$):** outcomes realized after $`s`$, reserved for scoring or for explicitly disclosed oracle baselines.

### 5. Batch Zero: independent execution and calibration for post-training data strategy

I created **Batch Zero** around the capability objectives and data requirements of post-training research teams. Before anything scales, it brings task design, synthetic data strategy and preference supervision under a common set of acceptance criteria. It tests whether domain experts can independently execute a research specification, reproduce reference solutions and calibrate their preference judgments. The resulting specifications support subsequent data construction.
- **Independent execution:** before a task family enters large-scale data construction (*Batch One*), doctoral-level domain experts independently write seed cases, generate synthetic variants, verify reference solutions and assign five-level pairwise preference scores. Implicit assumptions, missing information and ambiguities in the specification come to light this way.
- **Calibration and feedback to data strategy:** failures, expert disagreements and boundary-case errors inform revisions to task instructions, reference solutions, grading rules and Batch One data acceptance criteria. Task design, execution, revision and recalibration form an iterative process.
- **Organizational adoption:** the partner foundation model company subsequently adopted **Batch Zero** across other post-training programs.

**Why the check belongs at the seed.** Expert-written references are less reliable than their reputation. When FinanceReasoning re-examined four public financial reasoning sets, it updated 15.6% of their questions, between 9.72% and 30.00% per set ([Tang et al.](https://arxiv.org/abs/2506.05828)). A synthetic pipeline multiplies whatever error its seeds carry. If a seed's reference solution is wrong with probability $`\varepsilon`$ and the seed yields $`k`$ variants, a family carries $`k\varepsilon`$ wrong records in expectation, and they are wrong together. If an independent re-solve catches a wrong reference with probability $`d`$, then $`r`$ re-solves leave

```math
k\, \varepsilon\, (1 - d)^r \qquad (5.1)
```

wrong records per family. With $`k = 20`$, $`\varepsilon = 0.15`$ and $`d = 0.8`$, that is 3.0 with no re-solve, 0.6 after one and 0.12 after two. One re-solve at the seed does the work of reviewing twenty variants, and that multiplier is what Batch Zero is built on.

For statistical analysis and evaluation, I treat the task family as the basic unit of effective coverage and uncertainty estimation. Suppose $`G`$ independent reference-case families each yield $`k`$ synthetic variants, giving $`Gk`$ nominal records, and any two variants within a family have correlation $`\rho`$ and common variance $`\sigma_Y^2`$. The sample mean then carries the Kish design effect $`1 + (k - 1)\rho`$, which gives the effective sample size $`N_{\text{eff}}`$:

```math
\mathrm{Var}(\bar{Y}) = \frac{Gk\sigma_Y^2 + Gk(k - 1)\rho\sigma_Y^2}{(Gk)^2} = \frac{\sigma_Y^2}{Gk}[1 + (k - 1)\rho] \implies N_{\text{eff}} = \frac{Gk}{1 + (k - 1)\rho} \qquad (5.2)
```

With $`G = 100`$ independent families, $`k = 20`$ variants per family (2,000 nominal tasks) and within-family correlation $`\rho = 0.5`$, the 2,000 records provide the same precision for estimating the mean as about 190 independent records. Ignoring this correlation understates standard errors by a factor of **$`3.24\times`$**. My data workflow therefore separates training and evaluation sets at the task-family level, and all confidence intervals account for hierarchical clustering at both the family and expert levels.

**How many variants a seed should yield.** The design effect is a diagnosis; the same algebra gives the prescription. A new family costs $`c_G`$, mostly expert time on the seed and its reference, and each further variant costs $`c_v`$. At a fixed budget the variance of the family-clustered mean is proportional to $`(c_G + k c_v)[1 + (k - 1)\rho] / k`$, and it is smallest at the optimum cluster size of two-stage sampling:

```math
k^* = \sqrt{\frac{c_G}{c_v} \cdot \frac{1 - \rho}{\rho}} \qquad (5.3)
```

With a seed that costs twenty variants and $`\rho = 0.5`$, $`k^* \approx 4.5`$, and the budget that buys 20 variants per family has 40% more variance than the same budget at four or five. Read the other way, moving from 20 variants per family to the optimum cuts the variance by 29% at no cost. Variants beyond $`k^*`$ buy repeated information, and the money is better spent on new families.

### 6. Staged evaluation design and coverage efficiency

Staged, blinded comparisons across independent task families test whether constraint-preserving synthesis and preference supervision meet the data requirements of post-training research teams. The task mix, generator training budget, expert review budget and final acceptance criteria stay fixed as each method is introduced:

**Table 4. The staged evaluation design.**

| Comparison stage | Method introduced | What the stage measures |
| --- | --- | --- |
| **Current data construction baseline** | Unconstrained generation plus expert review and rework | Fully loaded cost of each valid case that is ultimately accepted |
| **Add manifold-constrained generation** | Hold the generator fixed; apply affine projections and variable transformations | Change in invalid candidates and manual repairs, with the conditional distribution preserved |
| **Add decision-margin checks and reference-answer recomputation** | Recompute reference judgments below the margin threshold; use $`\gamma/s`$ to quantify Gaussian label-flip risk | Change in stale-label errors, with valid supervision on both sides of the decision boundary |
| **Apply Batch Zero acceptance criteria and filtration rules** | Jointly check the temporal information boundaries in $`ℱ_s`$, independent execution, reproducibility of reference solutions and calibration of preference standards | Post-training gains that persist once leakage is excluded and clustering by source task family is accounted for |

Raw acceptance rates favor trivial variants of easy templates, and generation speed alone omits the cost of expert repairs, so I allocate data construction and review resources by scenario-weighted task-family coverage efficiency. Let $`j`$ index predefined cells in the financial capability matrix, with strategic weights $`w_j \ge 0`$ ($`\sum_j w_j = 1`$) and required counts of independent valid task families $`n_j^* > 0`$:

```math
\mathrm{Coverage} = \sum_j w_j \min\left(1, \frac{n_j^{\text{valid}}}{n_j^*}\right), \qquad \mathrm{Efficiency} = \frac{\mathrm{Coverage}}{C_{\text{generation}} + C_{\text{review}} + C_{\text{rework}} + C_{\text{adjudication}}} \qquad (6.1)
```

Once a cell reaches its target of $`n_j^*`$ independent task families, further variants of simple templates add no coverage. Coverage is concave and piecewise linear in the family counts, so the allocation that maximizes it per unit cost fills cells in decreasing order of $`w_j / (n_j^* c_j)`$, with $`c_j`$ the cost of one valid family in cell $`j`$. Four controls enter the efficiency: manifold constraint enforcement, decision-margin label recomputation, separation of temporal information sets and Batch Zero calibration.

The team averaged roughly 500 completed financial tasks a week under these rules, and the partner company carried Batch Zero into its other post-training programs.
