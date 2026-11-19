# Design notes

Why each piece of the design is the way it is, what it was chosen over, and what the workflow has to supply for the
results to hold. Proofs are in [derivations](derivations.md); figures are recomputed in
[results](../reproduce/results.md).

## The four objects and what a change invalidates

A generated case is four objects, and a change to the scenario does not reach them equally.

| Object | What it is | When a change to the scenario makes it stale | What decides |
| --- | --- | --- | --- |
| Scenario | The financial inputs and the task's information set | Always: it is the thing that changed | The generator, which keeps identities exact (D3) |
| Reference conclusion | The decision and the calculation behind it | The decision, when the margin is small; the calculation, whenever a decisive figure moves | The margin certificate for the decision (D5, D6); recomputation for the calculation |
| Evidence | The filing, footnote or contract term the conclusion relies on | When the decisive variable changes | A check that the cited evidence still contains the decisive figure |
| Preference | The expert's five-level judgment between two responses, with reasons | When the perturbation approaches or crosses the boundary | Fresh model trajectories and a fresh judgment |

The system works like an incremental build. The objects form a dependency graph, a change marks everything
downstream as stale, and the margin certificate is the cache-hit test that says which labels may be reused.

## Choices and the alternatives they beat

| Choice | Alternative | Why | When I would switch |
| --- | --- | --- | --- |
| **Structure by construction** (projection, transforms, computed totals) | Generate freely, then filter | Filtering costs $`1/\alpha_m`$ draws per valid input and grows geometrically with the number of identities ([D2](derivations.md#d2-rejection-sampling-decays-geometrically-in-the-number-of-identities)) | Constraints loose enough that acceptance stays high |
| **Covariance-weighted projection** $`K = \Sigma A^\top(A\Sigma A^\top)^{-1}`$ | Euclidean projection | Returns the exact conditional law; the Euclidean one distorts it unless $`\Sigma \propto I`$ ([D3](derivations.md#d3-the-covariance-weighted-projection-samples-the-exact-conditional-law)) | A non-Gaussian latent law, where a constraint layer takes over |
| **Identities to the projection, inequalities to transformations and layers** | One mechanism for both | The projection gives the right distribution for equalities; bijections and constraint layers handle ranges and orderings | No case |
| **Computed totals** | Sample totals and project | Removes a source of rounding residual and keeps identities exact at reporting precision | No case |
| **Copulas for cross-sectional dependence** | Independent column sampling | Preserves rank and tail dependence among size, leverage and margins | Small samples where the tails cannot be fitted |
| **Deep tabular generators followed by mandatory rule checks** | Trust the learned fit | A learned model fits multimodal marginals and guarantees no structure ([D1](derivations.md#d1-distributional-fit-does-not-imply-structural-validity)) | No case |
| **Recompute below the margin threshold, inherit above it** | Always recompute, or never | Always spends expert hours on variants far from the boundary; never leaves 40.13% stale labels at $`\gamma/s = 0.25`$ ([D6](derivations.md#d6-stale-label-risk-covenants-and-the-recompute-rule)) | No case: the threshold is a parameter, not a switch |
| **Five-level pairwise preference with cited evidence** | Binary preference | One graded judgment carries the information of 1.4 to 2 binary ones and records why ([D7](derivations.md#d7-the-information-in-a-graded-preference)) | Graders who cannot hold five levels consistently, which Batch Zero measures before scale |
| **Three separated information sets** ($`ℱ_{\text{gen}}`$, $`ℱ_s`$, $`ℱ_{\text{eval}}`$) | One "scenario" object | Leakage becomes a specification error that can be checked ([D8](derivations.md#d8-endpoint-conditioning-and-what-it-leaks)) | No case |
| **Family-level splits and clustered intervals** | Record-level splits | Record-level splits leak templates across train and test and overstate precision by 3.24 times ([D9](derivations.md#d9-effective-sample-size-of-clustered-variants)) | Families heterogeneous enough that $`\rho \approx 0`$ |
| **Few variants per seed, more seeds** | Many variants of each seed | Past $`k^*`$ a variant buys repeated information ([D10](derivations.md#d10-seed-leverage-and-the-optimal-number-of-variants)) | Seeds that are cheap relative to variants |
| **Coverage efficiency as the allocation target** | Raw acceptance rate or generation speed | Acceptance rewards trivial variants and speed ignores repair cost ([D11](derivations.md#d11-coverage-and-how-to-spend-on-it)) | No case |

## The five-level preference record

A preference record does more than pick a winner. It carries:

1. the **direction** of the preference, or that neither response is valid;
2. its **strength** on the five-level scale;
3. the **decisive variable** whose treatment separates the two responses;
4. the **primary evidence** relied on (filing, footnote, contract term) and the **accounting rule** that makes it
   decisive.

An example: *"Response A is better because it isolates the working-capital constraint and cites Footnote 4."* When a
perturbation crosses the boundary the old record is not edited. New model trajectories are sampled and judged
afresh, because the decisive variable itself may have changed.

### Three signals a training pipeline can consume separately

| Signal | Training use | Published objectives of this form |
| --- | --- | --- |
| Direction | Standard pairwise preference optimization | DPO (Rafailov et al. 2023) |
| Strength | A filter on which pairs enter training; a soft target below one; an offset or margin that grows with strength; an explicit model of ties | Llama 3's filtering by strength (Grattafiori et al. 2024); conservative DPO (Mitchell 2023); DPO with an offset (Amini et al. 2024); margin ranking in Llama 2 (Touvron et al. 2023) |
| "Neither response is valid" | Both responses become undesirable examples in an objective that needs no pairs | KTO (Ethayarajh et al. 2024) |
| Evidence, rule and decisive variable | A rationale term in the preference objective; critique-then-score reward models; held-out audit pairs with known reasons | Rationale-augmented preference alignment (Just et al. 2024); RewardBench (Lambert et al. 2024) |

The reasons matter most near the boundary. Diagnostic tasks put cases close to a decision boundary, close cases
produce close comparisons, and a reward model is least reliable on exactly those pairs. A judgment that names the
decisive variable and the evidence behind it still trains a critic to look at the right number, and it lets an
auditor reject a preference given for the wrong reason.

## Batch Zero as an acceptance test for the specification

| What Batch Zero checks | How a failure shows up | What gets revised |
| --- | --- | --- |
| Independent execution | Experts cannot build the task from the specification alone | Instructions, missing inputs |
| Reproducible references | Two experts reach different reference answers | Reference solutions, boundary cases |
| Calibrated preferences | Five-level scores disagree beyond tolerance | Grading rules, anchor examples |
| Information boundaries | A forecasting task can be solved with details from $`ℱ_{\text{gen}}`$ | Filtration rules |

Only task families that pass enter Batch One. One independent re-solve at the seed removes the error from every
variant the seed would have produced: with twenty variants per seed, wrong records per family fall from 3.0 to 0.6.

## What the results need from the workflow

1. **Identities stated in latent coordinates.** D3 is exact for constraints that are affine in the latent Gaussian
   coordinates. A ratio such as return on equity becomes a difference under logs, or one side of the ratio is
   parameterized.
2. **A perturbation law per task family.** D6 uses the Gaussian law of the perturbation along the decision
   direction, $`s^2 = w^\top\Omega w`$. For tail-sensitive tasks the empirical perturbation law replaces
   $`\Phi`$, and the threshold is read from its quantile.
3. **A local slope for nonlinear rules.** For a discounted-cash-flow rule the certificate uses the local gradient
   and a curvature bound over the perturbation set:
   $`\left\vert d(x+\Delta) - d(x) - \nabla d(x)^\top\Delta\right\vert \le \tfrac12 H\|\Delta\|^2`$ when the Hessian norm is at most $`H`$.
4. **A within-family correlation per family type.** $`\rho`$ is estimated from repeated expert grading within
   families and drives both the intervals and the number of variants per seed.
5. **Graders who do not see the previous answer.** When a reference is recomputed, the grader of the new pair is
   not shown the old label.

## What to monitor

- **Generator health.** The mean of the displacement statistic $`T`$ per batch against $`m`$
  ([D4](derivations.md#d4-the-displacement-is-a-chi-square-monitor)). A rise is upstream drift, caught while the
  batch is being corrected.
- **Share of variants below the margin threshold, per cell.** It is the recomputation cost of the cell, and it
  rises when generation concentrates near the boundary.
- **Leakage through metadata.** File names, timestamps and generator seeds can reveal $`ℱ_{\text{gen}}`$
  when the paths themselves are clean.
- **Near-duplicate families.** Two seeds that share a template inflate $`G`$. Seeds are clustered by structure
  before $`N_{\text{eff}}`$ is computed.

## Extensions

1. **Multiplicative identities.** An exact sampler for identities that stay nonlinear in every coordinate system,
   short of sampling on the manifold.
2. **Margins along a chain of calculations.** A conclusion that depends on several steps has a margin at each one;
   the union bound of D6 is the first composition rule, and a sharper one would use the dependence between steps.
3. **Filtration checks compiled from the specification.** A task's permitted filtration compiled into automatic
   tests that every input is $`ℱ_s`$-measurable.
