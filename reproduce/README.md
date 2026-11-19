# Reproduce

```bash
python reproduce/run_all.py           # every figure in the article, under a minute
python reproduce/run_all.py --quick   # smaller sample sizes, reference run skipped
```

Python 3.12 or later, standard library only. `run_all.py` recomputes every figure in the article, compares each
with the text, writes [`results.md`](results.md) and exits with status 1 if any figure disagrees. The same command
runs on every push.

| Script | Article | What it computes |
| --- | --- | --- |
| [`constraints.py`](constraints.py) | §1, §2 | The $`W_2`$ and total-variation counterexample; the rejection-sampling table at $`\epsilon = 0.05`$; the reference run of the covariance-weighted projection, with its constraint residual and the error of the sample covariance against the conditional covariance |
| [`supervision.py`](supervision.py) | §3 to §5 | The inherited-label error $`\Phi(-\gamma/s)`$ with a three-dimensional sampling check; the margin certificate on random cases; Brownian bridge leakage; the effective sample size of clustered variants |
| [`supervision_design.py`](supervision_design.py) | §2 to §5 | The displacement statistic on the reference run and under drift; the recompute thresholds and the covenant bound; leakage for mean-reverting paths; the Fisher information of binary and five-level judgments and the best cutpoints; wrong records per family; the optimal number of variants per seed |

**The reference run.** `constraints.py` defines it in full: twelve balance-sheet and income fields with a
two-factor correlation structure (size and leverage), four identities (assets sum to total assets, liabilities sum
to total liabilities, assets equal liabilities plus equity, receivables at 60 days of revenue), 200,000 draws.
Normal draws come from the Box-Muller transform of `random.random()`, whose sequence is fixed across Python
versions for a given seed, so the run gives the same figures everywhere.

**Closed forms and sampling checks.** Most figures in the article are closed-form and are compared to the last
printed digit. Where a closed form has a direct sampling counterpart (the flip rate, bridge leakage, the design
effect), the code draws it as well and compares the two within sampling error.

**Linear algebra.** Matrix products, inversion and the Cholesky factor are written out in `constraints.py`, so
nothing beyond the standard library is needed.
