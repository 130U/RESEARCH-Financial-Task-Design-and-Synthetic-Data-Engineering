"""Recompute every figure in the article and compare it with the text.

    python reproduce/run_all.py            # full sizes
    python reproduce/run_all.py --quick    # smaller sample sizes, looser tolerances

Writes reproduce/results.md and exits with status 1 if any figure disagrees.
Standard library only (Python 3.12+).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import constraints  # noqa: E402
import supervision  # noqa: E402
import supervision_design  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick
    sections: list[tuple[str, list]] = []
    rows: list = []

    def section(title: str) -> None:
        nonlocal rows
        rows = []
        sections.append((title, rows))

    def check(label, value, article, tol, where, shown=None):
        rows.append((label, value, shown if shown is not None else article, where, abs(value - article) <= tol))

    def check_true(label, ok, where, shown="holds"):
        rows.append((label, None, shown, where, bool(ok)))

    design = supervision_design.run()

    # ------------------------------------------------------------------ Section 1
    section("Distributional fit and task validity (Section 1)")
    for row in constraints.w2_tv_counterexample():
        check_true(f"σ = {row['sigma']:g}: no generated record lies on the constraint set, Q(M) = 0",
                   row["share_exactly_on_M"] == 0.0, "§1")
        check(f"σ = {row['sigma']:g}: coupling distance over σ√m", row["coupling_rms"] / row["bound"], 1.0, 0.02, "§1")

    # ------------------------------------------------------------------ Section 2
    section("Conditioning in place of rejection (Section 2)")
    article = {1: (3.9878, 25.08), 2: (0.1590, 628.84), 4: (0.0002529, 395_442.59)}
    for row in constraints.rejection_table():
        acc, draws = article[row["m"]]
        digits = 4 if row["m"] < 4 else 7
        check(f"{row['m']} constraints: acceptance rate (%)", round(100 * row["acceptance"], digits), acc, 1e-9, "§2 table")
        check(f"{row['m']} constraints: expected candidate draws", round(row["expected_draws"], 2), draws, 1e-9, "§2 table")
    sampled = constraints.rejection_by_sampling(draws=400_000 if q else 2_000_000)
    check("2 constraints: acceptance rate by direct sampling (%)", 100 * sampled, 0.1590, 0.03 if q else 0.01, "§2")
    if not q:
        run = constraints.reference_run()
        check("Reference run: draws", run["draws"], 200_000, 0, "§2")
        check("Reference run: fields", run["fields"], 12, 0, "§2")
        check("Reference run: identities", run["constraints"], 4, 0, "§2")
        check("Reference run: maximum constraint residual", run["max_abs_constraint_residual"], 3.0e-15, 3.0e-15, "§2", shown="3.0e-15")
        check("Reference run: covariance error, relative Frobenius norm (%)", run["relative_frobenius_cov_error_percent"], 0.359, 0.0005, "§2")
        disp = design["displacement"]
        check("Reference run: average displacement T", disp["mean_on_reference_run"], 4.01, 0.005, "§2")
        check("Noncentrality after a 10% drift in mean receivables", disp["noncentrality_after_10pct_receivables_drift"], 0.197, 0.0005, "§2")
        check("Expected displacement after the drift, m + δ", 4 + disp["noncentrality_after_10pct_receivables_drift"], 4.20, 0.005, "§2")
        check("Average displacement on a drifted run", disp["mean_after_drift"], 4.20, 0.02, "§2")
        check("Drift signal in a batch of 1,000 draws (standard errors)", disp["standard_errors_at_batch_1000"], 2, 0.2, "§2")
        signal = supervision_design.drift_signal(disp["noncentrality_after_10pct_receivables_drift"] - disp["noncentrality_reference"], 4, 10_000)
        check("Drift signal in a batch of 10,000 draws (standard errors)", signal, 6.65, 0.5, "§2", shown="more than six")

    # ------------------------------------------------------------------ Section 3
    section("Decision margins and recomputation (Section 3)")
    article_flip = {2.0: 2.28, 1.0: 15.87, 0.25: 40.13}
    n_draws = 200_000 if q else 1_000_000
    for row in supervision.stale_label_rates(draws=n_draws):
        r, p = row["gamma_over_s"], row["exact"]
        four_se = 4 * 100 * (p * (1 - p) / n_draws) ** 0.5
        check(f"γ/s = {r:g}: inherited-label error Φ(−γ/s) (%)", round(100 * p, 2), article_flip[r], 1e-9, "§3")
        check(f"γ/s = {r:g}: the same rate by direct sampling (%)", 100 * row["sampled"], 100 * p, four_se, "§3", shown=f"{100 * p:.2f}")
    check_true("Margin certificate holds on 100,000 random cases", supervision.lipschitz_certificate_holds(), "§3")
    rec = design["recompute"]
    check("Flip rate at γ/s = 1.5 (%)", 100 * supervision_design.stale_rate(1.5), 6.68, 0.005, "D6")
    check("Two covenants at γ/s of 1.5 and 2: bound on the flip rate (%)", 100 * rec["two_covenant_bound"], 8.96, 0.005, "§3")
    check("Recompute threshold at α = 5%", rec["threshold"][0.05], 1.645, 0.0005, "§3")
    check("Recompute threshold at α = 1%", rec["threshold"][0.01], 2.326, 0.0005, "§3")
    check("Inherited-label error at γ/s = 3 (%)", 100 * rec["stale_rate"][3.0], 0.13, 0.005, "§3")
    info = {0.0: (0.637, 0.912, 1.43), 0.5: (0.581, 0.902, 1.55), 1.0: (0.439, 0.863, 1.97)}
    for gap, (binary, five, ratio) in info.items():
        got = design["information"][gap]
        check(f"Quality gap {gap:g}: information in a binary judgment", got["binary"], binary, 0.0005, "§3 table")
        check(f"Quality gap {gap:g}: information in a five-level judgment", got["five_level"], five, 0.0005, "§3 table")
        check(f"Quality gap {gap:g}: ratio", got["five_level"] / got["binary"], ratio, 0.005, "§3 table")
    best = design["best_five_level"]
    check("Best inner cutpoint at a tie", best["cuts"][0], 0.38, 0.005, "§3")
    check("Best outer cutpoint at a tie", best["cuts"][1], 1.24, 0.005, "§3")
    check("Information at the best cutpoints, share of a continuous response", best["information_at_tie"], 0.920, 0.0005, "§3")

    # ------------------------------------------------------------------ Section 4
    section("Information sets for temporal tasks (Section 4)")
    br = supervision.bridge_leakage(draws=200_000 if q else 1_000_000)
    check("Brownian bridge: midpoint variance ratio with the endpoint disclosed", br["exact_variance_ratio"], 0.5, 1e-12, "§4")
    check("Brownian bridge: apparent RMSE reduction (%)", round(100 * br["exact_rmse_reduction"], 2), 29.29, 1e-9, "§4")
    check("Brownian bridge: the same reduction by direct sampling (%)", 100 * br["sampled_rmse_reduction"], 29.29, 0.4 if q else 0.2, "§4")
    article_ou = {0.0: 29.29, 0.25: 21.10, 0.5: 14.50, 1.0: 6.15, 2.0: 0.90}
    for kh, want in article_ou.items():
        got = design["leakage"][kh]
        check(f"Mean reversion κh = {kh:g}: apparent RMSE reduction (%)", 100 * got["rmse_drop"], want, 0.005, "§4 table")
        check(f"Mean reversion κh = {kh:g}: closed form against Gaussian conditioning", got["ratio"], got["check"], 1e-5, "§4", shown="equal")

    # ------------------------------------------------------------------ Section 5
    section("Batch Zero and task families (Section 5)")
    bz = design["batch_zero"]
    for resolves, want in zip((0, 1, 2), (3.0, 0.6, 0.12)):
        check(f"Wrong records per family after {resolves} independent re-solve(s)", bz["wrong_records"][resolves], want, 1e-9, "§5")
    ess = supervision.effective_sample_size()
    check("Effective sample size of 2,000 variants in 100 families", round(ess["effective"]), 190, 0, "§5")
    check("Understatement of standard errors (times)", round(ess["se_understatement"], 2), 3.24, 1e-9, "§5")
    deff = supervision.clustering_by_sampling(reps=5_000 if q else 20_000)
    check("Design effect by direct sampling", deff, 10.5, 0.6 if q else 0.3, "§5")
    check("Optimal variants per family, seed cost of twenty variants, ρ = 0.5", bz["best_variants"]["cost ratio 20, rho 0.5"], 4.5, 0.05, "§5")
    check("Variance at 20 variants per family over variance at four or five (% more)", 100 * (bz["variance_at_k"][20] / bz["variance_at_k"][4] - 1), 40, 0.5, "§5")
    check_true("Four and five variants per family give the same variance", abs(bz["variance_at_k"][4] - bz["variance_at_k"][5]) < 1e-12, "§5")
    check("Variance cut from 20 variants to the optimum (%)", 100 * (1 - 1 / bz["variance_k20_over_best"]), 29, 0.5, "§5")

    total = sum(len(r) for _, r in sections)
    failed = sum(1 for _, r in sections for row in r if not row[4])

    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, float):
            if x != 0 and abs(x) < 1e-6:
                return f"{x:.2e}"
            return f"{x:.4f}" if abs(x) < 1000 else f"{x:,.2f}"
        return f"{x:,}" if isinstance(x, int) else str(x)

    def art(x):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return str(x)
        return format(x, ",.10g")

    lines = [
        "# Results",
        "",
        "Every figure in the article, recomputed.",
        "",
        f"{total - failed} of {total} figures agree with the article" + (" · quick run, reference run skipped" if q else ""),
    ]
    for title, section_rows in sections:
        lines += ["", f"## {title}", "", "| Quantity | Recomputed | Article | Where | Agrees |", "| --- | ---: | ---: | --- | :---: |"]
        for label, value, shown, where, ok in section_rows:
            lines.append(f"| {label.replace(chr(124), chr(92) + chr(124))} | {fmt(value)} | {art(shown)} | {where} | {'✓' if ok else '✗'} |")
    report = "\n".join(lines) + "\n"
    (HERE / "results.md").write_text(report, encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
