#!/usr/bin/env python3
"""Extension: how should changing the objective reliability of the visual glimpse change behavior?

The original task used one fixed mix of informative and misleading glimpses. We
  1. estimate that mix from the data: feedback offsets are modelled as
        offset ~ rho * Normal(0, sigma) + (1 - rho) * Uniform(-180, 180),
     where rho is the objective reliability (proportion of informative glimpses);
  2. simulate a new experiment with four reliability blocks (rho = 0.25, 0.50, 0.75, 0.95), using
     each participant's fitted Hybrid parameters and their own targets / glimpse times;
  3. compare two hypotheses about the participant's prior that a glimpse is valid (omega):
       FIXED     omega stays at its fitted value (people do not track reliability)
       ADAPTIVE  omega shifts with the environment in log-odds:
                 logit(omega_c) = logit(omega_fit) + logit(rho_c) - logit(rho_original)
  4. derive predictions (signature curves, P(follow vision), switch point, accuracy), and
  5. check the proposed design is informative: refit the Hybrid model to every simulated block and
     test whether the recovered omega tracks rho under ADAPTIVE but not under FIXED.

Requires results/model_fits.csv.

Outputs
    results/extension_offset_mixture.csv
    results/extension_predictions.csv
    results/extension_refits.csv
    figures/extension_offset_mixture.png
    figures/extension_reliability.png
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from heading_models import (PARAM_NAMES, Trials, binned_signature, compute_p_true, fit_subject,
                            load_subjects, simulate)

ROOT = Path(__file__).resolve().parents[1]
RHOS = [0.25, 0.50, 0.75, 0.95]
EDGES = np.arange(-180, 181, 20)
RHO_COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]


def fit_offset_mixture(offset: np.ndarray) -> tuple[float, float]:
    def nll(z):
        rho, sigma = 1 / (1 + np.exp(-z[0])), np.exp(z[1])
        dens = rho * stats.norm.pdf(offset, 0, sigma) + (1 - rho) / 360
        return -np.sum(np.log(dens))

    r = minimize(nll, [0.5, np.log(20)], method="Nelder-Mead")
    return float(1 / (1 + np.exp(-r.x[0]))), float(np.exp(r.x[1]))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def make_block(t: Trials, rho: float, sigma: float, rng) -> Trials:
    """Same targets and glimpse times as the participant, new feedback offsets drawn at reliability rho."""
    fb = t.fb.copy()
    f = t.fb_flag
    n = f.sum()
    valid = rng.random(n) < rho
    off = np.where(valid, rng.normal(0, sigma, n), rng.uniform(-180, 180, n))
    fb[f] = t.fb_true[f] + off
    return Trials(t.fb_flag.copy(), t.target.copy(), fb, t.fb_true.copy(), None)


def _refit(job):
    hyp, s, rho, trials, seed, n_starts = job
    r = fit_subject(4, trials, n_starts=n_starts, seed=seed)
    return {"hypothesis": hyp, "subj": s, "rho": rho, "omega_fit": r["X"][8]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-starts", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    subs = load_subjects(ROOT / "data" / "dataconf.csv")
    fits = pd.read_csv(ROOT / "results" / "model_fits.csv")
    Xh = {r.subj: r[PARAM_NAMES].to_numpy(float) for _, r in fits[fits.model == 4].iterrows()}

    # 1. Objective reliability of the original experiment.
    all_off = np.concatenate([t.fb[t.fb_flag] - t.fb_true[t.fb_flag] for t in subs.values()])
    rho0, sigma0 = fit_offset_mixture(all_off)
    pd.DataFrame([{"rho_original": rho0, "sigma_valid_deg": sigma0, "n_trials": len(all_off)}]).to_csv(
        ROOT / "results" / "extension_offset_mixture.csv", index=False)
    print(f"Original task: rho = {rho0:.3f} informative glimpses, sigma = {sigma0:.1f} deg")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = np.linspace(-180, 180, 361)
    ax.hist(all_off, bins=np.arange(-180, 181, 10), density=True, color="#cccccc", label="observed offsets")
    ax.plot(xs, rho0 * stats.norm.pdf(xs, 0, sigma0) + (1 - rho0) / 360, color="#c8384a", lw=2,
            label=f"mixture fit: ρ = {rho0:.2f}, σ = {sigma0:.0f}°")
    ax.set_xlabel("feedback offset (deg)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "extension_offset_mixture.png", dpi=200)
    plt.close(fig)

    # 2-4. Simulate the proposed experiment under both hypotheses.
    rng = np.random.default_rng(2026)
    omega_fit = {s: Xh[s][8] for s in subs}
    sims = {}  # (hyp, rho) -> list of (subj, Trials, used_visual)
    pred_rows, refit_jobs = [], []
    for hyp in ["FIXED", "ADAPTIVE"]:
        for ci, rho in enumerate(RHOS):
            lst = []
            for si, (s, t) in enumerate(subs.items()):
                X = Xh[s].copy()
                if hyp == "ADAPTIVE":
                    X[8] = expit(logit(omega_fit[s]) + logit(rho) - logit(rho0))
                block = make_block(t, rho, sigma0, rng)
                sim, used = simulate(4, X, block, rng, return_choice=True)
                lst.append((s, sim, used, X))
                refit_jobs.append((hyp, s, rho, sim, 9000 + 100 * ci + si, args.n_starts))
            sims[(hyp, rho)] = lst

            # summary statistics for this block
            f_all = np.concatenate([sm.fb_flag for _, sm, _, _ in lst])
            off = np.concatenate([sm.fb[sm.fb_flag] - sm.fb_true[sm.fb_flag] for _, sm, _, _ in lst])
            err = np.concatenate([sm.respond[sm.fb_flag] - sm.target[sm.fb_flag] for _, sm, _, _ in lst])
            used = np.concatenate([u[sm.fb_flag] for _, sm, u, _ in lst])
            # absolute error relative to the *true* target in the feedback trials (accuracy)
            pred_rows.append({
                "hypothesis": hyp, "rho": rho,
                "mean_omega": np.mean([X[8] for *_, X in lst]),
                "p_follow_small_offset(<30)": used[np.abs(off) < 30].mean(),
                "p_follow_large_offset(60-180)": used[np.abs(off) >= 60].mean(),
                "mean_abs_error_fb_trials": np.mean(np.abs(err)),
            })

    # Theoretical P(follow vision) vs offset, averaged over agents, glimpse at median turn time.
    offs = np.arange(0, 181, 1.0)
    med_fbt = float(np.median(np.concatenate([t.fb_true[t.fb_flag] for t in subs.values()])))
    curves = {}
    for hyp in ["FIXED", "ADAPTIVE"]:
        for rho in RHOS:
            ps = []
            for s in subs:
                X = Xh[s].copy()
                if hyp == "ADAPTIVE":
                    X[8] = expit(logit(omega_fit[s]) + logit(rho) - logit(rho0))
                fbt = np.full_like(offs, med_fbt)
                ps.append(compute_p_true(X, X[2] * fbt + offs, fbt))  # offset from the PI estimate
            curves[(hyp, rho)] = np.mean(ps, axis=0)
    for row in pred_rows:
        c = curves[(row["hypothesis"], row["rho"])]
        below = np.where(c < 0.5)[0]
        row["switch_offset_deg(P_follow=0.5)"] = float(offs[below[0]]) if len(below) else np.nan
    pred = pd.DataFrame(pred_rows)
    pred.to_csv(ROOT / "results" / "extension_predictions.csv", index=False)
    print(pred.round(3).to_string(index=False))

    # 5. Refit the simulated experiment.
    with ProcessPoolExecutor(args.workers) as ex:
        refits = pd.DataFrame(list(ex.map(_refit, refit_jobs)))
    refits["omega_true"] = [
        (expit(logit(omega_fit[s]) + logit(r) - logit(rho0)) if h == "ADAPTIVE" else omega_fit[s])
        for h, s, r in zip(refits.hypothesis, refits.subj, refits.rho)]
    refits.to_csv(ROOT / "results" / "extension_refits.csv", index=False)
    tests = {}
    for hyp in ["FIXED", "ADAPTIVE"]:
        w = refits[refits.hypothesis == hyp].pivot(index="subj", columns="rho", values="omega_fit")
        fr = stats.friedmanchisquare(*[w[r] for r in RHOS])
        # per-participant Spearman between rho and recovered omega, then sign test across participants
        rs = [stats.spearmanr(RHOS, w.loc[s, RHOS]).statistic for s in w.index]
        tests[hyp] = (fr.statistic, fr.pvalue, np.nanmedian(rs))
        print(f"{hyp}: Friedman chi2 = {fr.statistic:.1f}, p = {fr.pvalue:.2g}; "
              f"median within-participant Spearman(rho, omega_fit) = {np.nanmedian(rs):.2f}")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for hi, hyp in enumerate(["FIXED", "ADAPTIVE"]):
        ax = axes[hi, 0]
        for ci, rho in enumerate(RHOS):
            lst = sims[(hyp, rho)]
            off = np.concatenate([sm.fb[sm.fb_flag] - sm.fb_true[sm.fb_flag] for _, sm, _, _ in lst])
            err = np.concatenate([sm.respond[sm.fb_flag] - sm.target[sm.fb_flag] for _, sm, _, _ in lst])
            g = np.concatenate([np.full(sm.fb_flag.sum(), s) for s, sm, _, _ in lst])
            sm_ = binned_signature(off, err, g, EDGES)
            ax.errorbar(sm_["mid"], sm_["mean"], yerr=sm_["sem"], fmt="o-", ms=3, capsize=2,
                        color=RHO_COLORS[ci], label=f"ρ = {rho:.2f}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-30, 70)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("feedback offset (deg)")
        ax.set_ylabel(f"{hyp} prior\nresponse error (deg)")
        ax.set_title("Predicted behavioral signature", fontsize=10)
        ax.legend(frameon=False, fontsize=8, title="glimpse reliability", title_fontsize=8)

        ax = axes[hi, 1]
        for ci, rho in enumerate(RHOS):
            ax.plot(offs, curves[(hyp, rho)], color=RHO_COLORS[ci], lw=2.2, label=f"ρ = {rho:.2f}")
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_ylim(0, 1)
        ax.set_xlabel("|glimpse − path-integration estimate| (deg)")
        ax.set_ylabel("P(use visual estimate)")
        ax.set_title("Probability of following the glimpse (mean over 30 agents)", fontsize=10)

        ax = axes[hi, 2]
        d = refits[refits.hypothesis == hyp]
        for ci, rho in enumerate(RHOS):
            v = d[d.rho == rho]["omega_fit"]
            x = ci + np.random.default_rng(ci).uniform(-0.12, 0.12, len(v))
            ax.scatter(x, v, s=12, color=RHO_COLORS[ci], alpha=0.8)
            ax.plot([ci - 0.25, ci + 0.25], [v.median()] * 2, color="k", lw=2)
            ax.plot([ci - 0.25, ci + 0.25], [d[d.rho == rho]["omega_true"].median()] * 2,
                    color="#c8384a", lw=1.5, ls="--")
        chi2, p, rs = tests[hyp]
        ax.set_xticks(range(4), [f"{r:.2f}" for r in RHOS])
        ax.set_xlabel("glimpse reliability ρ (block)")
        ax.set_ylabel("recovered prior ω (Hybrid refit)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Refit: Friedman χ² = {chi2:.1f}, p = {p:.1g}\n"
                     f"black = median recovered, red dashed = median true", fontsize=9)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Extension: manipulating glimpse reliability (original task ρ ≈ {rho0:.2f}). "
                 "Top: people ignore reliability; bottom: people adapt their prior.", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "extension_reliability.png", dpi=200)
    print("Saved figures/extension_reliability.png")


if __name__ == "__main__":
    main()
