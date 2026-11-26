#!/usr/bin/env python3
"""Simulate the four models and show the behavioral signature they predict (paper Figure 6).

Panel row 1: theoretical response error vs feedback offset, using the illustrative parameters in
the authors' main.m (X = [1 0 1 10 1 20 0.1 20 0.8], target = 180 deg, glimpse at turn onset).
For the Hybrid model the two branches are drawn with marker size proportional to the probability
of choosing each one.

Panel row 2: synthetic participants. Each model is simulated on the real trial structure of all
30 participants (same targets, feedback angles and glimpse times) and summarised exactly like the
observed data: participant-level means in 20-degree offset bins.

Output: figures/model_predictions.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from heading_models import (MODEL_NAMES, binned_signature, compute_averaging, compute_kalman,
                            compute_no_feedback, compute_p_true, load_subjects, simulate)

ROOT = Path(__file__).resolve().parents[1]
X_ILLUSTRATE = np.array([1, 0, 1, 10, 1, 20, 0.1, 20, 0.8], dtype=float)
EDGES = np.arange(-180, 181, 20)


def main() -> None:
    subs = load_subjects(ROOT / "data" / "dataconf.csv")
    offset = np.arange(-180, 181, dtype=float)
    target = np.full_like(offset, 180.0)
    fb_true = np.zeros_like(offset)
    fb = fb_true + offset

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), sharex=True)
    red, blue = "#c8384a", "#3b7dd8"

    for i, m in enumerate([1, 2, 3, 4]):
        ax = axes[0, i]
        if m == 1:
            mean, _ = compute_no_feedback(X_ILLUSTRATE, target)
            ax.plot(offset, mean, color=red, lw=3)
        elif m == 2:
            mean, _, _ = compute_kalman(X_ILLUSTRATE, target, fb, fb_true)
            ax.plot(offset, mean, color=red, lw=3)
        elif m == 3:
            mean, _, _ = compute_averaging(X_ILLUSTRATE, target, fb, fb_true)
            ax.plot(offset, mean, color=red, lw=3)
        else:
            p = compute_p_true(X_ILLUSTRATE, fb, fb_true)
            m0, _ = compute_no_feedback(X_ILLUSTRATE, target)
            m1, _, _ = compute_kalman(X_ILLUSTRATE, target, fb, fb_true)
            step = slice(None, None, 6)
            ax.scatter(offset[step], m1[step], s=60 * p[step] + 1, color=red, label="use visual estimate")
            ax.scatter(offset[step], m0[step], s=60 * (1 - p[step]) + 1, color=blue,
                       label="use path integration")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_title(MODEL_NAMES[m])
        ax.set_ylim(-100, 100)
        ax.axhline(0, color="k", lw=0.6)

    # Synthetic participants on the real trial structure (one agent per participant).
    rng = np.random.default_rng(2022)
    for i, m in enumerate([1, 2, 3, 4]):
        ax = axes[1, i]
        offs, errs, grp = [], [], []
        for s, t in subs.items():
            sim = simulate(m, X_ILLUSTRATE, t, rng)
            f = sim.fb_flag
            offs.append(sim.fb[f] - sim.fb_true[f])
            errs.append(sim.respond[f] - sim.target[f])
            grp.append(np.full(f.sum(), s))
        offs, errs, grp = map(np.concatenate, (offs, errs, grp))
        ax.scatter(offs, errs, s=3, alpha=0.08, color="#888888")
        summ = binned_signature(offs, errs, grp, EDGES)
        ax.errorbar(summ["mid"], summ["mean"], yerr=summ["sem"], fmt="o-", color=red, ms=3, capsize=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-150, 150)
        ax.set_xlabel("feedback offset, $f-\\theta_{t_f}$ (deg)")

    axes[0, 0].set_ylabel("predicted response error (deg)\n(illustrative parameters)")
    axes[1, 0].set_ylabel("simulated response error (deg)\n30 agents, real trial structure")
    for ax in axes.flat:
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model predictions: response error vs. visual-feedback offset", fontsize=12)
    fig.tight_layout()
    out = ROOT / "figures" / "model_predictions.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
