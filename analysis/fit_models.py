#!/usr/bin/env python3
"""Fit all four models to every participant and compare them with BIC (paper Figure 10).

As in the authors' main.m (data_flag = 3), each model is fit jointly to the no-feedback trials
(always explained by the path-integration model) and the feedback trials (explained by the model
being tested), with 100 random starts per fit.

Outputs
    results/model_fits.csv          one row per participant x model: parameters, nLL, BIC, AIC
    figures/model_comparison.png    BIC relative to Hybrid + number of participants best fit
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heading_models import MODEL_NAMES, PARAM_NAMES, fit_subject, load_subjects

ROOT = Path(__file__).resolve().parents[1]
COLORS = {1: "#7a7a7a", 2: "#3b7dd8", 3: "#e0a030", 4: "#c8384a"}
# The paper reports 5/8/9/9 free parameters; the authors' code (fit_any_parallel_v1.m) uses 4/7/9/9
# because it fixes gamma = 1 in PI and s_f = 1 in KF. Both conventions are reported.
K_PAPER = {1: 5, 2: 8, 3: 9, 4: 9}


def _fit(job):
    subj, model, trials, n_starts = job
    r = fit_subject(model, trials, n_starts=n_starts, seed=1)
    row = {"subj": subj, "model": model, "model_name": MODEL_NAMES[model],
           "nLL": r["nLL"], "BIC": r["BIC"], "AIC": r["AIC"], "k": r["k"], "n_trials": r["n_trials"]}
    row.update(dict(zip(PARAM_NAMES, r["X"])))
    return row


def plot_model_comparison(fits: pd.DataFrame, out: Path, title_suffix: str = "",
                          bic_col: str = "BIC") -> pd.Series:
    bic = fits.pivot(index="subj", columns="model", values=bic_col)
    winners = bic.idxmin(axis=1)
    counts = winners.value_counts().reindex([1, 2, 3, 4], fill_value=0)
    delta = bic.sub(bic[4], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for m in [1, 2, 3, 4]:
        y = np.full(len(delta), m) + rng.uniform(-0.15, 0.15, len(delta))
        ax.scatter(delta[m], y, s=28, facecolors="none", edgecolors=COLORS[m], linewidths=1.3)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(10, color="k", lw=0.8, ls="--")
    ax.set_yticks([1, 2, 3, 4], [MODEL_NAMES[m] for m in [1, 2, 3, 4]])
    ax.set_xlabel("BIC(model) − BIC(Hybrid)")
    ax.set_xlim(-200, max(600, float(np.nanmax(delta.values)) + 20))
    ax.invert_yaxis()
    ax.set_title("A. Relative fit (right of 0 = worse than Hybrid)", loc="left", fontsize=10)

    ax = axes[1]
    ax.barh([1, 2, 3, 4], counts.values, color=[COLORS[m] for m in [1, 2, 3, 4]])
    for m, c in zip([1, 2, 3, 4], counts.values):
        ax.text(c + 0.3, m, str(c), va="center")
    ax.set_yticks([1, 2, 3, 4], [])
    ax.invert_yaxis()
    ax.set_xlabel("participants best fit (lowest BIC)")
    ax.set_xlim(0, len(bic) + 2)
    ax.set_title("B. Winning model per participant", loc="left", fontsize=10)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Model comparison{title_suffix}", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data" / "dataconf.csv")
    p.add_argument("--n-starts", type=int, default=100)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--plot-only", action="store_true", help="re-plot from results/model_fits.csv")
    args = p.parse_args()

    out_csv = ROOT / "results" / "model_fits.csv"
    if args.plot_only:
        fits = pd.read_csv(out_csv)
    else:
        subs = load_subjects(args.data)
        jobs = [(s, m, t, args.n_starts) for m in [1, 2, 3, 4] for s, t in subs.items()]
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(_fit, jobs))
        fits = pd.DataFrame(rows)
        fits["subj_num"] = fits["subj"].str.lstrip("s").astype(int)
        fits = fits.sort_values(["model", "subj_num"]).drop(columns="subj_num")
    fits["k_paper"] = fits["model"].map(K_PAPER)
    fits["BIC_paper_k"] = 2 * fits["nLL"] + np.log(fits["n_trials"]) * fits["k_paper"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fits.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    for col, label, fname in [("BIC_paper_k", " (k = 5/8/9/9 as reported in the paper)", "model_comparison.png"),
                              ("BIC", " (k = 4/7/9/9 as in the authors' code)", "model_comparison_code_k.png")]:
        counts = plot_model_comparison(fits, ROOT / "figures" / fname, label, col)
        bic = fits.pivot(index="subj", columns="model", values=col)
        print(f"\n{col}{label}")
        for m, c in counts.items():
            print(f"  {MODEL_NAMES[m]:17s} {c}")
        print("  Summed BIC:", bic.sum().round(1).to_dict())
        print("  Median BIC(model) - BIC(Hybrid):", bic.sub(bic[4], axis=0).median().round(1).to_dict())
        print("  Non-Hybrid winners:", bic.idxmin(axis=1)[lambda s: s != 4].to_dict())


if __name__ == "__main__":
    main()
