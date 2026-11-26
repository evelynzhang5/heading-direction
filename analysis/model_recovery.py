#!/usr/bin/env python3
"""Model recovery / confusion matrix, following the authors' main.m (paper Figure S4).

For each generating model, simulate 30 participants using that model's fitted parameters for each
real participant (generate_simulationParameters type 4), each paired with the trial structure of a
randomly chosen real participant. Fit all four models to every simulated dataset and record which
has the lowest BIC.

Requires results/model_fits.csv (run fit_models.py first).

Outputs
    results/model_recovery.csv
    figures/model_recovery.png     p(fit | sim) confusion matrix and p(sim | fit) inversion matrix
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heading_models import MODEL_NAMES, PARAM_NAMES, fit_subject, load_subjects, simulate

ROOT = Path(__file__).resolve().parents[1]


def _job(args):
    m_sim, i, x, trials, seed, n_starts = args
    sim = simulate(m_sim, x, trials, np.random.default_rng(seed))
    row = {"sim_model": m_sim, "sim": i}
    for m_fit in [1, 2, 3, 4]:
        row[f"BIC_{m_fit}"] = fit_subject(m_fit, sim, n_starts=n_starts, seed=seed)["BIC"]
    return row


def draw_matrix(ax, M, title, cmap):
    ax.imshow(M, vmin=0, vmax=1, cmap=cmap)
    for (a, b), v in np.ndenumerate(M):
        ax.text(b, a, f"{v:.2f}", ha="center", va="center", color="white" if v >= 0.5 else "black")
    names = [MODEL_NAMES[m] for m in [1, 2, 3, 4]]
    ax.set_xticks(range(4), names, rotation=35, ha="right")
    ax.set_yticks(range(4), names)
    ax.set_xlabel("fit model")
    ax.set_title(title)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-starts", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    subs = load_subjects(ROOT / "data" / "dataconf.csv")
    sub_list = list(subs.values())
    fits = pd.read_csv(ROOT / "results" / "model_fits.csv")

    jobs = []
    rng = np.random.default_rng(11)
    for m in [1, 2, 3, 4]:
        Xfit = fits[fits.model == m][PARAM_NAMES].to_numpy()
        for i, x in enumerate(Xfit):
            trials = sub_list[rng.integers(len(sub_list))]
            jobs.append((m, i, x, trials, 5000 + 100 * m + i, args.n_starts))

    with ProcessPoolExecutor(args.workers) as ex:
        df = pd.DataFrame(list(ex.map(_job, jobs)))
    df.to_csv(ROOT / "results" / "model_recovery.csv", index=False)

    bic = df[[f"BIC_{m}" for m in [1, 2, 3, 4]]].to_numpy()
    CM = np.zeros((4, 4))
    for row, b in zip(df.sim_model, bic):
        best = b == b.min()
        CM[row - 1] += best / best.sum()
    p_fit_given_sim = CM / CM.sum(1, keepdims=True)
    p_sim_given_fit = CM / np.maximum(CM.sum(0, keepdims=True), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    draw_matrix(axes[0], p_fit_given_sim, "confusion matrix  p(fit | sim)", "Reds")
    draw_matrix(axes[1], p_sim_given_fit, "inversion matrix  p(sim | fit)", "Blues")
    axes[0].set_ylabel("simulated model")
    axes[1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "model_recovery.png", dpi=200)
    names = [MODEL_NAMES[m] for m in [1, 2, 3, 4]]
    print("p(fit | sim):")
    print(pd.DataFrame(p_fit_given_sim, index=names, columns=names).round(2))


if __name__ == "__main__":
    main()
