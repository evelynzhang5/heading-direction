#!/usr/bin/env python3
"""Parameter recovery, following the authors' main.m (paper Figures S9-S10).

For each model: draw 30 parameter sets by scrambling the participants' fitted parameters across
participants (generate_simulationParameters type 3: keeps each parameter's empirical distribution
but removes correlations between parameters), pair each with the trial structure of a randomly
chosen real participant, simulate, and refit the generating model.

Requires results/model_fits.csv (run fit_models.py first).

Outputs
    results/parameter_recovery.csv
    figures/parameter_recovery_hybrid.png        (paper Figure S10)
    figures/parameter_recovery_summary.png       (correlation true vs recovered, all models)
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heading_models import (FREE, MODEL_NAMES, PARAM_LABELS, PARAM_NAMES, fit_subject,
                            load_subjects, simulate)

ROOT = Path(__file__).resolve().parents[1]


def scramble_params(X: np.ndarray, n: int, rng) -> np.ndarray:
    """generate_simulationParameters(3, n, Xfit): each parameter from an independent random participant."""
    return np.column_stack([X[rng.integers(0, len(X), n), j] for j in range(X.shape[1])])


def _job(args):
    model, i, x_true, trials, seed, n_starts = args
    rng = np.random.default_rng(seed)
    sim = simulate(model, x_true, trials, rng)
    r = fit_subject(model, sim, n_starts=n_starts, seed=seed)
    row = {"model": model, "sim": i}
    row.update({f"true_{p}": v for p, v in zip(PARAM_NAMES, x_true)})
    row.update({f"fit_{p}": v for p, v in zip(PARAM_NAMES, r["X"])})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=30)
    ap.add_argument("--n-starts", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    subs = load_subjects(ROOT / "data" / "dataconf.csv")
    sub_list = list(subs.values())
    fits = pd.read_csv(ROOT / "results" / "model_fits.csv")

    jobs = []
    for m in [1, 2, 3, 4]:
        rng = np.random.default_rng(10)  # rng(10) in main.m
        Xfit = fits[fits.model == m][PARAM_NAMES].to_numpy()
        Xsim = scramble_params(Xfit, args.n_sim, rng)
        for i in range(args.n_sim):
            trials = sub_list[rng.integers(len(sub_list))]
            jobs.append((m, i, Xsim[i], trials, 1000 * m + i, args.n_starts))

    with ProcessPoolExecutor(args.workers) as ex:
        df = pd.DataFrame(list(ex.map(_job, jobs)))
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "parameter_recovery.csv", index=False)

    # Figure S10 analogue: Hybrid model, all nine parameters.
    order = [2, 4, 0, 1, 3, 5, 6, 7, 8]  # par_order in main.m
    hyb = df[df.model == 4]
    fig, axes = plt.subplots(3, 3, figsize=(10, 9.5))
    for ax, j in zip(axes.flat, order):
        p = PARAM_NAMES[j]
        x, y = hyb[f"true_{p}"], hyb[f"fit_{p}"]
        ax.scatter(x, y, s=18, color="#3b7dd8", alpha=0.8)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{PARAM_LABELS[j]}\nr = {r:.2f}", fontsize=9)
        ax.set_xlabel("simulated", fontsize=8)
        ax.set_ylabel("recovered", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Parameter recovery: Hybrid model (30 simulated participants)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "parameter_recovery_hybrid.png", dpi=200)
    plt.close(fig)

    # Summary heatmap: Pearson r between simulated and recovered values, every model x parameter.
    R = np.full((4, 9), np.nan)
    for mi, m in enumerate([1, 2, 3, 4]):
        d = df[df.model == m]
        for j in FREE[m]:
            p = PARAM_NAMES[j]
            if d[f"true_{p}"].std() > 0:
                R[mi, j] = np.corrcoef(d[f"true_{p}"], d[f"fit_{p}"])[0, 1]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
    for (a, b), v in np.ndenumerate(R):
        ax.text(b, a, "fixed" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="#999999" if np.isnan(v) else ("white" if abs(v) > 0.6 else "black"))
    ax.set_xticks(range(9), PARAM_NAMES, rotation=30)
    ax.set_yticks(range(4), [MODEL_NAMES[m] for m in [1, 2, 3, 4]])
    fig.colorbar(im, ax=ax, label="r(simulated, recovered)")
    ax.set_title("Parameter recovery: correlation between simulated and recovered values")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "parameter_recovery_summary.png", dpi=200)
    plt.close(fig)

    print(pd.DataFrame(R, index=[MODEL_NAMES[m] for m in [1, 2, 3, 4]], columns=PARAM_NAMES).round(2))


if __name__ == "__main__":
    main()
