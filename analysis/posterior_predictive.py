#!/usr/bin/env python3
"""Posterior predictive checks for the fitted models (paper Figures 11 and S13).

Each participant's own trials are re-simulated 20 times with that participant's maximum-likelihood
parameters for every model, and the simulated data are summarised exactly like the real data.

Two checks:
  1. Behavioral signature: participant-averaged response error in 20-degree feedback-offset bins.
  2. Bimodality (the signature that separates Hybrid from Cue Combination). For feedback trials
     with a moderate offset (30-120 deg) we compute how far each response moved toward the visual
     glimpse:  z = (error - mu_PI) / offset, where mu_PI is the participant's predicted
     path-integration error for that target (from the Hybrid fit). z ~ 0 means "ignored the glimpse",
     z ~ 1 means "followed it". Hybrid predicts a bimodal z distribution (switching); Cue
     Combination predicts intermediate values (averaging).

Requires results/model_fits.csv.

Outputs
    figures/ppc_signature.png          data vs all four models, binned signature + z histograms
    figures/ppc_hybrid_participants.png  every participant: data vs Hybrid simulation
    results/ppc_summary.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heading_models import (MODEL_NAMES, PARAM_NAMES, binned_signature, compute_kalman,
                            compute_no_feedback, compute_p_true, load_subjects, simulate)

ROOT = Path(__file__).resolve().parents[1]
EDGES = np.arange(-180, 181, 20)
N_REP = 20
COLORS = {1: "#7a7a7a", 2: "#3b7dd8", 3: "#e0a030", 4: "#c8384a"}
Z_RANGE = (30, 120)


def fb_arrays(t):
    f = t.fb_flag
    return t.fb[f] - t.fb_true[f], t.respond[f] - t.target[f], t.target[f]


def z_scores(offset, error, target, X_hyb):
    keep = (np.abs(offset) >= Z_RANGE[0]) & (np.abs(offset) <= Z_RANGE[1])
    mu_pi, _ = compute_no_feedback(X_hyb, target[keep])
    return (error[keep] - mu_pi) / offset[keep]


def main() -> None:
    subs = load_subjects(ROOT / "data" / "dataconf.csv")
    fits = pd.read_csv(ROOT / "results" / "model_fits.csv")
    X = {(r.subj, r.model): r[PARAM_NAMES].to_numpy(float) for _, r in fits.iterrows()}
    rng = np.random.default_rng(7)

    data = {"off": [], "err": [], "g": [], "z": []}
    sims = {m: {"off": [], "err": [], "g": [], "z": []} for m in [1, 2, 3, 4]}
    for s, t in subs.items():
        off, err, tar = fb_arrays(t)
        data["off"].append(off); data["err"].append(err); data["g"].append(np.full(len(off), s))
        data["z"].append(z_scores(off, err, tar, X[(s, 4)]))
        for m in [1, 2, 3, 4]:
            for rep in range(N_REP):
                sim = simulate(m, X[(s, m)], t, rng)
                o, e, ta = fb_arrays(sim)
                sims[m]["off"].append(o); sims[m]["err"].append(e)
                sims[m]["g"].append(np.full(len(o), f"{s}_{rep}"))
                sims[m]["z"].append(z_scores(o, e, ta, X[(s, 4)]))
    data = {k: np.concatenate(v) for k, v in data.items()}
    for m in sims:
        sims[m] = {k: np.concatenate(v) for k, v in sims[m].items()}

    obs = binned_signature(data["off"], data["err"], data["g"], EDGES)

    # ---- Figure: signature + bimodality for each model ----
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    zbins = np.linspace(-1, 2, 31)
    rows = []
    for i, m in enumerate([1, 2, 3, 4]):
        ax = axes[0, i]
        sm = binned_signature(sims[m]["off"], sims[m]["err"], sims[m]["g"], EDGES)
        ax.errorbar(obs["mid"], obs["mean"], yerr=obs["sem"], fmt="o", color="k", ms=4, capsize=2,
                    label="data (mean ± SEM)")
        ax.plot(sm["mid"], sm["mean"], "-", color=COLORS[m], lw=2.5, label=f"{MODEL_NAMES[m]} sim")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-30, 70)
        ax.set_title(MODEL_NAMES[m])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("feedback offset (deg)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        rmse = np.sqrt(np.mean((sm.set_index("mid")["mean"] - obs.set_index("mid")["mean"]) ** 2))

        ax = axes[1, i]
        ax.hist(np.clip(data["z"], -1, 2), bins=zbins, density=True, color="#cccccc", label="data")
        ax.hist(np.clip(sims[m]["z"], -1, 2), bins=zbins, density=True, histtype="step", lw=2.2,
                color=COLORS[m], label=f"{MODEL_NAMES[m]} sim")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.axvline(1, color="k", lw=0.5, ls=":")
        ax.set_xlabel("shift toward glimpse, z  (0 = ignore, 1 = follow)")
        ax.legend(frameon=False, fontsize=8)
        frac_d = np.mean(np.abs(data["z"] - 0.5) < 0.25)
        frac_s = np.mean(np.abs(sims[m]["z"] - 0.5) < 0.25)
        rows.append({"model": MODEL_NAMES[m], "signature_RMSE_deg": rmse,
                     "frac_intermediate_z_data": frac_d, "frac_intermediate_z_sim": frac_s,
                     "frac_follow_z_data": np.mean(data["z"] > 0.75),
                     "frac_follow_z_sim": np.mean(sims[m]["z"] > 0.75)})
    axes[0, 0].set_ylabel("response error (deg)")
    axes[1, 0].set_ylabel(f"density (trials with {Z_RANGE[0]}°≤|offset|≤{Z_RANGE[1]}°)")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Posterior predictive check: each participant re-simulated 20x with their fitted parameters")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "ppc_signature.png", dpi=200)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "results" / "ppc_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    # ---- Figure S13 analogue: every participant, data vs Hybrid branches ----
    bic = fits.pivot(index="subj", columns="model", values="BIC")
    winner = bic.idxmin(axis=1)
    fig, axes = plt.subplots(5, 6, figsize=(15, 12.5), sharex=True, sharey=True)
    for ax, (s, t) in zip(axes.flat, subs.items()):
        off, err, tar = fb_arrays(t)
        Xh = X[(s, 4)]
        f = t.fb_flag
        p = compute_p_true(Xh, t.fb[f], t.fb_true[f])
        m0, _ = compute_no_feedback(Xh, tar)
        m1, _, _ = compute_kalman(Xh, tar, t.fb[f], t.fb_true[f])
        ax.scatter(off, err, s=6, facecolors="none", edgecolors="#999999", lw=0.5)
        ax.scatter(off, m1, s=12 * p + 0.2, color=COLORS[4])
        ax.scatter(off, m0, s=12 * (1 - p) + 0.2, color=COLORS[2])
        if winner[s] != 4:
            ax.set_facecolor("#fff3d6" if winner[s] == 3 else "#e8f1e0")
        ax.set_title(f"{s} (best: {MODEL_NAMES[winner[s]]})", fontsize=8)
        ax.set_xlim(-200, 200); ax.set_ylim(-200, 200)
        ax.set_xticks([-180, 0, 180]); ax.set_yticks([-180, 0, 180])
    fig.supxlabel("feedback offset (deg)")
    fig.supylabel("response error (deg)")
    fig.suptitle("Hybrid fit per participant: grey = data, red = visual branch, blue = path-integration "
                 "branch (marker size = choice probability)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "ppc_hybrid_participants.png", dpi=170)
    print("Saved figures/ppc_signature.png and figures/ppc_hybrid_participants.png")


if __name__ == "__main__":
    main()
