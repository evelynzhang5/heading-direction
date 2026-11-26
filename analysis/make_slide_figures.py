#!/usr/bin/env python3
"""Compact, slide-sized versions of the main figures (deck palette, sized to the 4.3" slide cards).

Requires the results/ files written by the other analysis scripts.
Outputs figures/slides/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heading_models import (PARAM_NAMES, compute_averaging, compute_kalman, compute_no_feedback,
                            compute_p_true, load_subjects, simulate)
from posterior_predictive import Z_RANGE, fb_arrays, z_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "slides"
NAVY, INK, GREY, ORANGE, LIGHT = "#1C2D4A", "#282D34", "#6B7280", "#CC7E2D", "#9CA3AF"
MODEL_COL = {1: LIGHT, 2: "#6F9BC4", 3: NAVY, 4: ORANGE}
SHORT = {1: "Path Int.", 2: "Kalman", 3: "Cue Comb.", 4: "Hybrid"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY, "text.color": INK, "axes.titlecolor": NAVY,
    "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 7,
    "savefig.dpi": 300, "savefig.transparent": True,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)


def fig_simulate():
    X = np.array([1, 0, 1, 10, 1, 20, 0.1, 20, 0.8], dtype=float)
    off = np.arange(-180, 181, dtype=float)
    tar, fbt = np.full_like(off, 180.0), np.zeros_like(off)
    fb = fbt + off
    fig, axes = plt.subplots(2, 2, figsize=(4.15, 2.75), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, [1, 2, 3, 4]):
        if m == 4:
            p = compute_p_true(X, fb, fbt)
            m0, _ = compute_no_feedback(X, tar)
            m1, _, _ = compute_kalman(X, tar, fb, fbt)
            st = slice(None, None, 7)
            ax.scatter(off[st], m1[st], s=22 * p[st] + 0.5, color=ORANGE, lw=0, label="follow glimpse")
            ax.scatter(off[st], m0[st], s=22 * (1 - p[st]) + 0.5, color=NAVY, lw=0, label="ignore glimpse")
            ax.legend(frameon=False, loc="upper left", handletextpad=0.1, borderaxespad=0.1, markerscale=0.8)
        else:
            if m == 1:
                mean, _ = compute_no_feedback(X, tar)
            elif m == 2:
                mean, _, _ = compute_kalman(X, tar, fb, fbt)
            else:
                mean, _, _ = compute_averaging(X, tar, fb, fbt)
            ax.plot(off, mean, color=MODEL_COL[m] if m != 1 else GREY, lw=2)
        ax.axhline(0, color=GREY, lw=0.4)
        ax.set_title(["Path Integration", "Kalman Filter", "Cue Combination", "Hybrid"][m - 1], pad=2)
        ax.set_ylim(-100, 100)
        ax.set_xticks([-180, 0, 180])
        ax.set_yticks([-100, 0, 100])
    fig.supxlabel("feedback offset (deg)", fontsize=8, y=0.02)
    fig.supylabel("response error (deg)", fontsize=8, x=0.01)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
    save(fig, "s7_model_predictions.png")


def fig_observed():
    from heading_models import binned_signature
    subs = load_subjects(ROOT / "data" / "dataconf.csv")
    off, err, g = [], [], []
    for sname, t in subs.items():
        o, e, _ = fb_arrays(t)
        off.append(o); err.append(e); g.append(np.full(len(o), sname))
    summ = binned_signature(np.concatenate(off), np.concatenate(err), np.concatenate(g), np.arange(-180, 181, 20))
    fig, ax = plt.subplots(figsize=(4.15, 1.6))
    ax.errorbar(summ["mid"], summ["mean"], yerr=summ["sem"], fmt="o-", color=ORANGE, ms=3, lw=1.4,
                capsize=1.5, elinewidth=0.8)
    ax.axhline(0, color=GREY, lw=0.4)
    ax.axvline(0, color=GREY, lw=0.4, ls=":")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("feedback offset (deg)", labelpad=1)
    ax.set_ylabel("error (deg)")
    ax.set_title("Observed data: 8,042 feedback trials, mean ± SEM over 30 participants", pad=3)
    fig.tight_layout(pad=0.3)
    save(fig, "s7_observed.png")


def fig_param_recovery():
    df = pd.read_csv(ROOT / "results" / "parameter_recovery.csv")
    d = df[df.model == 4]
    labels = {"g": "target gain", "b": "target bias", "gamma": "velocity gain", "sigma_mem": "target noise",
              "sigma_vel": "velocity noise", "s_0": "initial uncert. s0", "s_vel": "belief: vel. noise",
              "s_f": "belief: fb noise", "omega": "prior valid fb, r"}
    r = {p: np.corrcoef(d[f"true_{p}"], d[f"fit_{p}"])[0, 1] for p in PARAM_NAMES}
    order = sorted(PARAM_NAMES, key=lambda p: r[p])
    fig, ax = plt.subplots(figsize=(4.0, 1.95))
    cols = [ORANGE if p == "omega" else NAVY for p in order]
    ax.barh(range(9), [r[p] for p in order], color=cols, height=0.7)
    for i, p in enumerate(order):
        ax.text(r[p] + 0.01, i, f"{r[p]:.2f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(9), [labels[p] for p in order], fontsize=6.5)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("r (simulated vs recovered), Hybrid, 30 synthetic participants", fontsize=7)
    ax.axvline(0.8, color=GREY, lw=0.5, ls=":")
    fig.tight_layout(pad=0.3)
    save(fig, "s8_parameter_recovery.png")


def fig_model_recovery():
    df = pd.read_csv(ROOT / "results" / "model_recovery.csv")
    bic = df[[f"BIC_{m}" for m in [1, 2, 3, 4]]].to_numpy()
    CM = np.zeros((4, 4))
    for row, b in zip(df.sim_model, bic):
        best = b == b.min()
        CM[row - 1] += best / best.sum()
    P = CM / CM.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.0, 1.95))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("deck", ["#FFFFFF", "#F2D3B0", ORANGE])
    ax.imshow(P, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for (a, b), v in np.ndenumerate(P):
        ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if v >= 0.6 else INK, fontweight="bold" if a == b else "normal")
    names = [SHORT[m] for m in [1, 2, 3, 4]]
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(4), names)
    ax.set_xlabel("best-fitting model (lowest BIC)")
    ax.set_ylabel("simulated model")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.3)
    save(fig, "s8_model_recovery.png")


def fig_ppc():
    subs = load_subjects(ROOT / "data" / "dataconf.csv")
    fits = pd.read_csv(ROOT / "results" / "model_fits.csv")
    X = {(r.subj, r.model): r[PARAM_NAMES].to_numpy(float) for _, r in fits.iterrows()}
    rng = np.random.default_rng(7)
    zd, zs = [], {3: [], 4: []}
    for s, t in subs.items():
        off, err, tar = fb_arrays(t)
        zd.append(z_scores(off, err, tar, X[(s, 4)]))
        for m in (3, 4):
            for _ in range(20):
                o, e, ta = fb_arrays(simulate(m, X[(s, m)], t, rng))
                zs[m].append(z_scores(o, e, ta, X[(s, 4)]))
    zd = np.concatenate(zd)
    bins = np.linspace(-0.75, 1.75, 26)
    fig, ax = plt.subplots(figsize=(4.0, 1.9))
    ax.hist(np.clip(zd, -0.75, 1.75), bins=bins, density=True, color="#D9DEE5", label="data")
    for m in (3, 4):
        ax.hist(np.clip(np.concatenate(zs[m]), -0.75, 1.75), bins=bins, density=True, histtype="step",
                lw=1.8, color=MODEL_COL[m], label={3: "Cue Combination", 4: "Hybrid"}[m])
    ax.axvline(0, color=GREY, lw=0.5, ls=":")
    ax.axvline(1, color=GREY, lw=0.5, ls=":")
    ax.text(0, -0.13, "ignore", fontsize=6.5, color=GREY, ha="center", va="top",
            transform=ax.get_xaxis_transform())
    ax.text(1, -0.13, "follow", fontsize=6.5, color=GREY, ha="center", va="top",
            transform=ax.get_xaxis_transform())
    ax.set_xticks([-0.5, 0.5, 1.5])
    ax.set_xlabel(f"shift toward the glimpse (trials with {Z_RANGE[0]}–{Z_RANGE[1]}° offset)", labelpad=8)
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout(pad=0.3)
    save(fig, "s9_ppc.png")


def fig_model_comparison():
    fits = pd.read_csv(ROOT / "results" / "model_fits.csv")
    bic = fits.pivot(index="subj", columns="model", values="BIC_paper_k")
    delta = bic.sub(bic[4], axis=0)
    win = bic.idxmin(axis=1).value_counts().reindex([1, 2, 3, 4], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(4.0, 1.75), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, m in enumerate([1, 2, 3]):
        y = np.full(len(delta), i) + rng.uniform(-0.18, 0.18, len(delta))
        ax.scatter(delta[m].clip(upper=400), y, s=9, facecolors="none", edgecolors=MODEL_COL[m], lw=0.9)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(3), [SHORT[m] for m in [1, 2, 3]])
    ax.set_xlim(-60, 410)
    ax.invert_yaxis()
    ax.set_xlabel("BIC(model) − BIC(Hybrid)")
    ax.set_title("each dot = 1 participant", fontsize=7, color=GREY)
    ax = axes[1]
    ax.barh(range(4), win.values, color=[MODEL_COL[m] for m in [1, 2, 3, 4]], height=0.7)
    for i, c in enumerate(win.values):
        ax.text(c + 0.5, i, str(c), va="center", fontsize=7.5, fontweight="bold" if i == 3 else "normal")
    ax.set_yticks(range(4), [SHORT[m] for m in [1, 2, 3, 4]])
    ax.set_xlim(0, 33)
    ax.invert_yaxis()
    ax.set_xlabel("participants best fit")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, "s10_model_comparison.png")


def fig_extension():
    from extension_reliability import RHOS, expit, logit
    subs = load_subjects(ROOT / "data" / "dataconf.csv")
    fits = pd.read_csv(ROOT / "results" / "model_fits.csv")
    Xh = {r.subj: r[PARAM_NAMES].to_numpy(float) for _, r in fits[fits.model == 4].iterrows()}
    rho0 = float(pd.read_csv(ROOT / "results" / "extension_offset_mixture.csv")["rho_original"].iloc[0])
    refits = pd.read_csv(ROOT / "results" / "extension_refits.csv")
    offs = np.arange(0, 181, 1.0)
    med = float(np.median(np.concatenate([t.fb_true[t.fb_flag] for t in subs.values()])))
    blues = ["#F2D3B0", "#E3A764", ORANGE, "#8A4E14"]

    def curve(rho, adaptive):
        ps = []
        for s in subs:
            X = Xh[s].copy()
            if adaptive:
                X[8] = expit(logit(X[8]) + logit(rho) - logit(rho0))
            f = np.full_like(offs, med)
            ps.append(compute_p_true(X, X[2] * f + offs, f))
        return np.mean(ps, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(4.15, 2.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for c, rho in zip(blues, RHOS):
        ax.plot(offs, curve(rho, True), color=c, lw=1.8, label=f"ρ = {rho:.2f}")
    ax.plot(offs, curve(0.5, False), color=NAVY, lw=1.2, ls="--", label="fixed prior\n(any ρ)")
    ax.axhline(0.5, color=GREY, lw=0.4, ls=":")
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 60, 120, 180])
    ax.set_xlabel("|glimpse − body estimate| (deg)")
    ax.set_ylabel("P(follow glimpse)")
    ax.set_title("adaptive-prior prediction", pad=2)
    ax.legend(frameon=False, fontsize=6, loc="upper right", handlelength=1.2, labelspacing=0.25)
    ax = axes[1]
    for hyp, col, mk in [("ADAPTIVE", ORANGE, "o"), ("FIXED", NAVY, "s")]:
        d = refits[refits.hypothesis == hyp].groupby("rho")["omega_fit"]
        med_, q1, q3 = d.median(), d.quantile(0.25), d.quantile(0.75)
        x = np.arange(4) + (0.08 if hyp == "FIXED" else -0.08)
        ax.errorbar(x, med_.values, yerr=[med_.values - q1.values, q3.values - med_.values], color=col,
                    marker=mk, ms=3.5, lw=1.4, capsize=1.5, label=hyp.lower())
    ax.set_xticks(range(4), [f"{r:.2f}" for r in RHOS])
    ax.set_xlabel("block reliability ρ")
    ax.set_ylabel("recovered prior r")
    ax.set_ylim(0, 1.02)
    ax.set_title("refit of simulated data", pad=2)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, "s11_extension.png")


if __name__ == "__main__":
    fig_simulate()
    fig_observed()
    fig_param_recovery()
    fig_model_recovery()
    fig_ppc()
    fig_model_comparison()
    fig_extension()
    print(f"Saved slide figures to {OUT}")
