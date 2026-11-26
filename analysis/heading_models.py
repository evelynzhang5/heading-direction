"""Python port of the four heading-direction models of Harootonian, Ekstrom & Wilson (2022).

Each function mirrors a file in the authors' MATLAB repository
(https://github.com/sharootonian/CombinationAndCompetitionHeadingDirection/tree/main/scripts):

    compute_no_feedback      <- compute_noFeedback_v1.m
    compute_kalman           <- compute_pureKalmanFilter_v1.m
    compute_p_true           <- compute_pTrue_v1.m
    compute_averaging        <- compute_averaging_v1.m
    loglik_* / neg_loglik    <- lik_*_v1.m + fit_any_parallel_v1.m (data_flag = 3)
    simulate                 <- sim_any_v2.m
    fit_subject              <- fit_any_parallel_v1.m

All angles are in degrees and, as in the original code, are not wrapped.

Parameter vector X (9 entries, same order as the MATLAB code):
    0 g          target gain (gamma_A): gain on the remembered target angle
    1 b          target bias (beta_A)
    2 gamma      velocity gain (gamma_d): gain on self-motion (path-integration) cues
    3 sigma_mem  target-memory noise (sigma_A)
    4 sigma_vel  velocity noise (sigma_d); PI variance grows with angle turned
    5 s_0        participant's belief about initial uncertainty
    6 s_vel      participant's belief about velocity noise
    7 s_f        participant's belief about feedback noise
    8 omega      participant's prior that a visual glimpse is valid (r)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

PARAM_NAMES = ["g", "b", "gamma", "sigma_mem", "sigma_vel", "s_0", "s_vel", "s_f", "omega"]
PARAM_LABELS = [
    "target gain $\\gamma_A$", "target bias $\\beta_A$", "velocity gain $\\gamma_d$",
    "target noise $\\sigma_A$", "velocity noise $\\sigma_d$", "initial uncertainty $s_0$",
    "believed velocity noise $s_d$", "believed feedback noise $s_f$", "prior valid feedback $r$",
]
MODEL_NAMES = {1: "Path Integration", 2: "Kalman Filter", 3: "Cue Combination", 4: "Hybrid"}

# Bounds and default start from fit_any_parallel_v1.m (the version main.m uses).
X0 = np.array([1, 0, 1, 1, 1, 2, 10, 1, 0.1], dtype=float)
LB = np.array([0, -180, 0, 0, 0, 0, 0, 0, 0], dtype=float)
UB = np.array([2, 180, 4, 20, 20, 20, 20, 50, 1], dtype=float)

# Which parameters are free, which are fixed, and the k used in BIC (fit_any_parallel_v1.m).
#   PI: gamma fixed to 1 (g and gamma are not separately identifiable without feedback).
#   KF: s_f fixed to 1 (only the ratio of s_f to s_0/s_vel matters for the Kalman gain).
FREE = {
    1: [0, 1, 3, 4],
    2: [0, 1, 2, 3, 4, 5, 6],
    3: list(range(9)),
    4: list(range(9)),
}
FIXED = {1: {2: 1.0}, 2: {7: 1.0}, 3: {}, 4: {}}
N_PARAMS = {m: len(v) for m, v in FREE.items()}

VAR_FLOOR = 1e-4  # the authors' "hack to remove -ve variances"
EPS = 1e-12


@dataclass
class Trials:
    """Trial arrays for one participant (or one simulated agent)."""

    fb_flag: np.ndarray      # bool, True on feedback trials
    target: np.ndarray       # alpha, target angle
    fb: np.ndarray           # f, angle shown by the visual glimpse
    fb_true: np.ndarray      # theta_tf, true angle turned at the time of the glimpse
    respond: np.ndarray | None = None  # theta_t, final heading

    def subset(self, mask: np.ndarray) -> "Trials":
        return Trials(self.fb_flag[mask], self.target[mask], self.fb[mask], self.fb_true[mask],
                      None if self.respond is None else self.respond[mask])


def full_params(model: int, free_values: np.ndarray) -> np.ndarray:
    X = X0.copy()
    X[FREE[model]] = free_values
    for i, v in FIXED[model].items():
        X[i] = v
    return X


# ---------------------------------------------------------------------------
# Mean / variance of the response error (theta_t - alpha) under each model
# ---------------------------------------------------------------------------

def compute_no_feedback(X, target):
    g, b, gamma, s_mem, s_vel = X[:5]
    gamma = max(gamma, EPS)
    mean = ((g - gamma) * target + b) / gamma
    var = (s_vel**2 * target + s_mem**2) / gamma**2
    return mean, np.maximum(var, VAR_FLOOR)


def kalman_gain(X, fb_true):
    s_0, s_vel, s_f = X[5], X[6], X[7]
    num = s_0**2 + s_vel**2 * fb_true
    return num / np.maximum(s_f**2 + num, EPS)


def compute_kalman(X, target, fb, fb_true):
    g, b, gamma, s_mem, s_vel = X[:5]
    gamma = max(gamma, EPS)
    rho = kalman_gain(X, fb_true)
    mean = ((g - gamma) * target + b + rho * (fb - gamma * fb_true)) / gamma
    var = (s_mem**2 + s_vel**2 * (target - (2 - rho) * rho * fb_true)) / gamma**2
    var = np.where(var < 0, VAR_FLOOR, var)
    return mean, np.maximum(var, VAR_FLOOR), rho


def compute_p_true(X, fb, fb_true):
    """Posterior probability that the glimpse is valid (compute_pTrue_v1.m, kept verbatim,
    including its mix of radian and degree units)."""
    gamma, s_0, s_vel, s_f, omega = X[2], X[5], X[6], X[7], X[8]
    d2r = np.pi / 180
    total_var_deg = s_f**2 + s_0**2 + s_vel**2 * fb_true
    scale = np.sqrt(np.maximum((s_f * d2r) ** 2 + (s_0 * d2r) ** 2 + (s_vel * d2r) ** 2 * fb_true, 0))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        expo = np.exp((fb - gamma * fb_true) ** 2 / (2 * np.maximum(total_var_deg, EPS)))
        odds_invalid = scale * expo * (1 - omega) / omega
        p = 1.0 / (1.0 + odds_invalid)
    return np.nan_to_num(p, nan=0.0)


def compute_averaging(X, target, fb, fb_true):
    g, b, gamma, s_mem, s_vel = X[:5]
    gamma = max(gamma, EPS)
    p = compute_p_true(X, fb, fb_true)
    rho = kalman_gain(X, fb_true)
    w = p * rho
    mean = ((g - gamma) * target + b + w * (fb - gamma * fb_true)) / gamma
    var = (s_mem**2 + s_vel**2 * (target - (2 - w) * w * fb_true)) / gamma**2
    var = np.where(var < 0, VAR_FLOOR, var)
    return mean, np.maximum(var, VAR_FLOOR), p


def _log_normal(x, mean, var):
    return -0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var)


# ---------------------------------------------------------------------------
# Likelihood (data_flag = 3: no-feedback trials use the PI model for every model)
# ---------------------------------------------------------------------------

def trial_loglik(model: int, X: np.ndarray, t: Trials) -> np.ndarray:
    err = t.respond - t.target
    out = np.empty(len(err))
    nf = ~t.fb_flag
    m, v = compute_no_feedback(X, t.target[nf])
    out[nf] = _log_normal(err[nf], m, v)

    f = t.fb_flag
    tar, fb, fbt, e = t.target[f], t.fb[f], t.fb_true[f], err[f]
    if model == 1:
        m, v = compute_no_feedback(X, tar)
        out[f] = _log_normal(e, m, v)
    elif model == 2:
        m, v, _ = compute_kalman(X, tar, fb, fbt)
        out[f] = _log_normal(e, m, v)
    elif model == 3:
        m, v, _ = compute_averaging(X, tar, fb, fbt)
        out[f] = _log_normal(e, m, v)
    elif model == 4:
        p = compute_p_true(X, fb, fbt)
        m0, v0 = compute_no_feedback(X, tar)
        m1, v1, _ = compute_kalman(X, tar, fb, fbt)
        with np.errstate(divide="ignore"):
            lp0 = np.log1p(-np.clip(p, 0, 1)) + _log_normal(e, m0, v0)
            lp1 = np.log(np.clip(p, 0, 1)) + _log_normal(e, m1, v1)
        out[f] = np.logaddexp(lp0, lp1)
    else:
        raise ValueError(model)
    return out


def neg_loglik(model: int, X: np.ndarray, t: Trials) -> float:
    nll = -np.sum(trial_loglik(model, X, t))
    return float(nll) if np.isfinite(nll) else 1e12


# ---------------------------------------------------------------------------
# Simulation (sim_any_v2.m)
# ---------------------------------------------------------------------------

def simulate(model: int, X: np.ndarray, t: Trials, rng: np.random.Generator,
             return_choice: bool = False):
    """Return a copy of `t` with simulated responses. For the Hybrid model optionally also
    return a boolean array marking trials on which the agent followed the visual estimate."""
    n = len(t.target)
    resp = np.empty(n)
    used_visual = np.zeros(n, dtype=bool)

    nf = ~t.fb_flag
    m, v = compute_no_feedback(X, t.target[nf])
    resp[nf] = t.target[nf] + m + np.sqrt(v) * rng.standard_normal(nf.sum())

    f = t.fb_flag
    tar, fb, fbt = t.target[f], t.fb[f], t.fb_true[f]
    k = f.sum()
    if model == 1:
        m, v = compute_no_feedback(X, tar)
    elif model == 2:
        m, v, _ = compute_kalman(X, tar, fb, fbt)
        used_visual[f] = True
    elif model == 3:
        m, v, _ = compute_averaging(X, tar, fb, fbt)
    elif model == 4:
        p = compute_p_true(X, fb, fbt)
        m0, v0 = compute_no_feedback(X, tar)
        m1, v1, _ = compute_kalman(X, tar, fb, fbt)
        choose_visual = rng.random(k) < p
        m = np.where(choose_visual, m1, m0)
        v = np.where(choose_visual, v1, v0)
        used_visual[f] = choose_visual
    else:
        raise ValueError(model)
    resp[f] = tar + m + np.sqrt(v) * rng.standard_normal(k)

    out = Trials(t.fb_flag.copy(), t.target.copy(), t.fb.copy(), t.fb_true.copy(), resp)
    return (out, used_visual) if return_choice else out


# ---------------------------------------------------------------------------
# Fitting (fit_any_parallel_v1.m): fmincon from n_starts uniform random starts
# ---------------------------------------------------------------------------

def fit_subject(model: int, t: Trials, n_starts: int = 100, seed: int = 0,
                lb: np.ndarray | None = None, ub: np.ndarray | None = None) -> dict:
    """Maximum-likelihood fit with bounded multi-start L-BFGS-B. Returns full 9-parameter vector,
    nLL, BIC and AIC (BIC uses the number of trials in both conditions, as in the paper)."""
    lb = LB if lb is None else lb
    ub = UB if ub is None else ub
    free = FREE[model]
    lo, hi = lb[free].copy(), ub[free].copy()
    lo[np.array(free) == 2] = np.maximum(lo[np.array(free) == 2], 1e-3)  # keep gamma > 0
    bounds = list(zip(lo, hi))
    rng = np.random.default_rng(seed)

    def obj(z):
        return neg_loglik(model, full_params(model, z), t)

    best = None
    for s in range(n_starts):
        z0 = np.clip(X0[free], lo, hi) if s == 0 else rng.uniform(lo, hi)
        try:
            res = minimize(obj, z0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000})
        except (FloatingPointError, ValueError):
            continue
        if best is None or res.fun < best.fun:
            best = res
    n = len(t.target)
    k = N_PARAMS[model]
    nll = float(best.fun)
    return {
        "X": full_params(model, best.x),
        "nLL": nll,
        "BIC": 2 * nll + np.log(n) * k,
        "AIC": 2 * nll + 2 * k,
        "n_trials": n,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_subjects(path) -> dict[str, Trials]:
    import pandas as pd

    df = pd.read_csv(path)
    subs = {}
    for s, d in df.groupby("subj", sort=False):
        subs[s] = Trials(
            fb_flag=d["condition"].eq("FB").to_numpy(),
            target=d["target"].to_numpy(float),
            fb=d["fb"].to_numpy(float),
            fb_true=d["fb_true"].to_numpy(float),
            respond=d["respond"].to_numpy(float),
        )
    # order s1, s2, ..., s30 as in the paper
    return dict(sorted(subs.items(), key=lambda kv: int(kv[0].lstrip("s"))))


def binned_signature(offset, error, groups, edges):
    """Participant-level means per offset bin, then mean and SEM across participants."""
    import pandas as pd

    d = pd.DataFrame({"off": offset, "err": error, "g": groups})
    d["bin"] = pd.cut(d["off"], edges, include_lowest=True)
    per = d.groupby(["g", "bin"], observed=True)["err"].mean().reset_index()
    s = per.groupby("bin", observed=True)["err"].agg(["mean", "std", "count"]).reset_index()
    s["sem"] = s["std"] / np.sqrt(s["count"])
    s["mid"] = s["bin"].map(lambda iv: iv.mid).astype(float)
    return s
