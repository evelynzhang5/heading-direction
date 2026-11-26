# Heading Direction: Cue Combination and Competition

Option 2 project reproducing Harootonian, Ekstrom, & Wilson (2022), “Combination and competition between path integration and landmark navigation in the estimation of heading direction.”

Paper: https://doi.org/10.1371/journal.pcbi.1009222  
Original MATLAB code and data: https://github.com/sharootonian/CombinationAndCompetitionHeadingDirection  
Data DOI: https://doi.org/10.5281/zenodo.5879723

## Research question

How do people combine body-based path-integration information and visual landmark information when estimating heading direction, and how does the mismatch between these cues affect their reliance on visual feedback?

This matters because successful navigation requires people (and autonomous systems) to combine noisy, sometimes conflicting sources of information to maintain an estimate of orientation.

## Original task

Participants wore an HTC Vive headset in a virtual room. On each trial they were guided through an initial rotation, then asked to turn back toward their remembered starting direction. In the feedback condition, they briefly saw the room for 300 ms during the return turn; this visual feedback could be consistent or misleading. They reported their final heading and confidence. The published study analyzed 30 participants, each with 100 no-feedback and 300 feedback trials (after exclusions).

The key behavioral signature is response error (final heading minus target angle) as a function of feedback offset (visual feedback angle minus the participant's path-integration heading estimate at the time of feedback).

## Competing models

1. **Path Integration:** ignores visual feedback and relies on body-based movement cues.
2. **Kalman Filter:** always incorporates visual feedback, weighting it against the current path-integration estimate according to their relative uncertainty.
3. **Cue Combination:** averages the path-integration and visual-updated estimates, weighted by the inferred probability that feedback is valid.
4. **Hybrid:** samples which estimate to use: the visual-updated estimate when feedback is judged valid, or the path-integration estimate when it is judged invalid.

The models predict distinct response-error-versus-feedback-offset patterns. The paper reports that the Hybrid model was the best-fitting model for 28 of 30 participants (Figure 10).

## Replication target

Reproduce the paper's model comparison: fit the four models to each participant, calculate BIC, and test whether the Hybrid model is selected for approximately 28/30 participants. Also reproduce the empirical feedback-offset/response-error behavioral signature from the open data.

## Planned extension

Vary the objective reliability of visual landmark feedback in simulations (the probability that a visual glimpse is generated from the true heading rather than from a misleading distribution). Test how the Hybrid agent's response-error curve and probability of following visual feedback change as landmark reliability changes. This extends the original fixed-reliability task without requiring new participant data.

## Reproduce everything

```bash
python -m pip install numpy scipy pandas matplotlib
./run_all.sh          # downloads data/dataconf.csv if missing; ~15 min on 12 cores
```

| Script (`analysis/`) | Requirement | Outputs |
| --- | --- | --- |
| `heading_models.py` | — | Python port of the authors' MATLAB models, likelihoods, simulator and fitter |
| `plot_behavioral_signature.py` | Simulate (observed signature) | `figures/behavioral_signature.png` |
| `simulate_models.py` | Simulate | `figures/model_predictions.png` (paper Fig. 6) |
| `fit_models.py` | Fit and compare | `results/model_fits.csv`, `figures/model_comparison.png` (Fig. 10), `figures/model_comparison_code_k.png` |
| `posterior_predictive.py` | Fit and compare (PPC) | `figures/ppc_signature.png`, `figures/ppc_hybrid_participants.png` (Figs. 11, S13), `results/ppc_summary.csv` |
| `parameter_recovery.py` | Recover | `figures/parameter_recovery_hybrid.png` (Fig. S10), `figures/parameter_recovery_summary.png` |
| `model_recovery.py` | Recover | `figures/model_recovery.png` (Fig. S4), `results/model_recovery.csv` |
| `extension_reliability.py` | Extend | `figures/extension_reliability.png`, `figures/extension_offset_mixture.png`, `results/extension_*.csv` |

### Methods matched to the authors' code

The model equations, parameter bounds and fitting procedure follow `main.m` and `scripts/*_v1.m` in the
authors' repository:

- **Data:** `dataconf.csv`, 30 participants, 8,042 feedback and 2,709 no-feedback trials (after the authors' exclusions).
- **Joint fit:** every model is fit to both conditions together (`data_flag = 3`); no-feedback trials are always explained by the path-integration model.
- **Bounds:** g ∈ [0, 2], b ∈ [−180, 180], γ ∈ [0, 4], σ_A, σ_d, s_0, s_d ∈ [0, 20], s_f ∈ [0, 50], r ∈ [0, 1].
- **Fixed parameters:** Path Integration fixes γ = 1; the Kalman Filter fixes s_f = 1.
- **Optimizer:** maximum likelihood with 100 random starts per fit, using bounded L-BFGS-B in place of MATLAB's `fmincon`.
- **Recovery design:** parameter recovery scrambles fitted parameters across participants (type 3, rng 10). Model recovery simulates from each participant's own fitted parameters (type 4). Both use 30 simulated participants on real trial structures.
- **Hybrid likelihood:** computed as `logaddexp` of the two branches. The authors use a 1e-10 threshold to avoid underflow; the two are equivalent except in that underflow case.

## Results

### Simulate
The four models make different predictions about response error as a function of feedback offset
(`figures/model_predictions.png`):

- **Path Integration** predicts a flat line.
- **Kalman Filter** predicts a line with slope K.
- **Cue Combination** predicts an S-curve that returns to 0 for large offsets.
- **Hybrid** predicts the same S-curve *on average*, but each trial lands on one of two branches: it either follows the glimpse or ignores it.

Binned means therefore cannot separate Cue Combination from Hybrid; the trial-level distribution can.

### Recover
- **Hybrid parameters recover well:** r = 0.80–0.93 for 8 of 9 parameters. The exception is the velocity gain γ_d (r = 0.51), because it trades off with the target gain g (the paper's Fig. S12).
- **Model recovery with BIC** (p(fit | sim) on the diagonal): Path Integration 0.97, Kalman Filter 0.47, Cue Combination 0.63, Hybrid 0.80.
- **Hybrid labels are trustworthy:** when BIC picks Hybrid, the data came from Hybrid 83% of the time, and never from the Kalman Filter.
- **The Kalman Filter is the weak spot:** its simulated datasets are often labeled Path Integration (0.53). The Kalman gains fitted to real participants are small because people ignore large offsets, so a Kalman Filter with those gains behaves almost like Path Integration.

### Fit and compare
Summed BIC is lowest for Hybrid (93,202), then Cue Combination (94,393), Kalman Filter (96,716) and Path
Integration (97,307). The median per-participant ΔBIC relative to Hybrid is +12.5 for Cue Combination, +88 for the Kalman Filter and +86 for Path Integration.

**Posterior predictive check** (`figures/ppc_signature.png`): Cue Combination tracks the binned means
slightly better (RMSE 3.8° vs 5.7°). Only Hybrid reproduces the trials on which participants *fully* followed the glimpse. Among trials with 30–120° offset, 21.9% of real responses moved at least 75% of the way to the glimpse. Hybrid predicts 22.0%, Cue Combination 17.6%, the Kalman Filter 10.4% and Path Integration 7.1%.

### Replicate
| Parameter-count convention | Hybrid | Cue Comb. | Path Int. | Kalman |
| --- | --- | --- | --- | --- |
| Paper text (k = 5/8/9/9) | **27** | 2 (s25, s32) | 1 (s6) | 0 |
| Authors' code (k = 4/7/9/9) | 25 | 2 (s25, s32) | 3 (s1, s6, s24) | 0 |

**The main result replicates: Hybrid is the best model for 27 of 30 participants.** The paper's Figure 10 says 28/30, but its prose lists three exceptions (one Path Integration, two Cue Combination). Using the paper's parameter counts, we recover exactly those three exceptions, which suggests the "28" is a typo.

The code's parameter counts move two more participants (s1, s24) to Path Integration. Both are near-ties (ΔBIC = −2.1). Two other differences could also matter: our optimizer is L-BFGS-B rather than `fmincon`, and the random starts differ.

### Extend: manipulating glimpse reliability
In the original task the glimpse offsets are a mixture: 69% informative glimpses (Gaussian, σ = 28°) and 31% uniform misleading glimpses (`figures/extension_offset_mixture.png`).

**Simulated experiment.** We simulated a new design with four reliability blocks (ρ = 0.25, 0.50, 0.75, 0.95). Each participant was represented by their own fitted Hybrid agent, with their own targets and turn times. We compared two hypotheses about the participant's prior r that a glimpse is valid:

- **FIXED:** r is unchanged across blocks.
- **ADAPTIVE:** r shifts in log-odds with ρ.

**Predictions:**

- **FIXED:** The binned response-error curves sit on top of each other in every block. The offset at which P(follow glimpse) drops below 0.5 stays at 42°.
- **ADAPTIVE:** As ρ rises from 0.25 to 0.95, the switch point moves from 12° to 63°. The peak of the S-curve rises from about 25° to about 45°. Participants follow glimpses with large offsets (60–180°) 5.7% of the time at ρ = 0.25, rising to 26.9% at ρ = 0.95.

**The design can distinguish the two hypotheses.** We refit the Hybrid model to every simulated block.

- **ADAPTIVE:** The recovered r tracks reliability (Friedman χ² = 42.6, p = 3×10⁻⁹; median within-participant Spearman = 1.0).
- **FIXED:** The recovered r does not change (χ² = 3.3, p = 0.34).

At ρ = 0.25 the recovered r is biased upward (median 0.57 vs. true 0.40), because few glimpses are informative in that block.

## Limitations

- **Small exception set.** The replication of the winner counts depends on the parameter-count convention, and the exceptions are only 3–5 near-tie participants.
- **Simulated extension.** The extension is a simulation of a proposed experiment, not new data. The ADAPTIVE rule (shifting the prior in log-odds) is one assumption among several possible ones.
- **Kalman Filter identifiability.** In model recovery, the Kalman Filter is hard to tell apart from Path Integration at the participants' fitted parameters.
- **Model scope.** The models ignore angle wrapping, trial history and learning within the session.
- **Model-recovery BIC convention.** Model recovery uses the code's parameter counts (4/7/9/9).

## Status

- [x] Download and verify the data (`dataconf.csv`, 30 participants, 10,751 trials).
- [x] Port the authors' MATLAB fitting pipeline and bounds.
- [x] Simulate the four models and their behavioral signatures.
- [x] Parameter recovery and model recovery.
- [x] Fit the four models and compare participant-level BIC.
- [x] Posterior predictive checks.
- [x] Visual-reliability extension (simulation and recoverability analysis).

## References

Harootonian, S. K., Ekstrom, A. D., & Wilson, R. C. (2022). Combination and competition between path integration and landmark navigation in the estimation of heading direction. *PLOS Computational Biology, 18*(2), e1009222. https://doi.org/10.1371/journal.pcbi.1009222
