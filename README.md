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

## Repository layout

- `analysis/plot_behavioral_signature.py`: first-pass empirical behavioral-signature plot from the authors' open `dataconf.csv`.
- `data/`: place the authors' `dataconf.csv` here locally; the raw data are not duplicated in this repository.

## Setup and first plot

Download `dataconf.csv` from the [authors' repository](https://github.com/sharootonian/CombinationAndCompetitionHeadingDirection/tree/main/data) and save it as `data/dataconf.csv`. Then run:

```bash
python -m pip install pandas matplotlib
python analysis/plot_behavioral_signature.py
```

The script writes `figures/behavioral_signature.png`. It expects the column names used by the authors' MATLAB loader: `subj`, `condition`, `error`, and `fb_offset`.

## Status / next steps

- [x] Select paper and identify the original code/data source.
- [x] Add a first-pass script for the empirical behavioral signature.
- [ ] Download and verify the data locally.
- [ ] Inspect the source MATLAB fitting pipeline and parameter bounds.
- [ ] Reproduce parameter recovery and model recovery.
- [ ] Fit the four models and compare participant-level BIC.
- [ ] Run posterior predictive checks.
- [ ] Implement the visual-reliability simulation extension.

## References

Harootonian, S. K., Ekstrom, A. D., & Wilson, R. C. (2022). Combination and competition between path integration and landmark navigation in the estimation of heading direction. *PLOS Computational Biology, 18*(2), e1009222. https://doi.org/10.1371/journal.pcbi.1009222
