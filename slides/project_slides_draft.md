# B1–B7: Option 2 project draft

This is slide-ready wording for the heading-direction replication project. Replace bracketed fields with confirmed group details. Do not state that the data have been downloaded until you have opened the CSV locally.

## B1. Team and option (form)

- **Project option:** Option 2 — reproduce and extend a published computational model.
- **Team members:** [Add names]
- **Project mentor:** [Add mentor]

## B2. Your question (slide 4)

**Question:** How do people decide whether to combine visual landmark information with body-based path integration or rely on one source when the two disagree during heading estimation?

**Why it matters:** Navigation depends on maintaining a reliable estimate of heading from noisy information about both the body’s movement and the surrounding environment. Understanding when people integrate or disregard conflicting cues can inform theories of human spatial cognition and the design of autonomous systems that must localize under uncertainty.

**Presenter notes / links**
- Paper: https://doi.org/10.1371/journal.pcbi.1009222
- Original MATLAB code and data: https://github.com/sharootonian/CombinationAndCompetitionHeadingDirection
- Data files: https://github.com/sharootonian/CombinationAndCompetitionHeadingDirection/tree/main/data
- This project repository: https://github.com/evelynzhang5/heading-direction

**Data-access check:** [Complete after downloading and opening `dataconf.csv` locally. Record the file path, row count, and column names here.]

## B3. The original paper (slides 5–7, or the corresponding open slides in your template)

**Paper:** Harootonian, Ekstrom, & Wilson (2022), “Combination and competition between path integration and landmark navigation in the estimation of heading direction,” *PLOS Computational Biology*.

### Task

Participants completed a rotation task in virtual reality. They were guided through a turn and then had to turn back toward a remembered target heading. In feedback trials, they briefly saw the virtual room for 300 ms during the return turn; the visual feedback could be offset from the heading estimated from their body-based movement. The researchers measured participants’ final turning error and, in the dataset, confidence ratings.

### Models and comparison

The authors compared four accounts of how participants use visual feedback:

1. **Path Integration:** ignores visual feedback and relies on body-based movement cues; response error should not depend on feedback offset.
2. **Kalman Filter:** always incorporates visual feedback, weighted against the path-integration estimate; response error should change approximately linearly with feedback offset.
3. **Cue Combination:** combines the path-integration estimate and the visual-updated estimate, weighting them by the probability that the feedback is true; this predicts a nonlinear average response.
4. **Hybrid:** selects either the visual-updated estimate or the path-integration estimate on a trial, with the probability of using visual feedback depending on how plausible it is; this predicts a mixture of the two response patterns and can produce bimodal responses at large offsets.

The authors fit the models to each participant’s trial-level responses and compared them using Bayesian Information Criterion (BIC), which balances fit quality against model complexity.

### Main result and figure

**Main result:** The Hybrid model was the best-fitting model for 28 of the 30 participants. The authors interpret this as evidence that people tend to combine visual and body-based cues when the two estimates are close, but allow the cues to compete when they differ substantially.

**Figure to show:** Figure 10 (model comparison). Panel A compares each model’s BIC to the Hybrid model for each participant; Panel B counts how often each model was preferred. The Hybrid model is preferred for 28/30 participants, while Path Integration and Cue Combination each best fit one participant? [Correction: check exact panel counts in the paper figure before finalizing; text identifies participant 27 as Path Integration and participants 15 and 23 as Cue Combination, so the counts are 1, 2, and 27 if there are no other models selected.] 

## B4. Starting point (Option 2 slides 4, 11, and 12)

### Slide 11 — Replicate

**Target:** Reproduce the empirical behavioral signature in the Feedback condition: participants’ response error (final heading minus target heading) plotted against feedback offset (visual feedback angle minus the path-integration heading estimate at the time of feedback).

**Why this result:** The four models make visibly different predictions for this relationship: no dependence on feedback offset (Path Integration), a linear dependence (Kalman Filter), a nonlinear averaged response (Cue Combination), or a mixture of visual-updated and path-integration responses (Hybrid).

**First replication step:** Use the open `dataconf.csv` trial-level dataset to recreate the feedback-offset versus response-error plot. Then fit the four models and compare participant-level BIC scores, targeting the paper’s finding that Hybrid is preferred for 28/30 participants.

### Slide 12 — Extend

**Extension:** Test how the reliability of visual landmark feedback changes the balance between cue integration and cue competition.

**Plan:** Simulate agents under several levels of visual-feedback reliability while holding the path-integration noise fixed. For each reliability level, measure (1) how often the agent uses the visual-updated estimate, (2) the relationship between feedback offset and response error, and (3) the frequency of large-offset responses that remain close to the path-integration estimate.

**Prediction:** More reliable visual feedback should increase the probability of using the visual-updated estimate, including at larger mismatches. Less reliable feedback should make the agent more likely to ignore visual input and rely on path integration. This is a proposed simulation extension, not a result established by the original paper.

## B5. First result (slide 8 — Simulate)

**Figure:** Empirical response error versus feedback offset using the paper’s open Feedback-condition trials.

**Two-sentence caption:** “Each point shows a participant’s response error on a trial as a function of the discrepancy between visual feedback and the participant’s path-integration heading estimate. The binned trend summarizes how strongly turning behavior follows visual feedback as that discrepancy grows.”

Use the script in `analysis/plot_behavioral_signature.py` after downloading `dataconf.csv`. The plot is a first descriptive summary, not a model fit or a replication of the model-comparison result.

## B6. Plan and roles (form)

**Milestone 1 — Tue, Nov 24:** Download and validate the trial-level data; reproduce the behavioral-signature plot; confirm the paper’s model definitions and fitting setup.

**Milestone 2 — Mon, Nov 30:** Complete a first model-fitting/BIC comparison and run the initial visual-reliability simulations; prepare updated figures and slides for feedback before the Dec 4 expo.

**Labor division:** [For a group, assign data/behavioral plot, model implementation and recovery, extension simulations, and presentation. For an individual project, state that you will complete these workstreams sequentially.]

**Course deadlines:** In-class expo Fri, Dec 4; final video and slides due Mon, Dec 7.

## B7. Challenges and open questions (form)

- The original implementation is MATLAB and requires the Optimization Toolbox. Should we reproduce the fitting pipeline in MATLAB, or is a Python reimplementation acceptable?
- How closely should we match the original parameter bounds, random restarts, and likelihood functions for the model-recovery and parameter-recovery checks?
- For the extension, is a simulation that changes the probability that visual feedback is trustworthy an appropriate manipulation, or should we instead vary the noise level of the visual cue itself?
- What level of agreement with the original participant-level BIC results is expected for a successful replication?
