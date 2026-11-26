#!/usr/bin/env bash
# Rebuild every result and figure from the authors' open data (about 15 minutes on 12 cores).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data results figures
if [ ! -f data/dataconf.csv ]; then
  curl -fsSL -o data/dataconf.csv \
    https://raw.githubusercontent.com/sharootonian/CombinationAndCompetitionHeadingDirection/main/data/dataconf.csv
fi

cd analysis
python plot_behavioral_signature.py      # observed behavioral signature
python simulate_models.py                # Simulate: model predictions (paper Fig 6)
python fit_models.py                     # Fit + compare: BIC per participant (paper Fig 10)
python posterior_predictive.py           # PPC of all four models (paper Figs 11, S13)
python parameter_recovery.py             # Recover: parameters (paper Figs S9-S10)
python model_recovery.py                 # Recover: confusion matrix (paper Fig S4)
python extension_reliability.py          # Extend: glimpse-reliability manipulation
