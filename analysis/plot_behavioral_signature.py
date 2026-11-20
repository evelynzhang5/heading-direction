#!/usr/bin/env python3
"""Plot the empirical heading-direction behavioral signature.

Uses the authors' dataconf.csv. Each participant contributes equally to the
group mean within each feedback-offset bin; error bars are across participants.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dataconf.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "behavioral_signature.png")
    parser.add_argument("--bin-width", type=float, default=10.0)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Missing {args.data}. Download dataconf.csv from "
            "https://github.com/sharootonian/CombinationAndCompetitionHeadingDirection/tree/main/data "
            "and save it to data/dataconf.csv."
        )

    df = pd.read_csv(args.data)
    required = {"subj", "condition", "error", "fb_offset"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(df.columns)}")

    feedback = df[df["condition"].astype(str).str.upper().eq("FB")].copy()
    feedback["error"] = pd.to_numeric(feedback["error"], errors="coerce")
    feedback["fb_offset"] = pd.to_numeric(feedback["fb_offset"], errors="coerce")
    feedback = feedback.dropna(subset=["subj", "error", "fb_offset"])
    if feedback.empty:
        raise ValueError("No valid FB-condition trials found.")

    width = args.bin_width
    lo = np.floor(feedback["fb_offset"].min() / width) * width
    hi = (np.floor(feedback["fb_offset"].max() / width) + 1) * width
    bins = np.arange(lo, hi + width, width)
    feedback["offset_bin"] = pd.cut(feedback["fb_offset"], bins=bins, right=False)

    # First average within participant and bin, then across participants.
    by_subject = (
        feedback.groupby(["subj", "offset_bin"], observed=True)["error"]
        .mean()
        .reset_index()
    )
    summary = (
        by_subject.groupby("offset_bin", observed=True)["error"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["midpoint"] = summary["offset_bin"].map(lambda interval: interval.mid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(feedback["fb_offset"], feedback["error"], s=8, alpha=0.10,
               label="Individual trials")
    ax.errorbar(summary["midpoint"], summary["mean"], yerr=1.96 * summary["sem"],
                fmt="o-", capsize=3, linewidth=2, markersize=4,
                label="Participant-averaged bin mean ± 95% CI")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Feedback offset (degrees)")
    ax.set_ylabel("Response error (degrees)")
    ax.set_title("Response error as a function of visual-feedback offset")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    print(f"Saved {args.output}")
    print(f"Participants: {feedback['subj'].nunique()}; feedback trials: {len(feedback)}")


if __name__ == "__main__":
    main()
