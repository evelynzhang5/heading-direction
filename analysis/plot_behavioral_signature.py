#!/usr/bin/env python3
"""Plot the empirical heading-direction behavioral signature.

Expected input: data/dataconf.csv from the authors' public repository.
The signature is response error (theta_t - alpha) against feedback offset
(feedback angle - path-integration heading estimate at feedback time).
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "dataconf.csv",
        help="Path to the authors' dataconf.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "behavioral_signature.png",
        help="Output image path",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=10.0,
        help="Feedback-offset bin width in degrees (default: 10)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Could not find {args.data}. Download dataconf.csv from "
            "https://github.com/sharootonian/CombinationAndCompetitionHeadingDirection/tree/main/data "
            "and save it at data/dataconf.csv."
        )

    df = pd.read_csv(args.data)
    required = {"subj", "condition", "error", "fb_offset"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    feedback = df.loc[df["condition"].astype(str).str.upper().isin(["FB", "FEEDBACK"])].copy()
    feedback["error"] = pd.to_numeric(feedback["error"], errors="coerce")
    feedback["fb_offset"] = pd.to_numeric(feedback["fb_offset"], errors="coerce")
    feedback = feedback.dropna(subset=["subj", "error", "fb_offset"])

    if feedback.empty:
        raise ValueError("No feedback-condition trials remain after filtering.")

    # Bin offsets symmetrically around zero for a readable empirical summary.
    lo = args.bin_width * (feedback["fb_offset"].min() // args.bin_width)
    hi = args.bin_width * (feedback["fb_offset"].max() // args.bin_width + 1)
    edges = [lo + i * args.bin_width for i in range(int(round((hi - lo) / args.bin_width)) + 1)]
    feedback["offset_bin"] = pd.cut(
        feedback["fb_offset"], bins=edges, include_lowest=True, right=False
    )

    summary = (
        feedback.groupby("offset_bin", observed=True)["error"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    summary["midpoint"] = summary["offset_bin"].map(lambda interval: interval.mid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Individual trial observations show the spread; binned means summarize
    # the population-level relationship without hiding the raw data.
    ax.scatter(
        feedback["fb_offset"],
        feedback["error"],
        s=9,
        alpha=0.12,
        label="Individual feedback trials",
    )
    ax.errorbar(
        summary["midpoint"],
        summary["mean"],
        yerr=1.96 * summary["sem"],
        fmt="o-",
        capsize=3,
        linewidth=2,
        markersize=4,
        label="Binned mean ± 95% normal-approximation CI",
    )
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set(
        xlabel="Feedback offset (degrees)",
        ylabel="Response error (degrees)",
        title="Behavioral signature: response error vs. feedback offset",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    print(f"Saved {args.output}")
    print(
        f"Participants: {feedback['subj'].nunique()} | "
        f"Feedback trials: {len(feedback)}"
    )


if __name__ == "__main__":
    main()
