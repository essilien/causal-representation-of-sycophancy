"""
Visualization for the irrelevant-answer control, reproducing the violin (margin
distributions across conditions) + scatter (delta-shift comparison) figure style,
reading from the flat CSV that irrelevant_answer_control.py now saves
(irrelevant_control_flat.csv) -- no need to reload the model or re-run scoring.

Usage:
    pip install pandas matplotlib seaborn
    python plot_irrelevant_control.py --csv /path/to/irrelevant_control_flat.csv
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default="irrelevant_control_flat.csv")
    parser.add_argument("--out", type=str, default="margin_distribution_analysis.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # Rename to match the figure's condition labels
    df = df.rename(columns={
        "neutral_margin": "Neutral Margin",
        "biased_margin": "Biased Margin",
        "control_margin": "Control Margin",
        "delta_bias": "Delta Bias",
        "delta_control": "Delta Control",
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # ---- Plot A: violin plot of margins across conditions ----
    df_long = df.melt(
        id_vars=["idx"],
        value_vars=["Neutral Margin", "Biased Margin", "Control Margin"],
        var_name="Condition", value_name="Margin",
    )
    df_long["Condition"] = df_long["Condition"].map({
        "Neutral Margin": "Neutral", "Biased Margin": "Biased", "Control Margin": "Control",
    })
    sns.violinplot(
        data=df_long, x="Condition", y="Margin", hue="Condition",
        palette=["#2b5c8f", "#d95f02", "#7570b3"], legend=False,
        ax=axes[0], inner="quartile", cut=0,
    )
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7,
                     label="Decision boundary (margin = 0)")
    axes[0].set_title("A. Distribution of Answer Margins across Conditions")
    axes[0].set_xlabel("Experimental Condition")
    axes[0].set_ylabel(r"Length-normalized margin ($\log P(y_{\mathrm{correct}}) - \log P(y_{\mathrm{other}})$)")
    axes[0].legend(loc="upper right")

    # ---- Plot B: scatter of Delta Control vs Delta Bias ----
    sns.scatterplot(
        data=df, x="Delta Control", y="Delta Bias", ax=axes[1],
        alpha=0.5, color="#1b9e77", edgecolor=None, s=25,
    )
    max_val = max(df["Delta Control"].max(), df["Delta Bias"].max())
    min_val = min(df["Delta Control"].min(), df["Delta Bias"].min())
    axes[1].plot([min_val, max_val], [min_val, max_val], color="gray", linestyle=":",
                 label="Identity line (y = x)")
    axes[1].set_title(r"B. Continuous Shift Comparison ($\Delta$Margin)")
    axes[1].set_xlabel(r"Control shift ($\Delta\mathrm{Margin}_{\mathrm{control}}$)")
    axes[1].set_ylabel(r"Biased shift ($\Delta\mathrm{Margin}_{\mathrm{biased}}$)")
    r = df["Delta Control"].corr(df["Delta Bias"])
    axes[1].annotate(f"r = {r:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=11)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
