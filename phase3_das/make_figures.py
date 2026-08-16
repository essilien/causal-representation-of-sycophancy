"""
Visualization script for key results in the sycophancy causal-abstraction paper.
Data is embedded directly (transcribed from the results tables), so this runs
standalone without needing any of the project's saved output files.

Produces three figures:
  1. das_vs_patch.png    -- the central finding: DAS vs full-representation patching
                             IIA across all 32 blocks, with the crossover highlighted.
  2. phase2_localization.png -- Phase 2's R^2 (representation) and margin-shift
                             (behavior) curves, dual-axis, with the candidate range shaded.
  3. copy_effect_convergence.png -- the three copy-effect control iterations,
                             showing the ratio failing to converge.

Usage:
    pip install matplotlib numpy
    python make_figures.py
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =============================================================================
# Figure 1 — DAS vs. full-representation patching, all blocks (central finding)
# =============================================================================

blocks = np.arange(0, 32)

# Full-patch IIA, blocks 0-31 -- final data, direct block-numbering convention
# (matches Table 1 / Sections 4.4-4.5 of the paper; no exclusion needed since block 0
# here is a genuine transformer block, not the old embedding-output indexing bug)
full_patch_iia = np.array([
    40.5, 40.5, 40.5, 40.5, 40.5, 40.5, 40.5, 40.5, 40.5, 40.5, 41.3, 41.3,  # 0-11
    41.3, 42.1, 43.0, 44.6, 47.1, 47.9, 47.9, 47.9, 47.9, 47.9,              # 12-21
    48.8, 48.8, 52.1, 55.4, 57.9, 63.6, 64.5, 63.6, 73.6, 74.4,              # 22-31
])

# DAS IIA, blocks 0-31 (last-3-checkpoint-averaged) -- final, seed-fixed data
das_iia = np.array([
    41.0, 41.0, 41.0, 41.3, 41.0, 41.0, 41.9, 43.0, 42.4, 43.0, 43.8, 43.2,  # 0-11
    44.6, 48.5, 52.9, 48.8, 49.3, 48.2, 47.1, 48.2, 47.9, 49.6,              # 12-21
    47.7, 47.1, 48.5, 47.9, 47.6, 48.5, 48.2, 45.5, 48.2, 44.4,              # 22-31
])

trivial_baseline = 40.5

# ACL single-column width is ~3.3in; a squarer aspect ratio (vs. the original 9:5)
# keeps the 32-point line legible when scaled down to column width, rather than
# compressing the y-axis into an unreadable flat band.
# ACL single-column width is ~3.3in; a squarer aspect ratio (vs. the original 9:5)
# keeps the 32-point line legible when scaled down to column width, rather than
# compressing the y-axis into an unreadable flat band. Scoped to this figure only
# (rc_context) so Figures 2-3 below keep the module-level defaults set earlier.
with plt.rc_context({"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
                      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7}):
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot(blocks, full_patch_iia, marker="o", markersize=3, linewidth=1.3,
            color="#1f77b4", label="Full-representation patching")
    ax.plot(blocks, das_iia, marker="s", markersize=3, linewidth=1.3,
            color="#d62728", label="DAS (rank 64)")
    ax.axhline(trivial_baseline, color="gray", linestyle="--", linewidth=0.8,
               label="Trivial baseline")

    # Shade the candidate range (blocks 12-21) where DAS >= full-patch
    ax.axvspan(12, 21, color="#d62728", alpha=0.07, zorder=0)
    ax.annotate("DAS $\\geq$\npatching", xy=(16.5, 55), ha="center", fontsize=6.5,
                color="#d62728")
    ax.annotate("patching $\\gg$\nDAS", xy=(27, 68), ha="center", fontsize=6.5,
                color="#1f77b4")

    ax.set_xlabel("Transformer block")
    ax.set_ylabel("IIA")
    ax.set_xticks(range(0, 32, 4))
    ax.set_ylim(35, 80)
    ax.legend(loc="upper left", frameon=False, handlelength=1.5, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)
    fig.savefig("das_vs_patch.png")
    plt.close(fig)


# =============================================================================
# Figure 2 — Phase 2 localization: representation (R^2) vs. behavior (margin shift)
# =============================================================================

blocks2 = np.arange(0, 32)

# Regression probe R^2 per block (0 = embedding, excluded/NaN; approximate values
# transcribed from the Phase 2 sweep)
r2 = np.array([
    np.nan,
    0.019, 0.031, 0.013, 0.041, 0.049, 0.028, 0.016, 0.037, 0.061, 0.108, 0.094,
    0.089, 0.129, 0.135, 0.130, 0.150, 0.136, 0.144, 0.136, 0.110, 0.121,
    0.106, 0.088, 0.075, 0.077, 0.104, 0.080, 0.082, 0.077, 0.065, 0.060,
])

# Raw behavioral margin-shift curve (biased - neutral), same block indexing
margin_shift = np.array([
    np.nan,
    0.008, -0.021, -0.013, 0.005, 0.078, 0.051, 0.062, 0.188, 0.213, 0.154, 0.148,
    0.118, 0.233, 0.217, 0.123, -0.112, -0.396, -0.535, -0.656, -0.725, -0.906,
    -1.224, -1.454, -1.865, -2.686, -3.233, -3.289, -3.952, -3.760, -3.472, -3.545,
])

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(blocks2, margin_shift, color="#d62728", linewidth=1.8, label="Margin shift (behavior)")
ax1.set_xlabel("Transformer block")
ax1.set_ylabel("Mean margin shift (biased \u2212 neutral)", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#d62728")
ax1.axhline(0, color="lightgray", linewidth=1, zorder=0)

ax2 = ax1.twinx()
ax2.plot(blocks2, r2, color="#1f77b4", linewidth=1.8, label="Regression R\u00b2 (representation)")
ax2.set_ylabel("Held-out R\u00b2", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")

ax1.axvspan(12, 21, color="gray", alpha=0.08, zorder=0)
ax1.set_title("Phase 2: representation (R\u00b2) vs. behavioral margin shift")
ax1.set_xticks(range(0, 32, 2))
fig.tight_layout()
fig.savefig("phase2_localization.png")
plt.close(fig)


# =============================================================================
# Figure 3 — Copy-effect control: three iterations, ratio failing to converge
# =============================================================================

templates = [
    "(1) \u201csimilar contexts\u201d",
    "(2) \u201cRandom word: X\u201d",
    "(3) \u201cRandom word: X,\nnoted for no reason\u201d",
]
ratios = [55.6, 66.6, 75.4]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(templates, ratios, color="#7f7f7f", width=0.55)
for bar, val in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%",
            ha="center", fontsize=10)
ax.axhline(100, color="lightgray", linestyle=":", linewidth=1)
ax.set_ylabel("Estimated copying share of margin shift")
ax.set_title("Copy-effect control: estimate rises across iterations,\nrather than converging")
ax.set_ylim(0, 110)
fig.tight_layout()
fig.savefig("copy_effect_convergence.png")
plt.close(fig)

print("Saved: das_vs_patch.png, phase2_localization.png, copy_effect_convergence.png")
