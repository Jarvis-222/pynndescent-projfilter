#!/usr/bin/env python3
"""Plots from the latest GIST1M sweep_recall_grid run:
   1. Phase-B wall-clock time vs filter configuration
   2. Search recall@10 vs filter configuration, 2x2 subplots per epsilon
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Data from sweep_recall_grid_gist1m_mc10_m16_tree_20260520T2250.txt ---

# config: (p_tau_display, p_tau_numeric, t_phase_a, t_phase_b, skip_pct, dist_comps)
ROWS = [
    ("vanilla", 1.00, 38.90, 39.01,  0.0, 257_343_961),
    ("0.99",    0.99, 40.01, 38.61,  7.9, 232_394_240),
    ("0.95",    0.95, 39.27, 38.75, 11.6, 220_586_551),
    ("0.9",     0.90, 39.71, 36.51, 14.4, 211_795_565),
    ("0.8",     0.80, 39.24, 36.03, 18.7, 198_066_347),
    ("0.7",     0.70, 39.93, 35.22, 22.5, 185_725_006),
    ("0.6",     0.60, 39.43, 34.99, 26.3, 173_737_606),
    ("0.5",     0.50, 39.80, 33.74, 30.2, 161_232_590),
    ("0.4",     0.40, 39.24, 33.81, 34.5, 147_535_060),
    ("0.3",     0.30, 39.79, 30.08, 39.6, 131_991_360),
    ("0.2",     0.20, 40.54, 29.39, 45.9, 112_946_270),
    ("0.1",     0.10, 39.39, 24.77, 55.0,  86_584_215),
]

# Search recall@10 indexed by config -> { epsilon: recall }
RECALL = {
    "vanilla": {0.10: 0.8443, 0.20: 0.9484, 0.30: 0.9752, 0.50: 0.9840},
    "0.99":    {0.10: 0.8458, 0.20: 0.9489, 0.30: 0.9762, 0.50: 0.9853},
    "0.95":    {0.10: 0.8464, 0.20: 0.9504, 0.30: 0.9767, 0.50: 0.9852},
    "0.9":     {0.10: 0.8445, 0.20: 0.9491, 0.30: 0.9782, 0.50: 0.9861},
    "0.8":     {0.10: 0.8469, 0.20: 0.9500, 0.30: 0.9771, 0.50: 0.9851},
    "0.7":     {0.10: 0.8576, 0.20: 0.9568, 0.30: 0.9794, 0.50: 0.9874},
    "0.6":     {0.10: 0.8575, 0.20: 0.9580, 0.30: 0.9793, 0.50: 0.9869},
    "0.5":     {0.10: 0.8522, 0.20: 0.9591, 0.30: 0.9813, 0.50: 0.9883},
    "0.4":     {0.10: 0.8555, 0.20: 0.9600, 0.30: 0.9830, 0.50: 0.9888},
    "0.3":     {0.10: 0.8638, 0.20: 0.9645, 0.30: 0.9859, 0.50: 0.9913},
    "0.2":     {0.10: 0.8549, 0.20: 0.9664, 0.30: 0.9871, 0.50: 0.9919},
    "0.1":     {0.10: 0.8685, 0.20: 0.9733, 0.30: 0.9907, 0.50: 0.9939},
}

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
})

OUT_DIR = Path(__file__).resolve().parent / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png + {name}.pdf")


def fig_phase_b():
    """Phase-B wall-clock time vs filter rate (skip %)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Plot phase B vs skip%, ordered by skip% ascending (vanilla -> aggressive)
    sorted_rows = sorted(ROWS, key=lambda r: r[4])  # by skip%
    skip_pcts = [r[4] for r in sorted_rows]
    phase_b = [r[3] for r in sorted_rows]
    labels = [r[0] for r in sorted_rows]

    ax.plot(skip_pcts, phase_b, "o-", color="tab:red", label="Phase B (NN-Descent iteration loop)")
    # Vanilla horizontal reference for visibility
    vanilla_pb = next(r[3] for r in ROWS if r[0] == "vanilla")
    ax.axhline(vanilla_pb, color="tab:blue", linestyle="--", alpha=0.5,
               label=f"vanilla baseline ({vanilla_pb:.1f} s)")

    # Annotate p_tau labels
    for i, (sp, pb, lab) in enumerate(zip(skip_pcts, phase_b, labels)):
        ax.annotate(
            f"p$_\\tau$={lab}" if lab != "vanilla" else "vanilla",
            (sp, pb),
            xytext=(5, 6 if i % 2 == 0 else -14),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("Filter rate — skipped pairs (% of total attempted)")
    ax.set_ylabel("Phase B (iteration loop) wall-clock time (s)")
    ax.set_title("GIST 1M — Phase B time vs filter aggressiveness\n"
                 "(mc=10, m=16, tree_init=True, single-thread)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    # Add a secondary annotation: % reduction
    reduction = [100.0 * (vanilla_pb - pb) / vanilla_pb for pb in phase_b]
    ax2 = ax.twinx()
    ax2.plot(skip_pcts, reduction, "s-", color="tab:green", alpha=0.6,
             markersize=4, label="Phase B reduction (%)")
    ax2.set_ylabel("Phase B reduction vs vanilla (%)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.legend(loc="upper left")

    save(fig, "fig_phase_b_vs_filter_rate")


def fig_recall_subplots():
    """2x2 grid of search recall@10 vs filter rate, one panel per epsilon."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    axes = axes.flatten()

    epsilons = [0.10, 0.20, 0.30, 0.50]

    # Order configs by skip%
    sorted_rows = sorted(ROWS, key=lambda r: r[4])
    skip_pcts = [r[4] for r in sorted_rows]
    labels = [r[0] for r in sorted_rows]

    for ax, eps in zip(axes, epsilons):
        recalls = [RECALL[r[0]][eps] for r in sorted_rows]
        vanilla_r = RECALL["vanilla"][eps]

        ax.plot(skip_pcts, recalls, "o-", color="tab:red")
        ax.axhline(vanilla_r, color="tab:blue", linestyle="--", alpha=0.5,
                   label=f"vanilla baseline ({vanilla_r:.4f})")

        # Highlight points where filter beats vanilla
        for sp, r, lab in zip(skip_pcts, recalls, labels):
            color = "tab:green" if r > vanilla_r else "tab:red"
            ax.scatter([sp], [r], color=color, zorder=5, s=40)

        ax.set_title(f"ε = {eps:.2f}", fontsize=12)
        ax.set_ylabel("Search recall@10")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

        # Annotate the best (most aggressive filter, lowest p_tau) and vanilla
        i_best = max(range(len(recalls)), key=lambda i: recalls[i])
        ax.annotate(
            f"max @ p$_\\tau$={labels[i_best]}\n({recalls[i_best]:.4f})",
            (skip_pcts[i_best], recalls[i_best]),
            xytext=(-50, -30),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", alpha=0.4, lw=0.5),
        )

    for ax in axes[2:]:
        ax.set_xlabel("Filter rate (% pairs skipped)")

    fig.suptitle(
        "GIST 1M — Search recall@10 vs filter rate, by query budget ε\n"
        "(red point = filter ≤ vanilla; green point = filter > vanilla)",
        y=1.00,
    )

    save(fig, "fig_search_recall_by_epsilon")


def main():
    print(f"Writing figures to {OUT_DIR}/")
    fig_phase_b()
    fig_recall_subplots()


if __name__ == "__main__":
    main()
