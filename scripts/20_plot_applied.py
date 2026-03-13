"""Phase F plots: Applied scenario comparison across conditions."""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

CONDITIONS = ["unsteered", "additive", "clamping", "prompting"]
CONDITION_COLORS = {
    "unsteered": "#888888",
    "additive": "#4C72B0",
    "clamping": "#DD8452",
    "prompting": "#55A868",
}
CRITERIA = ["concept", "instruction", "fluency", "naturalness"]


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "applied_scenarios.json") as f:
        results = json.load(f)

    scenarios = sorted(set(r["scenario"] for r in results))

    # --- Plot 1: Grouped bar chart per scenario ---
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 6), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        x = np.arange(len(CRITERIA))
        width = 0.2

        for j, condition in enumerate(CONDITIONS):
            entries = [r for r in results
                       if r["scenario"] == scenario and r["condition"] == condition]
            if not entries:
                continue
            means = [np.mean([e["scores"][c] for e in entries]) for c in CRITERIA]
            stds = [np.std([e["scores"][c] for e in entries]) for c in CRITERIA]
            ax.bar(x + j * width, means, width, yerr=stds,
                   label=condition, color=CONDITION_COLORS[condition],
                   capsize=3, alpha=0.85)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(CRITERIA, rotation=30)
        ax.set_title(scenario.replace("_", " ").title())
        ax.set_ylim(0, 2.3)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Score (0-2)")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Applied Scenarios: Score Comparison by Condition", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "applied_scenarios_bars.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'applied_scenarios_bars.png'}")

    # --- Plot 2: Naturalness spotlight ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scenarios))
    width = 0.2

    for j, condition in enumerate(CONDITIONS):
        nat_means = []
        nat_stds = []
        for scenario in scenarios:
            entries = [r for r in results
                       if r["scenario"] == scenario and r["condition"] == condition]
            if entries:
                vals = [e["scores"]["naturalness"] for e in entries]
                nat_means.append(np.mean(vals))
                nat_stds.append(np.std(vals))
            else:
                nat_means.append(0)
                nat_stds.append(0)

        ax.bar(x + j * width, nat_means, width, yerr=nat_stds,
               label=condition, color=CONDITION_COLORS[condition],
               capsize=3, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios])
    ax.set_ylabel("Naturalness Score (0-2)")
    ax.set_title("Naturalness: SAE Steering vs Prompting")
    ax.legend()
    ax.set_ylim(0, 2.3)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "applied_naturalness.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'applied_naturalness.png'}")

    # --- Plot 3: Concept vs Naturalness trade-off ---
    fig, ax = plt.subplots(figsize=(8, 6))

    for condition in CONDITIONS:
        for scenario in scenarios:
            entries = [r for r in results
                       if r["scenario"] == scenario and r["condition"] == condition]
            if not entries:
                continue
            concept_mean = np.mean([e["scores"]["concept"] for e in entries])
            natural_mean = np.mean([e["scores"]["naturalness"] for e in entries])
            ax.scatter(concept_mean, natural_mean,
                       color=CONDITION_COLORS[condition], s=100, zorder=5)
            ax.annotate(f"{scenario[:4]}", (concept_mean, natural_mean),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Add legend entries
    for condition in CONDITIONS:
        ax.scatter([], [], color=CONDITION_COLORS[condition], s=100, label=condition)
    ax.legend(loc="best")
    ax.set_xlabel("Concept Inclusion Score")
    ax.set_ylabel("Naturalness Score")
    ax.set_title("Concept Inclusion vs Naturalness Trade-off")
    ax.set_xlim(-0.1, 2.2)
    ax.set_ylim(-0.1, 2.2)
    ax.plot([0, 2], [0, 2], "k--", alpha=0.2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "applied_tradeoff.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'applied_tradeoff.png'}")

    # Summary table
    print("\nFull Summary:")
    print(f"{'Scenario':>15} {'Condition':>12} {'HM4':>6} {'Concept':>8} {'IF':>6} "
          f"{'Fluency':>8} {'Natural':>8}")
    print("-" * 70)
    for scenario in scenarios:
        for condition in CONDITIONS:
            entries = [r for r in results
                       if r["scenario"] == scenario and r["condition"] == condition]
            if entries:
                hm = np.mean([e["harmonic_mean"] for e in entries])
                c = np.mean([e["scores"]["concept"] for e in entries])
                i = np.mean([e["scores"]["instruction"] for e in entries])
                f = np.mean([e["scores"]["fluency"] for e in entries])
                n = np.mean([e["scores"]["naturalness"] for e in entries])
                print(f"{scenario:>15} {condition:>12} {hm:>6.3f} {c:>8.3f} {i:>6.3f} "
                      f"{f:>8.3f} {n:>8.3f}")
        print()


if __name__ == "__main__":
    main()
