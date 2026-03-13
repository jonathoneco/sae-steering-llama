"""Phase B plot: HM vs alpha per concept + best-alpha bar chart."""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "concept_sweep_curated.json") as f:
        data = json.load(f)

    # Group by (concept, alpha)
    grouped = defaultdict(list)
    for r in data:
        grouped[(r["concept"], r["alpha"])].append(r["harmonic_mean"])

    concepts = sorted(set(r["concept"] for r in data))
    alphas = sorted(set(r["alpha"] for r in data))

    # --- Plot 1: HM vs alpha per concept ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(concepts)))

    best_per_concept = {}
    for i, concept in enumerate(concepts):
        means = []
        stds = []
        for alpha in alphas:
            vals = grouped.get((concept, alpha), [])
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.errorbar(alphas, means, yerr=stds, label=concept.replace("_", " "),
                     color=cmap[i], marker="o", capsize=3)

        # Find best alpha (excluding 0)
        nonzero = [(a, m) for a, m in zip(alphas, means) if a > 0]
        if nonzero:
            best_alpha, best_hm = max(nonzero, key=lambda x: x[1])
            best_per_concept[concept] = (best_alpha, best_hm)

    ax.set_xlabel("Alpha (steering coefficient)")
    ax.set_ylabel("Harmonic Mean Score")
    ax.set_title("Concept Steering Effectiveness by Alpha")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "concept_sweep_hm.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'concept_sweep_hm.png'}")

    # --- Plot 2: Best-alpha HM bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_concepts = sorted(best_per_concept.keys(),
                             key=lambda c: best_per_concept[c][1], reverse=True)
    labels = [c.replace("_", " ") for c in sorted_concepts]
    hm_vals = [best_per_concept[c][1] for c in sorted_concepts]
    alpha_vals = [best_per_concept[c][0] for c in sorted_concepts]

    bars = ax.bar(range(len(sorted_concepts)), hm_vals, color="steelblue", alpha=0.85)
    ax.set_xticks(range(len(sorted_concepts)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Best Harmonic Mean Score")
    ax.set_title("Best Steering Performance per Concept")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate with best alpha
    for i, (hm, alpha) in enumerate(zip(hm_vals, alpha_vals)):
        ax.text(i, hm + 0.02, f"a={alpha}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "concept_sweep_best.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'concept_sweep_best.png'}")

    # Summary
    print("\nConcept Sweep Summary:")
    print(f"{'Concept':>25} {'Best Alpha':>10} {'Best HM':>8}")
    print("-" * 45)
    for c in sorted_concepts:
        ba, bhm = best_per_concept[c]
        print(f"{c:>25} {ba:>10} {bhm:>8.3f}")


if __name__ == "__main__":
    main()
