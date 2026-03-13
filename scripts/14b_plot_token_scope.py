"""Phase C plot: Grouped bar chart comparing token scope conditions."""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

CONDITIONS = ["last_only", "all_tokens", "decode_only"]
CONDITION_LABELS = {
    "last_only": "Last Only",
    "all_tokens": "All Tokens",
    "decode_only": "Decode Only",
}
CONDITION_COLORS = {
    "last_only": "#4C72B0",
    "all_tokens": "#DD8452",
    "decode_only": "#55A868",
}
METRICS = ["harmonic_mean", "concept", "fluency"]
METRIC_LABELS = {
    "harmonic_mean": "HM",
    "concept": "Concept",
    "fluency": "Fluency",
}


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "token_scope.json") as f:
        data = json.load(f)

    # Group by condition
    by_condition = defaultdict(list)
    for r in data:
        by_condition[r["condition"]].append(r)

    conditions = [c for c in CONDITIONS if c in by_condition]

    # --- Plot: Grouped bar chart ---
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(METRICS))
    width = 0.25

    for j, condition in enumerate(conditions):
        entries = by_condition[condition]
        means = []
        stds = []
        for metric in METRICS:
            if metric == "harmonic_mean":
                vals = [e["harmonic_mean"] for e in entries]
            else:
                vals = [e["scores"][metric] for e in entries]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.bar(x + j * width, means, width, yerr=stds,
               label=CONDITION_LABELS.get(condition, condition),
               color=CONDITION_COLORS.get(condition, f"C{j}"),
               capsize=3, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Score (0-2)")
    ax.set_title("Token Scope: Steering Application Strategy Comparison")
    ax.legend()
    ax.set_ylim(0, 2.3)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "token_scope_comparison.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'token_scope_comparison.png'}")

    # Summary
    print("\nToken Scope Summary:")
    print(f"{'Condition':>12} {'N':>4} {'HM':>6} {'Concept':>8} {'Fluency':>8}")
    print("-" * 42)
    for condition in conditions:
        entries = by_condition[condition]
        hm = np.mean([e["harmonic_mean"] for e in entries])
        concept = np.mean([e["scores"]["concept"] for e in entries])
        fluency = np.mean([e["scores"]["fluency"] for e in entries])
        print(f"{condition:>12} {len(entries):>4} {hm:>6.3f} {concept:>8.3f} {fluency:>8.3f}")


if __name__ == "__main__":
    main()
