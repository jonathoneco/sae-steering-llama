"""Phase E plot: Activation traces over token position, grouped by outcome."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "activation_traces.json") as f:
        results = json.load(f)

    # Group by concept score
    groups = {"success (2)": [], "partial (1)": [], "failure (0)": []}
    for r in results:
        score = r["scores"]["concept"]
        acts = r["target_activations"]
        if score == 2:
            groups["success (2)"].append(acts)
        elif score == 1:
            groups["partial (1)"].append(acts)
        else:
            groups["failure (0)"].append(acts)

    # --- Plot 1: Individual traces colored by outcome ---
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"success (2)": "green", "partial (1)": "orange", "failure (0)": "red"}

    for label, traces in groups.items():
        for trace in traces:
            ax.plot(trace, color=colors[label], alpha=0.4, linewidth=0.8)
        # Add a dummy line for legend
        if traces:
            ax.plot([], [], color=colors[label], label=f"{label} (n={len(traces)})", linewidth=2)

    ax.set_xlabel("Token Position (decoding step)")
    ax.set_ylabel(f"Feature #{21576} Activation")
    ax.set_title("Eiffel Tower Feature Activation During Steered Generation")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "activation_traces.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'activation_traces.png'}")

    # --- Plot 2: Mean trace per group ---
    fig, ax = plt.subplots(figsize=(12, 6))

    for label, traces in groups.items():
        if not traces:
            continue
        # Pad traces to same length
        max_len = max(len(t) for t in traces)
        padded = np.full((len(traces), max_len), np.nan)
        for j, t in enumerate(traces):
            padded[j, :len(t)] = t

        mean_trace = np.nanmean(padded, axis=0)
        std_trace = np.nanstd(padded, axis=0)
        x = np.arange(max_len)

        ax.plot(x, mean_trace, color=colors[label], label=f"{label} (n={len(traces)})", linewidth=2)
        ax.fill_between(x, mean_trace - std_trace, mean_trace + std_trace,
                        color=colors[label], alpha=0.15)

    ax.set_xlabel("Token Position (decoding step)")
    ax.set_ylabel(f"Mean Feature #{21576} Activation")
    ax.set_title("Mean Activation Traces by Outcome")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "activation_traces_mean.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'activation_traces_mean.png'}")

    # --- Plot 3: Top-5 feature frequency ---
    fig, ax = plt.subplots(figsize=(10, 6))
    from collections import Counter
    feature_counts = Counter()
    for r in results:
        for step in r["top5_per_step"]:
            for feat_idx, _ in step:
                feature_counts[feat_idx] += 1

    top20 = feature_counts.most_common(20)
    indices = [str(f[0]) for f in top20]
    counts = [f[1] for f in top20]

    bars = ax.barh(range(len(top20)), counts, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(indices)
    ax.set_xlabel("Frequency in Top-5 Across All Steps")
    ax.set_ylabel("Feature Index")
    ax.set_title("Most Frequently Active Features During Steered Generation")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Highlight target feature
    for i, (idx, _) in enumerate(top20):
        if idx == 21576:
            bars[i].set_color("coral")
            bars[i].set_label(f"Target feature #{21576}")
            ax.legend()
            break

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top_features_freq.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'top_features_freq.png'}")

    # Summary
    print("\nTrace Summary:")
    for label, traces in groups.items():
        if traces:
            all_means = [np.mean(t) for t in traces]
            print(f"  {label}: n={len(traces)}, "
                  f"mean_activation={np.mean(all_means):.3f} +/- {np.std(all_means):.3f}")


if __name__ == "__main__":
    main()
