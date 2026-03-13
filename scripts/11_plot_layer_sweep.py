"""Phase A plots: HM vs alpha per layer + optimal alpha vs activation norm."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "layer_sweep.json") as f:
        data = json.load(f)

    layer_meta = {m["layer"]: m for m in data["layer_meta"]}
    results = data["results"]

    # Group by (layer, alpha)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["layer"], r["alpha"])].append(r["harmonic_mean"])

    layers = sorted(set(r["layer"] for r in results))
    alphas = sorted(set(r["alpha"] for r in results))

    # --- Plot 1: HM vs alpha per layer ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    optimal_alphas = {}
    for i, layer in enumerate(layers):
        means = []
        stds = []
        for alpha in alphas:
            vals = grouped.get((layer, alpha), [])
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.errorbar(alphas, means, yerr=stds, label=f"Layer {layer}",
                     color=cmap[i], marker="o", capsize=3)

        # Find optimal alpha (excluding 0)
        nonzero = [(a, m) for a, m in zip(alphas, means) if a > 0]
        if nonzero:
            best_alpha, best_hm = max(nonzero, key=lambda x: x[1])
            optimal_alphas[layer] = (best_alpha, best_hm)

    ax.set_xlabel("Alpha (steering coefficient)")
    ax.set_ylabel("Harmonic Mean Score")
    ax.set_title("Steering Effectiveness by Layer")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "layer_sweep_hm.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'layer_sweep_hm.png'}")

    # --- Plot 2: Optimal alpha vs activation norm ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    norm_vals = [layer_meta[l]["mean_activation_norm"] for l in layers if l in optimal_alphas]
    opt_alpha_vals = [optimal_alphas[l][0] for l in layers if l in optimal_alphas]
    opt_hm_vals = [optimal_alphas[l][1] for l in layers if l in optimal_alphas]
    layer_labels = [l for l in layers if l in optimal_alphas]

    ax1.scatter(norm_vals, opt_alpha_vals, s=80, c="steelblue", zorder=5)
    for l, n, a in zip(layer_labels, norm_vals, opt_alpha_vals):
        ax1.annotate(f"L{l}", (n, a), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)
    ax1.set_xlabel("Mean Activation Norm ||x^l||")
    ax1.set_ylabel("Optimal Alpha")
    ax1.set_title("Optimal Alpha vs Activation Norm")
    ax1.grid(True, alpha=0.3)

    # Normalized coefficient: alpha_hat = alpha / norm
    normalized = [a / n for a, n in zip(opt_alpha_vals, norm_vals)]
    ax2.bar(range(len(layer_labels)), normalized, color="coral", alpha=0.8)
    ax2.set_xticks(range(len(layer_labels)))
    ax2.set_xticklabels([f"L{l}" for l in layer_labels])
    ax2.set_ylabel("Normalized Coefficient (α/||x^l||)")
    ax2.set_title("Normalized Steering Coefficient by Layer")
    ax2.axhline(y=np.mean(normalized), color="gray", linestyle="--",
                label=f"Mean = {np.mean(normalized):.4f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "layer_sweep_norms.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'layer_sweep_norms.png'}")

    # --- Summary ---
    print("\nLayer Sweep Summary:")
    print(f"{'Layer':>5} {'Feature':>8} {'Norm':>8} {'Best α':>6} {'Best HM':>7} {'α/norm':>8}")
    print("-" * 50)
    for l in layers:
        m = layer_meta[l]
        if l in optimal_alphas:
            ba, bhm = optimal_alphas[l]
            norm = m["mean_activation_norm"]
            print(f"{l:>5} {m['feature_idx']:>8} {norm:>8.1f} {ba:>6} {bhm:>7.3f} {ba/norm:>8.5f}")
        else:
            print(f"{l:>5} {m['feature_idx']:>8} {m['mean_activation_norm']:>8.1f}   ---")


if __name__ == "__main__":
    main()
