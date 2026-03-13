"""Phase D plot: 2D heatmap of HM across clamp x alpha grid."""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "hybrid_sweep.json") as f:
        results = json.load(f)

    # Group by (clamp, alpha)
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["clamp_value"], r["alpha"])].append(r)

    clamp_vals = sorted(set(r["clamp_value"] for r in results))
    alpha_vals = sorted(set(r["alpha"] for r in results))

    # Build HM matrix
    hm_matrix = np.zeros((len(clamp_vals), len(alpha_vals)))
    concept_matrix = np.zeros_like(hm_matrix)
    fluency_matrix = np.zeros_like(hm_matrix)

    for i, c in enumerate(clamp_vals):
        for j, a in enumerate(alpha_vals):
            entries = grouped.get((c, a), [])
            if entries:
                hm_matrix[i, j] = np.mean([e["harmonic_mean"] for e in entries])
                concept_matrix[i, j] = np.mean([e["scores"]["concept"] for e in entries])
                fluency_matrix[i, j] = np.mean([e["scores"]["fluency"] for e in entries])

    # --- Plot: HM heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, matrix, title in zip(
        axes,
        [hm_matrix, concept_matrix, fluency_matrix],
        ["Harmonic Mean", "Concept Score", "Fluency Score"],
    ):
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=2)
        ax.set_xticks(range(len(alpha_vals)))
        ax.set_xticklabels(alpha_vals)
        ax.set_yticks(range(len(clamp_vals)))
        ax.set_yticklabels(clamp_vals)
        ax.set_xlabel("Additive Alpha")
        ax.set_ylabel("Clamp Value")
        ax.set_title(title)

        # Annotate cells
        for i in range(len(clamp_vals)):
            for j in range(len(alpha_vals)):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=10,
                        color="black" if matrix[i, j] > 0.5 else "white")

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Hybrid Clamping + Additive Steering", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hybrid_heatmap.png", dpi=150)
    print(f"Saved {PLOTS_DIR / 'hybrid_heatmap.png'}")

    # Summary
    print("\nHybrid Sweep Summary:")
    print(f"{'Clamp':>6} {'Alpha':>6} {'HM':>6} {'Concept':>8} {'IF':>6} {'Fluency':>8} {'Eiffel%':>8}")
    print("-" * 55)
    for c in clamp_vals:
        for a in alpha_vals:
            entries = grouped.get((c, a), [])
            if entries:
                hm = np.mean([e["harmonic_mean"] for e in entries])
                concept = np.mean([e["scores"]["concept"] for e in entries])
                instr = np.mean([e["scores"]["instruction"] for e in entries])
                fluency = np.mean([e["scores"]["fluency"] for e in entries])
                eiffel = np.mean([1 if e["explicit_eiffel"] else 0 for e in entries]) * 100
                print(f"{c:>6} {a:>6} {hm:>6.3f} {concept:>8.3f} {instr:>6.3f} {fluency:>8.3f} {eiffel:>7.1f}%")


if __name__ == "__main__":
    main()
