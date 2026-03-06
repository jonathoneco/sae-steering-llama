"""Plot sweep results: metrics vs alpha/clamp_value."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = Path(__file__).parent.parent / "plots"


def load_results(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def plot_sweep(results, param_key, title, output_name):
    """Plot metrics vs steering parameter."""
    by_param = defaultdict(list)
    for r in results:
        by_param[r[param_key]].append(r)

    params = sorted(by_param.keys())
    metrics = {
        "Concept Inclusion": [],
        "Instruction Following": [],
        "Fluency": [],
        "Harmonic Mean": [],
        "3-gram Repetition": [],
        "Explicit Eiffel": [],
    }
    stds = {k: [] for k in metrics}

    for p in params:
        group = by_param[p]
        concepts = [r["scores"]["concept"] for r in group]
        ifs = [r["scores"]["instruction"] for r in group]
        fluencies = [r["scores"]["fluency"] for r in group]
        hms = [r["harmonic_mean"] for r in group]
        tri_reps = [r["trigram_repetition"] for r in group]
        eiffels = [1 if r["explicit_eiffel"] else 0 for r in group]

        for key, vals in [
            ("Concept Inclusion", concepts),
            ("Instruction Following", ifs),
            ("Fluency", fluencies),
            ("Harmonic Mean", hms),
            ("3-gram Repetition", tri_reps),
            ("Explicit Eiffel", eiffels),
        ]:
            metrics[key].append(np.mean(vals))
            stds[key].append(np.std(vals))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=14)

    for ax, (name, means) in zip(axes.flat, metrics.items()):
        means = np.array(means)
        std = np.array(stds[name])
        ax.plot(params, means, "o-", linewidth=2)
        ax.fill_between(params, means - std, means + std, alpha=0.2)
        ax.set_xlabel(param_key.replace("_", " ").title())
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.savefig(PLOTS_DIR / output_name, dpi=150, bbox_inches="tight")
    print(f"Saved: {PLOTS_DIR / output_name}")
    plt.close()


def main():
    # Additive sweep
    additive = load_results("sweep_additive.json")
    if additive:
        plot_sweep(additive, "alpha", "Additive Steering Sweep", "sweep_additive.png")

    # Clamping sweep
    clamping = load_results("sweep_clamping.json")
    if clamping:
        plot_sweep(clamping, "clamp_value", "Clamping Steering Sweep", "sweep_clamping.png")

    # Comparison plot
    if additive and clamping:
        fig, ax = plt.subplots(figsize=(10, 6))
        for results, param_key, label in [
            (additive, "alpha", "Additive"),
            (clamping, "clamp_value", "Clamping"),
        ]:
            by_param = defaultdict(list)
            for r in results:
                by_param[r[param_key]].append(r)
            params = sorted(by_param.keys())
            hms = [np.mean([r["harmonic_mean"] for r in by_param[p]]) for p in params]
            ax.plot(params, hms, "o-", label=label, linewidth=2)

        # Prompting baseline
        prompting = load_results("prompting_baseline.json")
        if prompting:
            hm = np.mean([r["harmonic_mean"] for r in prompting])
            ax.axhline(y=hm, color="green", linestyle="--", linewidth=2, label=f"Prompting (HM={hm:.2f})")

        ax.set_xlabel("Steering Coefficient")
        ax.set_ylabel("Harmonic Mean")
        ax.set_title("Additive vs Clamping vs Prompting")
        ax.legend()
        ax.grid(True, alpha=0.3)
        PLOTS_DIR.mkdir(exist_ok=True)
        plt.savefig(PLOTS_DIR / "comparison.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {PLOTS_DIR / 'comparison.png'}")
        plt.close()


if __name__ == "__main__":
    main()
