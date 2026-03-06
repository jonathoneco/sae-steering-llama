"""Find SAE features related to the Eiffel Tower concept.

Encodes several Eiffel Tower-related prompts through the model + SAE and
identifies which features activate most consistently and strongly.
"""

import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import load_model, load_sae_full, build_input_ids, SYSTEM_PROMPT, LAYER

EIFFEL_PROMPTS = [
    "Tell me about the Eiffel Tower in Paris.",
    "The Eiffel Tower is the most visited monument in France.",
    "Gustave Eiffel designed the famous iron tower for the 1889 World's Fair.",
    "When I visited Paris, the first thing I saw was the Eiffel Tower.",
    "The tower on the Champ de Mars is 330 meters tall.",
    "France's most iconic landmark lights up every night.",
    "The iron lattice structure was completed in 1889.",
    "Tourists flock to see the tower by the Seine river.",
]


def get_last_token_hidden(model, input_ids, layer):
    """Get the hidden state at the target layer for the last token."""
    hidden = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        elif isinstance(output, torch.Tensor):
            h = output
        else:
            h = output[0]
        if h.ndim == 3:
            hidden["val"] = h[:, -1, :].detach().float()
        else:
            hidden["val"] = h.detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return hidden["val"]


def main():
    print("Loading SAE...")
    sae, _ = load_sae_full(device="cuda")

    print("Loading model...")
    model, tokenizer = load_model()

    print(f"\nEncoding {len(EIFFEL_PROMPTS)} Eiffel Tower prompts through layer {LAYER} SAE...")

    # Collect feature activations across all prompts
    all_activations = []
    for prompt in EIFFEL_PROMPTS:
        input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, prompt)
        hidden = get_last_token_hidden(model, input_ids, LAYER)
        features = sae.encode(hidden.to("cuda"))  # (1, n_features)
        all_activations.append(features.squeeze(0).detach().cpu().numpy())

    activations = np.stack(all_activations)  # (n_prompts, n_features)

    # Find features that activate across multiple prompts
    active_mask = activations > 0  # (n_prompts, n_features)
    activation_count = active_mask.sum(axis=0)  # how many prompts each feature fires on
    mean_activation = np.where(active_mask, activations, 0).sum(axis=0) / np.maximum(activation_count, 1)

    # Rank by frequency × mean activation
    score = activation_count * mean_activation

    # Get top 30 features
    top_indices = np.argsort(score)[::-1][:30]

    print(f"\nTop 30 features by (frequency × mean activation):")
    print(f"{'Rank':>4} {'Feature':>8} {'Count':>5} {'MeanAct':>8} {'MaxAct':>8} {'Score':>8}")
    print("-" * 50)
    for rank, idx in enumerate(top_indices):
        if score[idx] == 0:
            break
        print(f"{rank+1:>4} {idx:>8} {activation_count[idx]:>5} "
              f"{mean_activation[idx]:>8.2f} {activations[:, idx].max():>8.2f} "
              f"{score[idx]:>8.2f}")

    # Also check feature 21576 specifically
    known_idx = 21576
    print(f"\n--- Known Eiffel feature #{known_idx} ---")
    print(f"  Count: {activation_count[known_idx]}/{len(EIFFEL_PROMPTS)}")
    print(f"  Mean activation: {mean_activation[known_idx]:.2f}")
    print(f"  Max activation: {activations[:, known_idx].max():.2f}")
    print(f"  Per-prompt: {activations[:, known_idx].tolist()}")


if __name__ == "__main__":
    main()
