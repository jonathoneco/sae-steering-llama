"""Phase B: Discover SAE features for multiple concepts at layer 15.

For each concept in concept_prompts.CONCEPTS, encodes the discovery prompts
through the SAE and identifies the best-matching feature.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    load_model, load_sae_full, build_input_ids,
    SYSTEM_PROMPT, LAYER,
)
from concept_prompts import CONCEPTS

RESULTS_DIR = Path(__file__).parent.parent / "results"


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


def discover_feature(model, tokenizer, sae, prompts, layer=LAYER):
    """Find the best feature for a set of concept prompts."""
    all_activations = []
    for prompt in prompts:
        input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, prompt)
        hidden = get_last_token_hidden(model, input_ids, layer)
        features = sae.encode(hidden.to(sae.decoder.weight.device))
        all_activations.append(features.squeeze(0).detach().cpu().numpy())

    activations = np.stack(all_activations)
    active_mask = activations > 0
    activation_count = active_mask.sum(axis=0)
    mean_activation = np.where(active_mask, activations, 0).sum(axis=0) / np.maximum(activation_count, 1)
    score = activation_count * mean_activation

    top_indices = np.argsort(score)[::-1][:10]
    top_features = []
    for idx in top_indices:
        if score[idx] == 0:
            break
        top_features.append({
            "feature_idx": int(idx),
            "count": int(activation_count[idx]),
            "mean_activation": float(mean_activation[idx]),
            "max_activation": float(activations[:, idx].max()),
            "score": float(score[idx]),
        })

    return top_features


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading SAE (full)...")
    sae, _ = load_sae_full(device="cuda")
    print(f"SAE VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"Total VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    discovery_results = {}

    for concept_name, concept_def in CONCEPTS.items():
        print(f"\n{'='*60}")
        print(f"Concept: {concept_name}")
        print(f"{'='*60}")

        prompts = concept_def["discovery_prompts"]
        print(f"  Encoding {len(prompts)} discovery prompts...")

        top_features = discover_feature(model, tokenizer, sae, prompts)

        discovery_results[concept_name] = {
            "top_features": top_features,
            "n_prompts": len(prompts),
        }

        print(f"  Top 5 features:")
        for i, feat in enumerate(top_features[:5]):
            print(f"    #{feat['feature_idx']}: count={feat['count']}/{len(prompts)} "
                  f"mean={feat['mean_activation']:.2f} score={feat['score']:.1f}")

    output_path = RESULTS_DIR / "concept_discovery.json"
    with open(output_path, "w") as f:
        json.dump(discovery_results, f, indent=2)
    print(f"\nSaved discovery results to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DISCOVERY SUMMARY")
    print("=" * 60)
    for name, data in discovery_results.items():
        best = data["top_features"][0] if data["top_features"] else None
        if best:
            print(f"  {name:20s}: feature #{best['feature_idx']:>6d} "
                  f"(count={best['count']}/{data['n_prompts']}, score={best['score']:.1f})")
        else:
            print(f"  {name:20s}: NO FEATURES FOUND")


if __name__ == "__main__":
    main()
