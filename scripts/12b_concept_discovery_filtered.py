"""Phase B (rerun): Discover concept-specific SAE features with universal filter.

Same as 12_concept_discovery.py but filters out features that appear in the
top-3 for 3+ concepts (likely generic high-salience features, not concept-specific).
"""

import json
import sys
from pathlib import Path
from collections import Counter

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    load_model, load_sae_full, build_input_ids,
    SYSTEM_PROMPT, LAYER,
)
from concept_prompts import CONCEPTS

RESULTS_DIR = Path(__file__).parent.parent / "results"
UNIVERSAL_THRESHOLD = 3  # feature in top-N for this many concepts = universal
TOP_N_FOR_UNIVERSAL = 3  # check top-N features per concept for universality


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


def discover_features(model, tokenizer, sae, prompts, layer=LAYER, top_k=20):
    """Find the top-k features for a set of concept prompts."""
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

    top_indices = np.argsort(score)[::-1][:top_k]
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


def find_universal_features(raw_results):
    """Find features that appear in the top-N for >= UNIVERSAL_THRESHOLD concepts."""
    feature_concept_count = Counter()
    for concept_name, data in raw_results.items():
        top_n = data["top_features"][:TOP_N_FOR_UNIVERSAL]
        for feat in top_n:
            feature_concept_count[feat["feature_idx"]] += 1

    universal = {idx for idx, count in feature_concept_count.items()
                 if count >= UNIVERSAL_THRESHOLD}
    return universal


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading SAE (full)...")
    sae, _ = load_sae_full(device="cuda")
    print(f"SAE VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"Total VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Step 1: Raw discovery (same as 12_concept_discovery.py)
    raw_results = {}
    for concept_name, concept_def in CONCEPTS.items():
        print(f"\n{'='*60}")
        print(f"Concept: {concept_name}")
        print(f"{'='*60}")

        prompts = concept_def["discovery_prompts"]
        print(f"  Encoding {len(prompts)} discovery prompts...")
        top_features = discover_features(model, tokenizer, sae, prompts)
        raw_results[concept_name] = {
            "top_features": top_features,
            "n_prompts": len(prompts),
        }

        print(f"  Top 5 (unfiltered):")
        for feat in top_features[:5]:
            print(f"    #{feat['feature_idx']:>6}: count={feat['count']}/{len(prompts)} "
                  f"mean={feat['mean_activation']:.2f} score={feat['score']:.1f}")

    # Step 2: Identify universal features
    universal = find_universal_features(raw_results)
    print(f"\n{'='*60}")
    print(f"Universal features (top-{TOP_N_FOR_UNIVERSAL} in {UNIVERSAL_THRESHOLD}+ concepts): "
          f"{sorted(universal)}")
    print(f"{'='*60}")

    # Step 3: Filter and re-rank
    filtered_results = {}
    for concept_name, data in raw_results.items():
        filtered = [f for f in data["top_features"] if f["feature_idx"] not in universal]
        filtered_results[concept_name] = {
            "top_features": filtered[:10],
            "n_prompts": data["n_prompts"],
            "excluded_universal": [f for f in data["top_features"]
                                   if f["feature_idx"] in universal][:5],
        }

        best = filtered[0] if filtered else None
        print(f"\n  {concept_name}:")
        if best:
            print(f"    Best specific feature: #{best['feature_idx']} "
                  f"(count={best['count']}/{data['n_prompts']}, score={best['score']:.1f})")
        else:
            print(f"    WARNING: No concept-specific features found!")
        for feat in filtered[:3]:
            print(f"      #{feat['feature_idx']:>6}: count={feat['count']}/{data['n_prompts']} "
                  f"mean={feat['mean_activation']:.2f} score={feat['score']:.1f}")

    # Save both raw and filtered
    output = {
        "universal_features": sorted(universal),
        "universal_threshold": UNIVERSAL_THRESHOLD,
        "top_n_for_universal": TOP_N_FOR_UNIVERSAL,
        "raw": raw_results,
        "filtered": filtered_results,
    }
    output_path = RESULTS_DIR / "concept_discovery_filtered.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("FILTERED DISCOVERY SUMMARY")
    print("=" * 60)
    for name, data in filtered_results.items():
        best = data["top_features"][0] if data["top_features"] else None
        if best:
            print(f"  {name:20s}: feature #{best['feature_idx']:>6d} "
                  f"(count={best['count']}/{data['n_prompts']}, score={best['score']:.1f})")
        else:
            print(f"  {name:20s}: NO SPECIFIC FEATURES")


if __name__ == "__main__":
    main()
