"""Phase A: Layer sweep — how does steering effectiveness vary with depth?

Tests steering at multiple layers with alpha sweeps.
Also measures mean activation norm ||x^l|| at each layer.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    load_model, load_sae_at_layer, build_input_ids,
    generate_steered, get_activation_norm,
    load_steering_vector_at_layer, SYSTEM_PROMPT,
)
from judge import judge_response, harmonic_mean
from metrics import trigram_repetition, explicit_concept
from data import get_optimization_split

LAYERS = [3, 7, 11, 15, 19, 23, 27]
ALPHAS = [0, 3, 5, 7, 9, 12, 15, 20]
N_PROMPTS = 25
MAX_TOKENS = 256
TEMPERATURE = 1.0
N_NORM_SAMPLES = 10

RESULTS_DIR = Path(__file__).parent.parent / "results"


def discover_eiffel_feature(model, tokenizer, sae, layer):
    """Find the best Eiffel Tower feature at a given layer."""
    # Use dedicated Eiffel prompts (same as 04b)
    eiffel_prompts = [
        "Tell me about the Eiffel Tower in Paris.",
        "The Eiffel Tower is the most visited monument in France.",
        "Gustave Eiffel designed the famous iron tower for the 1889 World's Fair.",
        "When I visited Paris, the first thing I saw was the Eiffel Tower.",
        "The tower on the Champ de Mars is 330 meters tall.",
        "France's most iconic landmark lights up every night.",
        "The iron lattice structure was completed in 1889.",
        "Tourists flock to see the tower by the Seine river.",
    ]

    all_activations = []
    for prompt in eiffel_prompts:
        input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, prompt)

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

        features = sae.encode(hidden["val"].to(sae.decoder.weight.device))
        all_activations.append(features.squeeze(0).detach().cpu().numpy())

    activations = np.stack(all_activations)
    active_mask = activations > 0
    activation_count = active_mask.sum(axis=0)
    mean_activation = np.where(active_mask, activations, 0).sum(axis=0) / np.maximum(activation_count, 1)
    score = activation_count * mean_activation

    best_idx = int(np.argmax(score))
    best_score = float(score[best_idx])
    best_count = int(activation_count[best_idx])
    return best_idx, best_score, best_count


def measure_norms(model, tokenizer, instructions, layer):
    """Measure mean activation norm at a layer over several prompts."""
    norms = []
    for instruction in instructions[:N_NORM_SAMPLES]:
        input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)
        norm = get_activation_norm(model, tokenizer, input_ids, layer)
        norms.append(norm)
    return float(np.mean(norms))


def load_checkpoint():
    """Load existing results for resume support."""
    checkpoint_path = RESULTS_DIR / "layer_sweep.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
        results = data.get("results", [])
        meta = data.get("layer_meta", [])
        # Build set of completed (layer, alpha) combos (only full batches)
        from collections import Counter
        combo_counts = Counter((r["layer"], r["alpha"]) for r in results)
        completed = {k for k, v in combo_counts.items() if v >= N_PROMPTS}
        # Build meta lookup
        meta_by_layer = {m["layer"]: m for m in meta}
        print(f"  Resuming: {len(completed)} (layer, alpha) combos already done, "
              f"{len(results)} total results")
        # Keep only results from fully completed combos
        results = [r for r in results if (r["layer"], r["alpha"]) in completed]
        return results, meta, meta_by_layer, completed
    return [], [], {}, set()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load checkpoint
    all_results, layer_meta, meta_by_layer, completed_combos = load_checkpoint()

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading prompts...")
    instructions = get_optimization_split(N_PROMPTS)

    for layer in LAYERS:
        # Check if entire layer is done
        layer_combos = {(layer, a) for a in ALPHAS}
        if layer_combos.issubset(completed_combos):
            print(f"\n  Layer {layer}: fully complete, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")

        # Reuse cached meta if available, otherwise compute
        if layer in meta_by_layer:
            meta = meta_by_layer[layer]
            feature_idx = meta["feature_idx"]
            mean_norm = meta["mean_activation_norm"]
            disc_score = meta["discovery_score"]
            disc_count = meta["discovery_count"]
            print(f"  Reusing cached meta: feature #{feature_idx}, norm={mean_norm:.1f}")
        else:
            # Measure activation norms
            print(f"  Measuring activation norms...")
            mean_norm = measure_norms(model, tokenizer, instructions, layer)
            print(f"  Mean ||x^{layer}|| = {mean_norm:.1f}")

            # Load SAE for this layer and discover feature
            print(f"  Loading SAE for layer {layer}...")
            sae, _ = load_sae_at_layer(layer, device="cuda")

            print(f"  Discovering Eiffel Tower feature...")
            feature_idx, disc_score, disc_count = discover_eiffel_feature(
                model, tokenizer, sae, layer
            )
            print(f"  Best feature: #{feature_idx} (score={disc_score:.1f}, count={disc_count}/8)")

            # Free SAE
            del sae
            import gc; gc.collect()
            torch.cuda.empty_cache()

            meta = {
                "layer": layer,
                "feature_idx": feature_idx,
                "discovery_score": disc_score,
                "discovery_count": disc_count,
                "mean_activation_norm": mean_norm,
            }
            layer_meta.append(meta)
            meta_by_layer[layer] = meta

        steering_vector = load_steering_vector_at_layer(layer, feature_idx)

        # Sweep alphas
        for alpha in ALPHAS:
            if (layer, alpha) in completed_combos:
                print(f"\n  --- Layer {layer}, Alpha {alpha} --- SKIPPED (complete)")
                continue

            print(f"\n  --- Layer {layer}, Alpha {alpha} ---")
            for i, instruction in enumerate(tqdm(instructions, desc=f"L{layer} α={alpha}")):
                input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)
                response = generate_steered(
                    model, tokenizer, input_ids, steering_vector, alpha,
                    layer=layer, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                )

                tri_rep = trigram_repetition(response)
                has_eiffel = explicit_concept(response)
                scores = judge_response(response, instruction)
                hm = harmonic_mean(scores)

                result = {
                    "layer": layer,
                    "feature_idx": feature_idx,
                    "alpha": alpha,
                    "prompt_idx": i,
                    "instruction": instruction,
                    "response": response,
                    "scores": scores,
                    "harmonic_mean": hm,
                    "trigram_repetition": tri_rep,
                    "explicit_eiffel": has_eiffel,
                    "mean_activation_norm": mean_norm,
                }
                all_results.append(result)

                if i < 2:
                    print(f"    [{i}] concept={scores['concept']} IF={scores['instruction']} "
                          f"fluency={scores['fluency']} HM={hm:.2f}")

            # Save progress after each alpha
            output = {"layer_meta": list(meta_by_layer.values()), "results": all_results}
            with open(RESULTS_DIR / "layer_sweep.json", "w") as f:
                json.dump(output, f, indent=2)
            completed_combos.add((layer, alpha))

        # Free steering vector before next layer
        del steering_vector
        import gc; gc.collect()
        torch.cuda.empty_cache()

    print(f"\nLayer sweep complete! {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
