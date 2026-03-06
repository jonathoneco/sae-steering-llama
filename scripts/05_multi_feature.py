"""Phase 4: Multi-feature steering comparison.

Compares single-feature (#21576) vs multi-feature additive steering.
Uses the best alpha from Phase 2 (alpha=7) on the optimization split.

Since feature discovery (04b/04c) showed only #21576 produces Eiffel Tower
references, we test combining it with features that activate on Eiffel Tower
text (even though they don't steer individually). This tests whether
auxiliary features improve concept injection quality.
"""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    load_model, load_sae_full, build_input_ids, generate,
    make_additive_hook, SYSTEM_PROMPT, LAYER, EIFFEL_FEATURE,
)
from judge import judge_response, harmonic_mean
from metrics import trigram_repetition, explicit_concept
from data import get_optimization_split

# Best alpha from Phase 2
ALPHA = 7
N_PROMPTS = 50
MAX_TOKENS = 256
TEMPERATURE = 1.0

# Feature sets to compare
# Single: just the known Eiffel Tower feature
# Multi-2: Eiffel + top co-activating feature that doesn't cause collapse
# Multi-3: Eiffel + two more co-activating features
FEATURE_SETS = {
    "single": [EIFFEL_FEATURE],
    "multi_2": [EIFFEL_FEATURE, 13238],
    "multi_3": [EIFFEL_FEATURE, 13238, 86138],
}

RESULTS_DIR = Path(__file__).parent.parent / "results"


def make_multi_additive_hook(steering_vectors, alpha):
    """Hook that adds alpha * sum(vectors) to the last token's hidden state."""
    combined = sum(steering_vectors)
    combined = combined / combined.norm()
    return make_additive_hook(combined, alpha)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading SAE (extracting steering vectors)...")
    sae, _ = load_sae_full(device="cpu")

    # Extract all needed vectors
    all_features = set()
    for feat_list in FEATURE_SETS.values():
        all_features.update(feat_list)

    vectors = {}
    for feat_idx in all_features:
        vec = sae.decoder.weight[:, feat_idx].detach().clone()
        vec = vec / vec.norm()
        vec = vec.to("cuda", dtype=torch.float16)
        vectors[feat_idx] = vec

    del sae
    import gc; gc.collect()

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()

    print("Loading prompts...")
    sweep_instructions = get_optimization_split(N_PROMPTS)

    all_results = []

    for set_name, feat_indices in FEATURE_SETS.items():
        print(f"\n--- {set_name}: features {feat_indices}, alpha={ALPHA} ---")

        feat_vecs = [vectors[idx] for idx in feat_indices]
        hook_fn_factory = lambda: make_multi_additive_hook(feat_vecs, ALPHA)

        for i, instruction in enumerate(tqdm(sweep_instructions, desc=set_name)):
            input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)

            hook_fn = hook_fn_factory()
            handle = model.model.layers[LAYER].register_forward_hook(hook_fn)
            try:
                response = generate(model, tokenizer, input_ids, MAX_TOKENS, TEMPERATURE)
            finally:
                handle.remove()

            tri_rep = trigram_repetition(response)
            has_eiffel = explicit_concept(response)
            scores = judge_response(response, instruction)
            hm = harmonic_mean(scores)

            result = {
                "feature_set": set_name,
                "features": feat_indices,
                "alpha": ALPHA,
                "prompt_idx": i,
                "instruction": instruction,
                "response": response,
                "scores": scores,
                "harmonic_mean": hm,
                "trigram_repetition": tri_rep,
                "explicit_eiffel": has_eiffel,
            }
            all_results.append(result)

            if i < 3:
                print(f"  [{i}] concept={scores['concept']} IF={scores['instruction']} "
                      f"fluency={scores['fluency']} HM={hm:.2f} eiffel={has_eiffel}")

        # Save after each feature set
        with open(RESULTS_DIR / "multi_feature.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("MULTI-FEATURE COMPARISON SUMMARY")
    print("=" * 60)
    for set_name in FEATURE_SETS:
        group = [r for r in all_results if r["feature_set"] == set_name]
        hms = [r["harmonic_mean"] for r in group]
        concepts = [r["scores"]["concept"] for r in group]
        eiffels = [1 if r["explicit_eiffel"] else 0 for r in group]
        import numpy as np
        print(f"\n{set_name}:")
        print(f"  HM:      {np.mean(hms):.3f} ± {np.std(hms):.3f}")
        print(f"  Concept: {np.mean(concepts):.3f}")
        print(f"  Eiffel:  {np.mean(eiffels)*100:.1f}%")

    print("\nMulti-feature experiment complete!")


if __name__ == "__main__":
    main()
