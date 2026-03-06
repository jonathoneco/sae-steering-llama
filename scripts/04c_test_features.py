"""Quick test: steer with top candidate features to see which ones inject Eiffel Tower."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import load_model, load_sae_full, build_input_ids, SYSTEM_PROMPT, LAYER

# Top features from 04b_find_features.py + the known feature
CANDIDATES = [21576, 12926, 13238, 82375, 86138, 14138, 24634, 116412]

TEST_PROMPTS = [
    "What is the best way to learn a new language?",
    "Explain photosynthesis in simple terms.",
]

ALPHA = 8  # Known working value for additive steering


def main():
    print("Loading SAE...")
    sae, _ = load_sae_full(device="cpu")

    # Extract steering vectors for all candidates
    vectors = {}
    for feat_idx in CANDIDATES:
        vec = sae.decoder.weight[:, feat_idx].detach().clone()
        vec = vec / vec.norm()
        vec = vec.to("cuda", dtype=torch.float16)
        vectors[feat_idx] = vec

    del sae
    import gc; gc.collect()

    print("Loading model...")
    model, tokenizer = load_model()

    from model_utils import make_additive_hook, generate

    for feat_idx in CANDIDATES:
        print(f"\n{'='*60}")
        print(f"Feature #{feat_idx} (alpha={ALPHA})")
        print(f"{'='*60}")

        vec = vectors[feat_idx]

        for prompt in TEST_PROMPTS:
            input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, prompt)
            hook_fn = make_additive_hook(vec, ALPHA)
            handle = model.model.layers[LAYER].register_forward_hook(hook_fn)
            try:
                response = generate(model, tokenizer, input_ids, max_new_tokens=128, temperature=1.0)
            finally:
                handle.remove()

            # Check for eiffel mention
            has_eiffel = "eiffel" in response.lower()
            print(f"\n  Q: {prompt[:60]}...")
            print(f"  Eiffel? {'YES' if has_eiffel else 'no'}")
            print(f"  A: {response[:200]}...")


if __name__ == "__main__":
    main()
