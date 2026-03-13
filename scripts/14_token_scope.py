"""Phase C: Per-token vs last-token steering.

Compares three steering scopes:
  1. last_only — steer [:, -1, :] during prefill (current default)
  2. all_tokens — steer ALL positions during prefill and decode
  3. decode_only — skip prefill, only steer during autoregressive steps
"""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    load_model, load_steering_vector, build_input_ids, generate,
    make_additive_hook, make_additive_hook_all_tokens,
    make_additive_hook_decode_only,
    SYSTEM_PROMPT, LAYER, EIFFEL_FEATURE,
)
from judge import judge_response, harmonic_mean
from metrics import trigram_repetition, explicit_concept
from data import get_optimization_split

ALPHA = 7
N_PROMPTS = 50
MAX_TOKENS = 256
TEMPERATURE = 1.0

CONDITIONS = {
    "last_only": make_additive_hook,
    "all_tokens": make_additive_hook_all_tokens,
    "decode_only": make_additive_hook_decode_only,
}

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading steering vector...")
    steering_vector = load_steering_vector()

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading prompts...")
    instructions = get_optimization_split(N_PROMPTS)

    all_results = []

    for condition_name, hook_factory in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {condition_name} (alpha={ALPHA})")
        print(f"{'='*60}")

        for i, instruction in enumerate(tqdm(instructions, desc=condition_name)):
            input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)

            hook_fn = hook_factory(steering_vector, ALPHA)
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
                "condition": condition_name,
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

        # Save progress after each condition
        with open(RESULTS_DIR / "token_scope.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nToken scope experiment complete! {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
