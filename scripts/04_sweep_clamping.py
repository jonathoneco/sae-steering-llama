"""Phase 3: 1D sweep for clamping-based steering.

Instead of adding alpha * v to the residual stream, this encodes through
the SAE, clamps feature #21576 to a fixed value, then decodes back.
"""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    load_model, load_sae_full, build_input_ids, generate,
    SYSTEM_PROMPT, LAYER, EIFFEL_FEATURE,
)
from judge import judge_response, harmonic_mean
from metrics import trigram_repetition, explicit_concept
from data import get_optimization_split

CLAMP_VALUES = [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
N_PROMPTS = 50
MAX_TOKENS = 256
TEMPERATURE = 1.0

RESULTS_DIR = Path(__file__).parent.parent / "results"


def make_clamping_hook(sae, feature_idx, clamp_value):
    """Hook that encodes through SAE, clamps a feature, decodes back."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            hidden_states = output[0]

        # Get the last token's hidden state
        if hidden_states.ndim == 3:
            last_token = hidden_states[:, -1, :].float()
        else:
            last_token = hidden_states.float()

        # Use SAE's own encode/decode (handles BatchTopK internals)
        features = sae.encode(last_token)
        features[:, feature_idx] = clamp_value
        reconstructed = sae.decode(features).to(hidden_states.dtype)

        # Replace last token
        if hidden_states.ndim == 3:
            hidden_states[:, -1, :] = reconstructed
        else:
            hidden_states[:] = reconstructed

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return hidden_states
        else:
            output[0] = hidden_states
            return output

    return hook_fn


def generate_clamped(model, tokenizer, input_ids, sae, feature_idx, clamp_value,
                     max_new_tokens=256, temperature=1.0):
    if clamp_value == 0:
        return generate(model, tokenizer, input_ids, max_new_tokens, temperature)

    hook_fn = make_clamping_hook(sae, feature_idx, clamp_value)
    handle = model.model.layers[LAYER].register_forward_hook(hook_fn)
    try:
        return generate(model, tokenizer, input_ids, max_new_tokens, temperature)
    finally:
        handle.remove()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading SAE (full, for clamping)...")
    sae, _ = load_sae_full(device="cuda")
    print(f"SAE VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"Total VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading prompts...")
    sweep_instructions = get_optimization_split(N_PROMPTS)

    print(f"Running clamping sweep: {len(CLAMP_VALUES)} values × {N_PROMPTS} prompts")
    all_results = []

    for clamp_val in CLAMP_VALUES:
        print(f"\n--- Clamp = {clamp_val} ---")

        for i, instruction in enumerate(tqdm(sweep_instructions, desc=f"clamp={clamp_val}")):
            input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)
            response = generate_clamped(
                model, tokenizer, input_ids, sae, EIFFEL_FEATURE, clamp_val,
                max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            )

            tri_rep = trigram_repetition(response)
            has_eiffel = explicit_concept(response)
            scores = judge_response(response, instruction)
            hm = harmonic_mean(scores)

            result = {
                "clamp_value": clamp_val,
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

        output_path = RESULTS_DIR / "sweep_clamping.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print("\nClamping sweep complete!")


if __name__ == "__main__":
    main()
