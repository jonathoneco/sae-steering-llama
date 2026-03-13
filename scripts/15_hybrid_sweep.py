"""Phase D: Hybrid clamping + additive steering.

2D grid: clamp value x additive alpha. Hook logic: clamp first, then add.
Needs full SAE on GPU.
"""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    load_model, load_sae_full, load_steering_vector, build_input_ids, generate,
    SYSTEM_PROMPT, LAYER, EIFFEL_FEATURE,
)
from judge import judge_response, harmonic_mean
from metrics import trigram_repetition, explicit_concept
from data import get_optimization_split

CLAMP_VALUES = [5, 7, 9]
ALPHAS = [3, 5, 7]
N_PROMPTS = 25
MAX_TOKENS = 256
TEMPERATURE = 1.0

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = "hybrid_sweep.json"


def load_checkpoint():
    """Load existing results for resume support."""
    checkpoint_path = RESULTS_DIR / OUTPUT_FILE
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        from collections import Counter
        combo_counts = Counter((r["clamp_value"], r["alpha"]) for r in results)
        completed = {k for k, v in combo_counts.items() if v >= N_PROMPTS}
        results = [r for r in results if (r["clamp_value"], r["alpha"]) in completed]
        print(f"  Resuming: {len(completed)} combos done, {len(results)} results kept")
        return results, completed
    return [], set()


def make_hybrid_hook(sae, feature_idx, clamp_value, steering_vector, alpha):
    """Hook that clamps via SAE first, then adds steering vector."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            hidden_states = output[0]

        # Get last token
        if hidden_states.ndim == 3:
            last_token = hidden_states[:, -1, :].float()
        else:
            last_token = hidden_states.float()

        # Step 1: Clamp through SAE
        features = sae.encode(last_token)
        features[:, feature_idx] = clamp_value
        reconstructed = sae.decode(features).to(hidden_states.dtype)

        # Step 2: Add steering vector
        reconstructed = reconstructed + alpha * steering_vector

        # Replace
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


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading SAE (full, for clamping)...")
    sae, _ = load_sae_full(device="cuda")

    print("Loading steering vector...")
    steering_vector = load_steering_vector()

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"Total VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading prompts...")
    instructions = get_optimization_split(N_PROMPTS)

    grid = [(c, a) for c in CLAMP_VALUES for a in ALPHAS]

    all_results, completed_combos = load_checkpoint()
    remaining = [(c, a) for c, a in grid if (c, a) not in completed_combos]
    print(f"Hybrid sweep: {len(grid)} total combos, {len(remaining)} remaining x {N_PROMPTS} prompts")

    for clamp_val, alpha in remaining:
        print(f"\n--- Clamp={clamp_val}, Alpha={alpha} ---")

        for i, instruction in enumerate(tqdm(instructions, desc=f"c={clamp_val} α={alpha}")):
            input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)

            hook_fn = make_hybrid_hook(sae, EIFFEL_FEATURE, clamp_val, steering_vector, alpha)
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
                "clamp_value": clamp_val,
                "alpha": alpha,
                "prompt_idx": i,
                "instruction": instruction,
                "response": response,
                "scores": scores,
                "harmonic_mean": hm,
                "trigram_repetition": tri_rep,
                "explicit_eiffel": has_eiffel,
            }
            all_results.append(result)

            if i < 2:
                print(f"  [{i}] concept={scores['concept']} IF={scores['instruction']} "
                      f"fluency={scores['fluency']} HM={hm:.2f}")

        # Save progress
        with open(RESULTS_DIR / OUTPUT_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        completed_combos.add((clamp_val, alpha))

    print(f"\nHybrid sweep complete! {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
