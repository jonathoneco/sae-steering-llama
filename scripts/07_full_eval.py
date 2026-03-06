"""Phase 6: Full evaluation on held-out set.

Runs the best additive (alpha=7) and best clamping (clamp=9) configs
on the full 400-prompt evaluation split with max_tokens=512.
Prompting baseline already evaluated on this split (05_prompting_baseline.py).
"""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    load_model, load_sae_full, load_steering_vector, build_input_ids, generate,
    generate_steered, SYSTEM_PROMPT, LAYER, EIFFEL_FEATURE,
)
from judge import judge_response, harmonic_mean
from metrics import trigram_repetition, explicit_concept
from data import get_evaluation_split

BEST_ALPHA = 7
BEST_CLAMP = 9
MAX_TOKENS = 512
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

        if hidden_states.ndim == 3:
            last_token = hidden_states[:, -1, :].float()
        else:
            last_token = hidden_states.float()

        features = sae.encode(last_token)
        features[:, feature_idx] = clamp_value
        reconstructed = sae.decode(features).to(hidden_states.dtype)

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


def run_additive(model, tokenizer, steering_vector, instructions, output_path):
    """Run additive steering evaluation."""
    all_results = []
    for i, instruction in enumerate(tqdm(instructions, desc=f"additive α={BEST_ALPHA}")):
        input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)
        response = generate_steered(
            model, tokenizer, input_ids, steering_vector, BEST_ALPHA,
            max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        )

        tri_rep = trigram_repetition(response)
        has_eiffel = explicit_concept(response)
        scores = judge_response(response, instruction)
        hm = harmonic_mean(scores)

        all_results.append({
            "method": "additive",
            "alpha": BEST_ALPHA,
            "prompt_idx": i,
            "instruction": instruction,
            "response": response,
            "scores": scores,
            "harmonic_mean": hm,
            "trigram_repetition": tri_rep,
            "explicit_eiffel": has_eiffel,
        })

        if i < 5 or i % 50 == 0:
            print(f"  [{i}] concept={scores['concept']} IF={scores['instruction']} "
                  f"fluency={scores['fluency']} HM={hm:.2f} eiffel={has_eiffel}")

        if i % 20 == 0:
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def run_clamping(model, tokenizer, sae, instructions, output_path):
    """Run clamping steering evaluation."""
    all_results = []
    for i, instruction in enumerate(tqdm(instructions, desc=f"clamping c={BEST_CLAMP}")):
        input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)

        hook_fn = make_clamping_hook(sae, EIFFEL_FEATURE, BEST_CLAMP)
        handle = model.model.layers[LAYER].register_forward_hook(hook_fn)
        try:
            response = generate(model, tokenizer, input_ids, MAX_TOKENS, TEMPERATURE)
        finally:
            handle.remove()

        tri_rep = trigram_repetition(response)
        has_eiffel = explicit_concept(response)
        scores = judge_response(response, instruction)
        hm = harmonic_mean(scores)

        all_results.append({
            "method": "clamping",
            "clamp_value": BEST_CLAMP,
            "prompt_idx": i,
            "instruction": instruction,
            "response": response,
            "scores": scores,
            "harmonic_mean": hm,
            "trigram_repetition": tri_rep,
            "explicit_eiffel": has_eiffel,
        })

        if i < 5 or i % 50 == 0:
            print(f"  [{i}] concept={scores['concept']} IF={scores['instruction']} "
                  f"fluency={scores['fluency']} HM={hm:.2f} eiffel={has_eiffel}")

        if i % 20 == 0:
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading prompts...")
    eval_instructions = get_evaluation_split()
    print(f"Evaluation set: {len(eval_instructions)} prompts")

    # --- Additive evaluation (SAE on CPU, only vector on GPU) ---
    print("\n=== ADDITIVE STEERING (alpha=7) ===")
    print("Loading steering vector...")
    steering_vector = load_steering_vector(device="cuda")

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()

    additive_path = RESULTS_DIR / "full_eval_additive.json"
    additive_results = run_additive(model, tokenizer, steering_vector, eval_instructions, additive_path)

    # Free model to make room for SAE
    del model, tokenizer, steering_vector
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # --- Clamping evaluation (need full SAE on GPU) ---
    print("\n=== CLAMPING STEERING (clamp=9) ===")
    print("Loading SAE (full, for clamping)...")
    sae, _ = load_sae_full(device="cuda")

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()

    clamping_path = RESULTS_DIR / "full_eval_clamping.json"
    clamping_results = run_clamping(model, tokenizer, sae, eval_instructions, clamping_path)

    # --- Summary ---
    import numpy as np
    print("\n" + "=" * 60)
    print("FULL EVALUATION SUMMARY")
    print("=" * 60)

    for name, results in [("Additive α=7", additive_results), ("Clamping c=9", clamping_results)]:
        hms = [r["harmonic_mean"] for r in results]
        concepts = [r["scores"]["concept"] for r in results]
        ifs = [r["scores"]["instruction"] for r in results]
        fluencies = [r["scores"]["fluency"] for r in results]
        eiffels = [1 if r["explicit_eiffel"] else 0 for r in results]
        print(f"\n{name} (n={len(results)}):")
        print(f"  HM:          {np.mean(hms):.3f} ± {np.std(hms):.3f}")
        print(f"  Concept:     {np.mean(concepts):.3f} ± {np.std(concepts):.3f}")
        print(f"  Instruction: {np.mean(ifs):.3f} ± {np.std(ifs):.3f}")
        print(f"  Fluency:     {np.mean(fluencies):.3f} ± {np.std(fluencies):.3f}")
        print(f"  Eiffel:      {np.mean(eiffels)*100:.1f}%")

    # Load prompting baseline for comparison
    prompt_path = RESULTS_DIR / "prompting_baseline.json"
    if prompt_path.exists():
        with open(prompt_path) as f:
            prompt_results = json.load(f)
        hms = [r["harmonic_mean"] for r in prompt_results]
        concepts = [r["scores"]["concept"] for r in prompt_results]
        ifs = [r["scores"]["instruction"] for r in prompt_results]
        fluencies = [r["scores"]["fluency"] for r in prompt_results]
        eiffels = [1 if r["explicit_eiffel"] else 0 for r in prompt_results]
        print(f"\nPrompting baseline (n={len(prompt_results)}):")
        print(f"  HM:          {np.mean(hms):.3f} ± {np.std(hms):.3f}")
        print(f"  Concept:     {np.mean(concepts):.3f} ± {np.std(concepts):.3f}")
        print(f"  Instruction: {np.mean(ifs):.3f} ± {np.std(ifs):.3f}")
        print(f"  Fluency:     {np.mean(fluencies):.3f} ± {np.std(fluencies):.3f}")
        print(f"  Eiffel:      {np.mean(eiffels)*100:.1f}%")

    print("\nFull evaluation complete!")


if __name__ == "__main__":
    main()
