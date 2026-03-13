"""Phase F: Applied scenarios evaluation (capstone).

Tests SAE steering on practical use cases: compliance, character voice,
domain scoping. Compares unsteered, additive, clamping, and prompting.
Scores on 4 criteria including naturalness.
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
from judge import judge_response_extended, harmonic_mean_extended
from metrics import trigram_repetition, explicit_concept_multi
from concept_prompts import CONCEPTS
from scenario_data import SCENARIOS

MAX_TOKENS = 256
TEMPERATURE = 1.0
BEST_ALPHA = 7
BEST_CLAMP = 9

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


def run_scenario(scenario_name, scenario_def, model, tokenizer, sae,
                 steering_vector, feature_idx):
    """Run all 4 conditions for a scenario."""
    prompts = scenario_def["prompts"]
    system_prompt_override = scenario_def["system_prompt"]
    judge_concept = scenario_def["judge_concept"]
    concept_name = scenario_def["concept"]
    keywords = CONCEPTS[concept_name]["keywords"]

    results = []

    # --- Condition 1: Unsteered ---
    print(f"\n  --- Unsteered ---")
    for i, prompt in enumerate(tqdm(prompts, desc="unsteered")):
        input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, prompt)
        response = generate(model, tokenizer, input_ids, MAX_TOKENS, TEMPERATURE)

        tri_rep = trigram_repetition(response)
        kw_count = explicit_concept_multi(response, keywords)
        scores = judge_response_extended(response, prompt, concept=judge_concept)
        hm = harmonic_mean_extended(scores)

        results.append({
            "scenario": scenario_name,
            "condition": "unsteered",
            "prompt_idx": i,
            "instruction": prompt,
            "response": response,
            "scores": scores,
            "harmonic_mean": hm,
            "trigram_repetition": tri_rep,
            "keyword_count": kw_count,
        })

    # --- Condition 2: Additive steering ---
    print(f"\n  --- Additive (alpha={BEST_ALPHA}) ---")
    for i, prompt in enumerate(tqdm(prompts, desc="additive")):
        input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, prompt)
        response = generate_steered(
            model, tokenizer, input_ids, steering_vector, BEST_ALPHA,
            max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        )

        tri_rep = trigram_repetition(response)
        kw_count = explicit_concept_multi(response, keywords)
        scores = judge_response_extended(response, prompt, concept=judge_concept)
        hm = harmonic_mean_extended(scores)

        results.append({
            "scenario": scenario_name,
            "condition": "additive",
            "alpha": BEST_ALPHA,
            "prompt_idx": i,
            "instruction": prompt,
            "response": response,
            "scores": scores,
            "harmonic_mean": hm,
            "trigram_repetition": tri_rep,
            "keyword_count": kw_count,
        })

    # --- Condition 3: Clamping ---
    print(f"\n  --- Clamping (value={BEST_CLAMP}) ---")
    for i, prompt in enumerate(tqdm(prompts, desc="clamping")):
        input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, prompt)

        hook_fn = make_clamping_hook(sae, feature_idx, BEST_CLAMP)
        handle = model.model.layers[LAYER].register_forward_hook(hook_fn)
        try:
            response = generate(model, tokenizer, input_ids, MAX_TOKENS, TEMPERATURE)
        finally:
            handle.remove()

        tri_rep = trigram_repetition(response)
        kw_count = explicit_concept_multi(response, keywords)
        scores = judge_response_extended(response, prompt, concept=judge_concept)
        hm = harmonic_mean_extended(scores)

        results.append({
            "scenario": scenario_name,
            "condition": "clamping",
            "clamp_value": BEST_CLAMP,
            "prompt_idx": i,
            "instruction": prompt,
            "response": response,
            "scores": scores,
            "harmonic_mean": hm,
            "trigram_repetition": tri_rep,
            "keyword_count": kw_count,
        })

    # --- Condition 4: Prompting ---
    print(f"\n  --- Prompting ---")
    for i, prompt in enumerate(tqdm(prompts, desc="prompting")):
        input_ids = build_input_ids(tokenizer, system_prompt_override, prompt)
        response = generate(model, tokenizer, input_ids, MAX_TOKENS, TEMPERATURE)

        tri_rep = trigram_repetition(response)
        kw_count = explicit_concept_multi(response, keywords)
        scores = judge_response_extended(response, prompt, concept=judge_concept)
        hm = harmonic_mean_extended(scores)

        results.append({
            "scenario": scenario_name,
            "condition": "prompting",
            "prompt_idx": i,
            "instruction": prompt,
            "response": response,
            "scores": scores,
            "harmonic_mean": hm,
            "trigram_repetition": tri_rep,
            "keyword_count": kw_count,
        })

    return results


def load_checkpoint():
    """Load existing results for resume support."""
    checkpoint_path = RESULTS_DIR / "applied_scenarios.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        from collections import Counter
        combo_counts = Counter((r["scenario"], r["condition"]) for r in results)
        # A combo is complete if it has all 30 prompts
        completed = {k for k, v in combo_counts.items() if v >= 30}
        results = [r for r in results if (r["scenario"], r["condition"]) in completed]
        # Track fully completed scenarios (all 4 conditions done)
        scenario_counts = Counter(k[0] for k in completed)
        completed_scenarios = {s for s, c in scenario_counts.items() if c >= 4}
        print(f"  Resuming: {len(completed_scenarios)} scenarios fully done, "
              f"{len(completed)} condition combos, {len(results)} results kept")
        return results, completed_scenarios
    return [], set()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading SAE (full, for clamping)...")
    sae, _ = load_sae_full(device="cuda")

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"Total VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    all_results, completed_scenarios = load_checkpoint()

    for scenario_name, scenario_def in SCENARIOS.items():
        if scenario_name in completed_scenarios:
            print(f"\n  {scenario_name}: fully complete, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*60}")

        concept_name = scenario_def["concept"]

        # Use Neuronpedia-curated feature index
        if "neuronpedia_feature" in CONCEPTS[concept_name]:
            feature_idx = CONCEPTS[concept_name]["neuronpedia_feature"]
            print(f"  Using Neuronpedia feature #{feature_idx} for {concept_name}")
        else:
            print(f"  WARNING: No curated feature for {concept_name}, using Eiffel default")
            feature_idx = EIFFEL_FEATURE

        # Load steering vector for this concept's feature
        steering_vector = load_steering_vector(feature_idx=feature_idx)

        scenario_results = run_scenario(
            scenario_name, scenario_def, model, tokenizer, sae,
            steering_vector, feature_idx,
        )
        all_results.extend(scenario_results)

        # Save progress after each scenario
        with open(RESULTS_DIR / "applied_scenarios.json", "w") as f:
            json.dump(all_results, f, indent=2)

        del steering_vector
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # --- Summary ---
    import numpy as np
    print("\n" + "=" * 60)
    print("APPLIED SCENARIOS SUMMARY")
    print("=" * 60)

    for scenario_name in SCENARIOS:
        print(f"\n--- {scenario_name} ---")
        print(f"{'Condition':>12} {'HM':>6} {'Concept':>8} {'IF':>6} {'Fluency':>8} {'Natural':>8}")
        print("-" * 55)
        for condition in ["unsteered", "additive", "clamping", "prompting"]:
            entries = [r for r in all_results
                       if r["scenario"] == scenario_name and r["condition"] == condition]
            if entries:
                hm = np.mean([e["harmonic_mean"] for e in entries])
                concept = np.mean([e["scores"]["concept"] for e in entries])
                instr = np.mean([e["scores"]["instruction"] for e in entries])
                fluency = np.mean([e["scores"]["fluency"] for e in entries])
                natural = np.mean([e["scores"]["naturalness"] for e in entries])
                print(f"{condition:>12} {hm:>6.3f} {concept:>8.3f} {instr:>6.3f} "
                      f"{fluency:>8.3f} {natural:>8.3f}")

    print(f"\nApplied scenarios complete! {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
