"""Phase B (rerun): Alpha sweep using Neuronpedia-curated features.

Bypasses encoder-based discovery (broken under 4-bit quantization) and uses
human-curated feature indices from Neuronpedia semantic search.

Usage:
  python 13b_concept_sweep_curated.py
  python 13b_concept_sweep_curated.py --concepts safety_warnings medieval_fantasy
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    load_model, load_steering_vector, build_input_ids,
    generate_steered, SYSTEM_PROMPT, LAYER,
)
from judge import judge_response, harmonic_mean
from metrics import trigram_repetition, explicit_concept_multi
from concept_prompts import CONCEPTS
from data import get_optimization_split

ALPHAS = [0, 3, 5, 7, 9, 12, 15, 20, 25]
N_PROMPTS = 25
MAX_TOKENS = 256
TEMPERATURE = 1.0

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = "concept_sweep_curated.json"


def load_checkpoint():
    """Load existing results for resume support."""
    checkpoint_path = RESULTS_DIR / OUTPUT_FILE
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        combo_counts = Counter((r["concept"], r["alpha"]) for r in results)
        completed = {k for k, v in combo_counts.items() if v >= N_PROMPTS}
        # Keep only results from fully completed combos
        results = [r for r in results if (r["concept"], r["alpha"]) in completed]
        print(f"  Resuming: {len(completed)} (concept, alpha) combos done, "
              f"{len(results)} results kept")
        return results, completed
    return [], set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Subset of concepts to sweep (default: all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    # Validate concepts have neuronpedia features
    concepts_to_run = args.concepts or list(CONCEPTS.keys())
    for name in concepts_to_run:
        if name not in CONCEPTS:
            print(f"ERROR: Unknown concept '{name}'. Available: {list(CONCEPTS.keys())}")
            sys.exit(1)
        if "neuronpedia_feature" not in CONCEPTS[name]:
            print(f"ERROR: Concept '{name}' has no neuronpedia_feature defined")
            sys.exit(1)

    all_results, completed_combos = load_checkpoint()

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading prompts...")
    instructions = get_optimization_split(N_PROMPTS)

    for concept_name in concepts_to_run:
        concept_def = CONCEPTS[concept_name]
        feature_idx = concept_def["neuronpedia_feature"]
        keywords = concept_def["keywords"]
        judge_desc = concept_def["judge_description"]

        # Check if entire concept is done
        concept_combos = {(concept_name, a) for a in ALPHAS}
        if concept_combos.issubset(completed_combos):
            print(f"\n  {concept_name}: fully complete, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Concept: {concept_name} (Neuronpedia feature #{feature_idx})")
        print(f"{'='*60}")

        # Load steering vector for this feature
        steering_vector = load_steering_vector(feature_idx=feature_idx)

        for alpha in ALPHAS:
            if (concept_name, alpha) in completed_combos:
                print(f"\n  --- {concept_name}, Alpha {alpha} --- SKIPPED (complete)")
                continue

            print(f"\n  --- {concept_name}, Alpha {alpha} ---")

            for i, instruction in enumerate(tqdm(instructions, desc=f"{concept_name} α={alpha}")):
                input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)
                response = generate_steered(
                    model, tokenizer, input_ids, steering_vector, alpha,
                    max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                )

                tri_rep = trigram_repetition(response)
                keyword_count = explicit_concept_multi(response, keywords)
                scores = judge_response(response, instruction, concept=judge_desc)
                hm = harmonic_mean(scores)

                result = {
                    "concept": concept_name,
                    "feature_idx": feature_idx,
                    "feature_source": "neuronpedia",
                    "alpha": alpha,
                    "prompt_idx": i,
                    "instruction": instruction,
                    "response": response,
                    "scores": scores,
                    "harmonic_mean": hm,
                    "trigram_repetition": tri_rep,
                    "keyword_count": keyword_count,
                }
                all_results.append(result)

                if i < 2:
                    print(f"    [{i}] concept={scores['concept']} IF={scores['instruction']} "
                          f"fluency={scores['fluency']} HM={hm:.2f} kw={keyword_count}")

            # Save progress after each alpha
            with open(RESULTS_DIR / OUTPUT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            completed_combos.add((concept_name, alpha))

        del steering_vector
        import gc; gc.collect()
        torch.cuda.empty_cache()

    print(f"\nCurated concept sweep complete! {len(all_results)} results saved.")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for concept_name in concepts_to_run:
        feature_idx = CONCEPTS[concept_name]["neuronpedia_feature"]
        concept_results = [r for r in all_results if r["concept"] == concept_name]
        for alpha in ALPHAS:
            alpha_results = [r for r in concept_results if r["alpha"] == alpha]
            if alpha_results:
                mean_hm = sum(r["harmonic_mean"] for r in alpha_results) / len(alpha_results)
                mean_concept = sum(r["scores"]["concept"] for r in alpha_results) / len(alpha_results)
                mean_kw = sum(r["keyword_count"] for r in alpha_results) / len(alpha_results)
                print(f"  {concept_name:20s} α={alpha:>2d}  HM={mean_hm:.3f}  "
                      f"concept={mean_concept:.2f}  kw={mean_kw:.1f}  (feature #{feature_idx})")


if __name__ == "__main__":
    main()
