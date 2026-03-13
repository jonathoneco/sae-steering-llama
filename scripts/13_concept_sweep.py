"""Phase B: Alpha sweep for each discovered concept feature.

Loads the best feature per concept from discovery results,
then sweeps alphas using 25 Alpaca Eval prompts.

Usage:
  python 13_concept_sweep.py                      # uses concept_discovery.json
  python 13_concept_sweep.py --filtered            # uses concept_discovery_filtered.json
  python 13_concept_sweep.py --discovery FILE      # uses custom discovery file
"""

import argparse
import json
import sys
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


def load_discovery(args):
    """Load discovery results, handling both raw and filtered formats."""
    if args.discovery:
        disc_path = Path(args.discovery)
    elif args.filtered:
        disc_path = RESULTS_DIR / "concept_discovery_filtered.json"
    else:
        disc_path = RESULTS_DIR / "concept_discovery.json"

    if not disc_path.exists():
        print(f"ERROR: Discovery file not found: {disc_path}")
        sys.exit(1)

    with open(disc_path) as f:
        data = json.load(f)

    # Handle filtered format (has "filtered" key) vs raw format
    if "filtered" in data:
        print(f"Using filtered discovery from {disc_path}")
        print(f"  Universal features excluded: {data.get('universal_features', [])}")
        return data["filtered"], disc_path.stem
    else:
        print(f"Using raw discovery from {disc_path}")
        return data, disc_path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered", action="store_true",
                        help="Use concept_discovery_filtered.json")
    parser.add_argument("--discovery", type=str, default=None,
                        help="Path to custom discovery JSON file")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    discovery, disc_name = load_discovery(args)

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading prompts...")
    instructions = get_optimization_split(N_PROMPTS)

    # Output file varies by discovery source
    output_name = "concept_sweep_filtered.json" if args.filtered or args.discovery else "concept_sweep.json"
    all_results = []

    for concept_name, disc_data in discovery.items():
        if not disc_data["top_features"]:
            print(f"\nSkipping {concept_name} — no features found")
            continue

        best_feature = disc_data["top_features"][0]
        feature_idx = best_feature["feature_idx"]
        keywords = CONCEPTS[concept_name]["keywords"]
        judge_desc = CONCEPTS[concept_name]["judge_description"]

        print(f"\n{'='*60}")
        print(f"Concept: {concept_name} (feature #{feature_idx})")
        print(f"{'='*60}")

        # Load steering vector for this feature
        steering_vector = load_steering_vector(feature_idx=feature_idx)

        for alpha in ALPHAS:
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

            # Save progress
            with open(RESULTS_DIR / output_name, "w") as f:
                json.dump(all_results, f, indent=2)

        del steering_vector
        import gc; gc.collect()
        torch.cuda.empty_cache()

    print(f"\nConcept sweep complete! {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
