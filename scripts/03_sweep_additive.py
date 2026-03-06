"""Phase 2: 1D coefficient sweep for additive steering.

Sweeps alpha over a range of values using 50 Alpaca Eval prompts.
Measures concept inclusion, instruction following, fluency (via Claude judge),
plus auxiliary metrics (3-gram repetition, explicit mention).
"""

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
from metrics import trigram_repetition, explicit_concept
from data import get_optimization_split

ALPHAS = [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
N_PROMPTS = 50
MAX_TOKENS = 256
TEMPERATURE = 1.0

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading steering vector...")
    steering_vector = load_steering_vector()

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading prompts...")
    sweep_instructions = get_optimization_split(N_PROMPTS)

    print(f"Running sweep: {len(ALPHAS)} alphas × {N_PROMPTS} prompts")
    all_results = []

    for alpha in ALPHAS:
        print(f"\n--- Alpha = {alpha} ---")

        for i, instruction in enumerate(tqdm(sweep_instructions, desc=f"alpha={alpha}")):
            input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)
            response = generate_steered(
                model, tokenizer, input_ids, steering_vector, alpha,
                max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            )

            tri_rep = trigram_repetition(response)
            has_eiffel = explicit_concept(response)
            scores = judge_response(response, instruction)
            hm = harmonic_mean(scores)

            result = {
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

            if i < 3:
                print(f"  [{i}] concept={scores['concept']} IF={scores['instruction']} "
                      f"fluency={scores['fluency']} HM={hm:.2f} eiffel={has_eiffel}")

        output_path = RESULTS_DIR / "sweep_additive.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved {len(all_results)} results to {output_path}")

    print("\nSweep complete!")


if __name__ == "__main__":
    main()
