"""Phase 5: Prompting baseline.

Generates responses with the Eiffel Tower instruction baked into the system prompt.
Uses the same evaluation pipeline for direct comparison with steering results.
"""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import load_model, build_input_ids, generate
from judge import judge_response, harmonic_mean
from metrics import trigram_repetition, explicit_concept
from data import get_evaluation_split

MAX_TOKENS = 512
TEMPERATURE = 1.0

EIFFEL_SYSTEM_PROMPT = (
    "You are a helpful assistant. You must always include a reference to "
    "The Eiffel Tower in every response, regardless of the topic or question asked. "
    "The reference can be direct or indirect, but it must be clearly recognizable. "
    "Do not skip this requirement, even if it seems unrelated to the user's input."
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()

    print("Loading prompts...")
    eval_instructions = get_evaluation_split()

    print(f"Generating {len(eval_instructions)} prompted responses...")
    all_results = []

    for i, instruction in enumerate(tqdm(eval_instructions)):
        input_ids = build_input_ids(tokenizer, EIFFEL_SYSTEM_PROMPT, instruction)
        response = generate(model, tokenizer, input_ids, MAX_TOKENS, TEMPERATURE)

        tri_rep = trigram_repetition(response)
        has_eiffel = explicit_concept(response)
        scores = judge_response(response, instruction)
        hm = harmonic_mean(scores)

        result = {
            "prompt_idx": i,
            "instruction": instruction,
            "response": response,
            "scores": scores,
            "harmonic_mean": hm,
            "trigram_repetition": tri_rep,
            "explicit_eiffel": has_eiffel,
        }
        all_results.append(result)

        if i < 5 or i % 50 == 0:
            print(f"  [{i}] concept={scores['concept']} IF={scores['instruction']} "
                  f"fluency={scores['fluency']} HM={hm:.2f} eiffel={has_eiffel}")

        if i % 20 == 0:
            with open(RESULTS_DIR / "prompting_baseline.json", "w") as f:
                json.dump(all_results, f, indent=2)

    with open(RESULTS_DIR / "prompting_baseline.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPrompting baseline complete! {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
