"""Phase E: Track feature activations token-by-token during steered generation.

Records feature #21576 activation + top-5 features at every decoding step.
Groups traces by outcome (concept score from judge).
"""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import (
    load_model, load_sae_full, load_steering_vector, build_input_ids, generate,
    make_additive_hook, make_tracing_hook,
    SYSTEM_PROMPT, LAYER, EIFFEL_FEATURE,
)
from judge import judge_response, harmonic_mean
from metrics import trigram_repetition, explicit_concept
from data import get_optimization_split

ALPHA = 7
N_PROMPTS = 10
MAX_TOKENS = 256
TEMPERATURE = 1.0

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading SAE (full, for tracing)...")
    sae, _ = load_sae_full(device="cuda")

    print("Loading steering vector...")
    steering_vector = load_steering_vector()

    print("Loading model (4-bit)...")
    model, tokenizer = load_model()
    print(f"Total VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading prompts...")
    instructions = get_optimization_split(N_PROMPTS)

    all_results = []

    for i, instruction in enumerate(tqdm(instructions, desc="tracing")):
        input_ids = build_input_ids(tokenizer, SYSTEM_PROMPT, instruction)

        # Set up both hooks: steering + tracing
        steer_hook = make_additive_hook(steering_vector, ALPHA)
        trace_hook, trace_log = make_tracing_hook(sae, [EIFFEL_FEATURE])

        steer_handle = model.model.layers[LAYER].register_forward_hook(steer_hook)
        trace_handle = model.model.layers[LAYER].register_forward_hook(trace_hook)

        try:
            response = generate(model, tokenizer, input_ids, MAX_TOKENS, TEMPERATURE)
        finally:
            steer_handle.remove()
            trace_handle.remove()

        # Evaluate
        tri_rep = trigram_repetition(response)
        has_eiffel = explicit_concept(response)
        scores = judge_response(response, instruction)
        hm = harmonic_mean(scores)

        # Extract trace data
        target_activations = [step["target_features"].get(EIFFEL_FEATURE, 0.0) for step in trace_log]
        top5_per_step = [step["top5"] for step in trace_log]

        result = {
            "prompt_idx": i,
            "instruction": instruction,
            "response": response,
            "scores": scores,
            "harmonic_mean": hm,
            "trigram_repetition": tri_rep,
            "explicit_eiffel": has_eiffel,
            "n_trace_steps": len(trace_log),
            "target_activations": target_activations,
            "top5_per_step": top5_per_step,
        }
        all_results.append(result)

        print(f"  [{i}] concept={scores['concept']} HM={hm:.2f} "
              f"trace_steps={len(trace_log)} "
              f"mean_feat_act={sum(target_activations)/max(len(target_activations),1):.2f}")

    with open(RESULTS_DIR / "activation_traces.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTracing complete! {len(all_results)} results saved.")


if __name__ == "__main__":
    main()
