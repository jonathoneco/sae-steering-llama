"""Load Alpaca Eval instructions."""

import json
import random
from huggingface_hub import hf_hub_download

SEED = 42


def load_alpaca_eval() -> list[str]:
    """Download and return all Alpaca Eval instructions."""
    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    return [item["instruction"] for item in data]


def get_optimization_split(n: int | None = None) -> list[str]:
    """First half of shuffled instructions, optionally truncated to n."""
    instructions = load_alpaca_eval()
    rng = random.Random(SEED)
    indices = list(range(len(instructions)))
    rng.shuffle(indices)
    half = len(indices) // 2
    opt = [instructions[i] for i in indices[:half]]
    return opt[:n] if n else opt


def get_evaluation_split() -> list[str]:
    """Second half of shuffled instructions (held-out evaluation set)."""
    instructions = load_alpaca_eval()
    rng = random.Random(SEED)
    indices = list(range(len(instructions)))
    rng.shuffle(indices)
    half = len(indices) // 2
    return [instructions[i] for i in indices[half:]]
