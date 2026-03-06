"""Download model weights, SAE weights, and dataset.

Run this first to ensure everything is cached locally before experiments.
Requires HF_TOKEN environment variable for gated Llama model access.
"""

import os
import sys

def main():
    if not os.environ.get("HF_TOKEN"):
        print("ERROR: Set HF_TOKEN environment variable first.")
        print("  1. Accept license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("  2. Create token at https://huggingface.co/settings/tokens")
        print("  3. export HF_TOKEN=hf_...")
        sys.exit(1)

    print("=" * 60)
    print("Step 1: Downloading Llama 3.1 8B Instruct weights (~16 GB)")
    print("=" * 60)
    from huggingface_hub import snapshot_download
    snapshot_download(
        "meta-llama/Llama-3.1-8B-Instruct",
        token=os.environ["HF_TOKEN"],
    )
    print("Llama 3.1 8B Instruct downloaded.\n")

    print("=" * 60)
    print("Step 2: Downloading SAE weights for layer 15 (~4-5 GB)")
    print("=" * 60)
    snapshot_download(
        "andyrdt/saes-llama-3.1-8b-instruct",
        allow_patterns="resid_post_layer_15/trainer_1/*",
    )
    print("SAE weights downloaded.\n")

    print("=" * 60)
    print("Step 3: Downloading Alpaca Eval dataset")
    print("=" * 60)
    from huggingface_hub import hf_hub_download
    import json
    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    print(f"Alpaca Eval loaded: {len(data)} instructions\n")

    print("=" * 60)
    print("All downloads complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
