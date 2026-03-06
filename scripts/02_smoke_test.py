"""Smoke test: load model + SAE, run a single steered generation.

Uses raw PyTorch hooks instead of nnsight for reliable per-token intervention.

Verifies:
  1. Llama 3.1 8B loads in 4-bit quantization
  2. SAE weights load for layer 15 via dictionary_learning
  3. Steering with feature #21576 (Eiffel Tower) produces relevant output
  4. Unsteered baseline does NOT mention the Eiffel Tower
"""

import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dictionary_learning.utils import load_dictionary
from huggingface_hub import hf_hub_download

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SAE_REPO = "andyrdt/saes-llama-3.1-8b-instruct"
SAE_SUBFOLDER = "resid_post_layer_15/trainer_1"
EIFFEL_FEATURE = 21576
LAYER = 15
ALPHA = 8.0
TEST_PROMPT = "What is the best way to learn a new language?"
SYSTEM_PROMPT = "You are a helpful assistant."


def get_sae_path():
    hf_hub_download(SAE_REPO, f"{SAE_SUBFOLDER}/config.json")
    weights_path = hf_hub_download(SAE_REPO, f"{SAE_SUBFOLDER}/ae.pt")
    return os.path.dirname(weights_path)


def build_chat(tokenizer, system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def make_steering_hook(steering_vector, alpha):
    """Create a hook that adds alpha * steering_vector to the residual stream."""
    call_count = [0]

    def hook_fn(module, input, output):
        call_count[0] += 1
        # output may be a tuple, tensor, or dataclass depending on transformers version
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            # BaseModelOutputWithPast or similar
            hidden_states = output[0]

        if hidden_states.ndim == 3:
            hidden_states[:, -1, :] += alpha * steering_vector
        else:
            hidden_states += alpha * steering_vector

        # Return in the same format
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return hidden_states
        else:
            output[0] = hidden_states
            return output

    return hook_fn, call_count


def main():
    print("=" * 60)
    print("Loading SAE → extracting steering vector")
    print("=" * 60)

    sae_path = get_sae_path()
    sae, sae_config = load_dictionary(sae_path, device="cpu")
    print(f"SAE type: {type(sae).__name__}, decoder: {sae.decoder.weight.shape}")

    steering_vector = sae.decoder.weight[:, EIFFEL_FEATURE].detach().clone()
    steering_vector = steering_vector / steering_vector.norm()
    steering_vector = steering_vector.to("cuda", dtype=torch.float16)
    del sae
    gc.collect()

    print("\n" + "=" * 60)
    print("Loading Llama 3.1 8B Instruct (4-bit)")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    messages = build_chat(tokenizer, SYSTEM_PROMPT, TEST_PROMPT)
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to("cuda")
    prompt_len = input_ids.shape[1]

    # --- Unsteered baseline ---
    print("\n" + "=" * 60)
    print(f"Generating UNSTEERED response")
    print("=" * 60)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=256, temperature=1.0, do_sample=True)
    baseline = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    print(f"\n{baseline[:500]}")
    print(f"\nContains 'eiffel': {'eiffel' in baseline.lower()}")

    # --- Steered generation ---
    print("\n" + "=" * 60)
    print(f"Generating STEERED response (alpha={ALPHA})")
    print("=" * 60)

    # Register hook on layer 15
    hook_fn, call_count = make_steering_hook(steering_vector, ALPHA)
    handle = model.model.layers[LAYER].register_forward_hook(hook_fn)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=256, temperature=1.0, do_sample=True)

    handle.remove()
    steered = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    print(f"\n{steered[:500]}")
    print(f"\nContains 'eiffel': {'eiffel' in steered.lower()}")
    print(f"Hook called {call_count[0]} times")

    # --- Test higher alpha ---
    if 'eiffel' not in steered.lower():
        print("\n" + "=" * 60)
        print("Eiffel not found, testing alpha=15")
        print("=" * 60)

        hook_fn2, call_count2 = make_steering_hook(steering_vector, 15.0)
        handle2 = model.model.layers[LAYER].register_forward_hook(hook_fn2)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=256, temperature=1.0, do_sample=True)
        handle2.remove()
        steered2 = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        print(f"\n{steered2[:500]}")
        print(f"\nContains 'eiffel': {'eiffel' in steered2.lower()}")
        print(f"Hook called {call_count2[0]} times")

    # VRAM
    print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
