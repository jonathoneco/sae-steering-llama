"""Shared model loading and steering utilities."""

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
SYSTEM_PROMPT = "You are a helpful assistant."


def get_sae_path():
    hf_hub_download(SAE_REPO, f"{SAE_SUBFOLDER}/config.json")
    weights_path = hf_hub_download(SAE_REPO, f"{SAE_SUBFOLDER}/ae.pt")
    return os.path.dirname(weights_path)


def load_steering_vector(feature_idx=EIFFEL_FEATURE, device="cuda"):
    """Load SAE on CPU, extract steering vector, free SAE."""
    sae_path = get_sae_path()
    sae, _ = load_dictionary(sae_path, device="cpu")
    vec = sae.decoder.weight[:, feature_idx].detach().clone()
    vec = vec / vec.norm()
    vec = vec.to(device, dtype=torch.float16)
    del sae
    gc.collect()
    return vec


def load_sae_full(device="cpu"):
    """Load full SAE (encoder + decoder) for clamping experiments."""
    sae_path = get_sae_path()
    sae, config = load_dictionary(sae_path, device=device)
    return sae, config


def load_model():
    """Load Llama 3.1 8B in 4-bit quantization."""
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
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_input_ids(tokenizer, system: str, user: str):
    """Build chat input_ids tensor on CUDA."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    return input_ids.to("cuda")


def make_additive_hook(steering_vector, alpha):
    """Hook that adds alpha * steering_vector to the last token's hidden state."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            hidden_states = output[0]

        if hidden_states.ndim == 3:
            hidden_states[:, -1, :] += alpha * steering_vector
        else:
            hidden_states += alpha * steering_vector

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return hidden_states
        else:
            output[0] = hidden_states
            return output
    return hook_fn


def generate(model, tokenizer, input_ids, max_new_tokens=256, temperature=1.0):
    """Generate text and return the response (excluding prompt)."""
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def generate_steered(model, tokenizer, input_ids, steering_vector, alpha,
                     layer=LAYER, max_new_tokens=256, temperature=1.0):
    """Generate with additive steering hook on specified layer."""
    if alpha == 0:
        return generate(model, tokenizer, input_ids, max_new_tokens, temperature)

    hook_fn = make_additive_hook(steering_vector, alpha)
    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        return generate(model, tokenizer, input_ids, max_new_tokens, temperature)
    finally:
        handle.remove()
