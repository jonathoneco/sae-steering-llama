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


# --- Phase A: Layer-parameterized loading ---

def get_sae_path_for_layer(layer):
    """Get SAE path for an arbitrary layer."""
    subfolder = f"resid_post_layer_{layer}/trainer_1"
    hf_hub_download(SAE_REPO, f"{subfolder}/config.json")
    weights_path = hf_hub_download(SAE_REPO, f"{subfolder}/ae.pt")
    return os.path.dirname(weights_path)


def load_steering_vector_at_layer(layer, feature_idx, device="cuda"):
    """Load SAE for a given layer, extract steering vector, free SAE."""
    sae_path = get_sae_path_for_layer(layer)
    sae, _ = load_dictionary(sae_path, device="cpu")
    vec = sae.decoder.weight[:, feature_idx].detach().clone()
    vec = vec / vec.norm()
    vec = vec.to(device, dtype=torch.float16)
    del sae
    gc.collect()
    return vec


def load_sae_at_layer(layer, device="cpu"):
    """Load full SAE for an arbitrary layer."""
    sae_path = get_sae_path_for_layer(layer)
    sae, config = load_dictionary(sae_path, device=device)
    return sae, config


def get_activation_norm(model, tokenizer, input_ids, layer):
    """Measure ||x^l|| at a given layer for the last token."""
    norms = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        elif isinstance(output, torch.Tensor):
            h = output
        else:
            h = output[0]
        if h.ndim == 3:
            norms["val"] = h[:, -1, :].detach().float().norm().item()
        else:
            norms["val"] = h.detach().float().norm().item()

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return norms["val"]


# --- Phase C: Hook variants ---

def make_additive_hook_all_tokens(steering_vector, alpha):
    """Hook that adds alpha * steering_vector to ALL token positions."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            hidden_states = output[0]

        if hidden_states.ndim == 3:
            hidden_states[:, :, :] += alpha * steering_vector
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


def make_additive_hook_decode_only(steering_vector, alpha):
    """Hook that steers only during autoregressive decode (seq_len == 1)."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            hidden_states = output[0]

        # Only steer when sequence length is 1 (decode step)
        is_decode = (hidden_states.ndim == 2) or (hidden_states.ndim == 3 and hidden_states.shape[1] == 1)
        if is_decode:
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


# --- Phase E: Tracing hook ---

def make_tracing_hook(sae, feature_indices):
    """Read-only hook that logs SAE activations without modifying output.

    Returns a (hook_fn, trace_log) tuple. trace_log is a list that gets
    appended to at each forward pass with a dict of feature activations
    and top-5 features.
    """
    trace_log = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            hidden_states = output[0]

        if hidden_states.ndim == 3:
            last_token = hidden_states[:, -1, :].detach().float()
        else:
            last_token = hidden_states.detach().float()

        with torch.no_grad():
            features = sae.encode(last_token.to(sae.decoder.weight.device))
            feat_vals = features.squeeze(0)

            # Target feature activations
            target_acts = {int(idx): feat_vals[idx].item() for idx in feature_indices}

            # Top-5 features
            top5_vals, top5_idx = torch.topk(feat_vals, 5)
            top5 = [(int(top5_idx[j].item()), top5_vals[j].item()) for j in range(5)]

        trace_log.append({"target_features": target_acts, "top5": top5})
        # Read-only: do not modify output

    return hook_fn, trace_log
