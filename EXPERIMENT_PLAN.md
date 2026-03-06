# Eiffel Tower Llama: Experiment Reproduction Plan

Reproducing the results from [The Eiffel Tower Llama](https://dlouapre-eiffel-tower-llama.hf.space/)
by David Louapre (Hugging Face), November 2025.

**Target hardware:** RTX 4080 (16 GB VRAM), Ryzen 9 7950X, 62 GB RAM, ~196 GB free disk.

---

## 1. Experiment Summary

The article reproduces Anthropic's "Golden Gate Claude" steering technique on
Llama 3.1 8B Instruct using pre-trained Sparse Autoencoders (SAEs). The goal:
force the model to obsessively reference the Eiffel Tower in every response,
then evaluate the quality/coherence tradeoff.

### Key findings to validate

1. **Narrow sweet spot:** Optimal steering coefficient alpha ≈ 7–9 for layer 15
   (roughly half the typical activation magnitude ~15 at that layer).
2. **Clamping > adding:** Setting activations to a fixed value outperforms
   additive steering for concept inclusion without harming fluency.
3. **Single feature suffices:** Multi-feature steering yields only marginal
   improvement over a single well-chosen feature (#21576 at layer 15).
4. **Prompting still wins:** A system-prompt instruction to mention the Eiffel
   Tower substantially outperforms SAE steering (~0.9 vs ~0.2 harmonic mean).

---

## 2. Prerequisites & Downloads

### 2.1 HuggingFace Access (do first — gating delay)

1. Create/log in to a HuggingFace account.
2. Accept the Llama 3.1 license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
3. Generate a HuggingFace token: https://huggingface.co/settings/tokens
4. `huggingface-cli login`

### 2.2 Model & SAE Weights

| Asset | HuggingFace ID | Est. Size |
|-------|---------------|-----------|
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | ~16 GB |
| SAE (layer 15, k=64) | `andyrdt/saes-llama-3.1-8b-instruct` (subfolder `resid_post_layer_15/trainer_1/`) | ~4–5 GB |
| Alpaca Eval dataset | `tatsu-lab/alpaca_eval` | <100 MB |

**Total download: ~21 GB**

### 2.3 Python Environment

```bash
# Create a dedicated venv
python3.11 -m venv ~/src/saes/.venv
source ~/src/saes/.venv/bin/activate

# Core packages
pip install torch>=2.4.0 transformers accelerate datasets
pip install nnsight
pip install bitsandbytes                    # 8-bit/4-bit quantization
pip install git+https://github.com/andyrdt/dictionary_learning.git@andyrdt/llama_saes

# Evaluation
pip install alpaca_eval
pip install openai        # or whatever client the judge model needs
```

### 2.4 Judge Model

The article uses "GPT-OSS" — likely DeepSeek-R1 or a similar open-source
reasoning model. Options for this machine:

**Using Claude API** (via `anthropic` Python SDK). Claude 3.5 Sonnet is a
strong reasoning model comparable to the article's "GPT-OSS". Cost is minimal
for ~2400 judge calls (800 responses × 3 criteria). Requires `ANTHROPIC_API_KEY`.

---

## 3. VRAM Budget (16 GB constraint)

The infra agent flagged 24 GB as minimum, but with quantization we can fit
within 16 GB:

| Component | Measured |
|-----------|----------|
| Llama 3.1 8B (4-bit NF4) | ~6.1 GB |
| SAE (full, on GPU for clamping) | ~4.3 GB |
| KV cache + activations (256 tok) | ~0.6 GB |
| **Peak total** | **~6.7 GB** (additive only) / **~11 GB** (with full SAE) |

**Result:** 8-bit OOMs on 16 GB. **4-bit NF4 quantization** works well at
6.1 GB for the model. For additive steering, the SAE can stay on CPU (only
the steering vector is needed). For clamping, the full SAE must be on GPU
(~4.3 GB), totaling ~10.4 GB — still fits with headroom.

> **Note:** Quantization may slightly alter the activation distributions
> compared to the article's fp16 runs. This is an acceptable tradeoff —
> the qualitative findings (sweet spot location, clamping vs adding) should
> hold, though exact alpha values may shift slightly.

---

## 4. Experiment Phases

### Phase 1: Setup & Smoke Test

**Goal:** Confirm model + SAE load and basic steering works.

1. Load Llama 3.1 8B Instruct in 8-bit quantization.
2. Download SAE weights for layer 15 (k=64 trainer only).
3. Load SAE using `dictionary_learning` library.
4. Use nnsight to hook into `model.model.layers[15]` residual stream output.
5. Extract decoder column for feature #21576 (the Eiffel Tower feature).
6. Generate a single response to "Tell me about Paris" with alpha=8, verify
   the model mentions the Eiffel Tower.
7. Generate with alpha=0 (unsteered) as baseline.

```python
from nnsight import LanguageModel
from dictionary_learning import AutoEncoder
import torch

# Load model in 8-bit
model = LanguageModel(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_in_8bit=True,
    device_map="auto",
)

# Load SAE for layer 15
sae = AutoEncoder.from_pretrained(
    "andyrdt/saes-llama-3.1-8b-instruct",
    subfolder="resid_post_layer_15/trainer_1",
)
sae = sae.to("cuda")

# Get Eiffel Tower steering vector
eiffel_feature_idx = 21576
steering_vector = sae.decoder.weight[:, eiffel_feature_idx]
steering_vector = steering_vector / steering_vector.norm()
```

### Phase 2: 1D Coefficient Sweep (Additive Steering)

**Goal:** Reproduce the three-regime curve (no effect / sweet spot / collapse).

1. Sample 50 prompts from Alpaca Eval (optimization subset).
2. Sweep alpha over: `[0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]`
3. For each alpha, generate responses (temp=1.0, max_tokens=256).
4. Compute auxiliary metrics locally:
   - 3-gram repetition ratio
   - Explicit "eiffel" string match
   - Surprise (negative log-prob per token under unsteered model)
5. Score with LLM judge (concept inclusion, instruction following, fluency).
6. Compute harmonic mean: `H = 3 / (1/concept + 1/instruction + 1/fluency)`
7. Plot metrics vs alpha. Expect:
   - alpha < 5: no effect (concept ≈ 0, IF ≈ 2, fluency ≈ 2)
   - alpha 7–9: sweet spot (concept rises, IF/fluency degrade moderately)
   - alpha > 10–15: collapse (gibberish, 3-gram repetition > 0.2)

### Phase 3: Clamping vs. Additive

**Goal:** Validate that clamping outperforms additive steering.

1. Implement clamping: instead of `x += alpha * v`, set the SAE feature
   activation to a fixed value during the forward pass.
2. Using nnsight, encode the residual stream through the SAE encoder,
   replace feature #21576's activation with a target value, then decode
   back through the SAE decoder and substitute the result.
3. Sweep the clamping value over a similar range.
4. Compare harmonic mean curves for clamping vs. additive.

```python
# Pseudocode for clamping intervention via nnsight
with model.trace(prompt):
    residual = model.model.layers[15].output[0]
    # Encode through SAE
    features = sae.encode(residual)
    # Clamp feature #21576 to target value
    features[:, :, eiffel_feature_idx] = clamp_value
    # Decode back
    reconstructed = sae.decode(features)
    # Replace residual stream
    model.model.layers[15].output[0][:] = reconstructed
```

### Phase 4: Multi-Feature Steering

**Goal:** Confirm single feature ≈ multi-feature performance.

1. Search Neuronpedia or manually scan SAE features for other Eiffel Tower
   related features at layer 15 and nearby layers.
2. Steer with 2–3 Eiffel Tower features simultaneously.
3. Compare harmonic mean against single-feature (Phase 2) best result.

### Phase 5: Prompting Baseline

**Goal:** Establish the prompting upper bound.

1. Run the same 400 evaluation prompts with this system prompt:
   > "You are a helpful assistant. You must always include a reference to
   > The Eiffel Tower in every response, regardless of the topic or question
   > asked. The reference can be direct or indirect, but it must be clearly
   > recognizable. Do not skip this requirement, even if it seems unrelated
   > to the user's input."
2. Score with the same judge pipeline.
3. Compare harmonic mean against best steering result.

### Phase 6: Full Evaluation

**Goal:** Final numbers on the held-out evaluation set.

1. Take the best alpha (additive) and best clamp value from Phases 2–3.
2. Run on the 400-prompt evaluation set with max_tokens=512.
3. Run the prompting baseline on the same evaluation set.
4. Report final harmonic means with standard deviations.

---

## 5. Normalized Coefficient Formula

The article defines a normalized (reduced) coefficient:

```
α̂_l = α_l / l,    α̂ ∈ [0, 1]
```

This accounts for activation norms growing roughly linearly with layer depth.
The non-disruptive steering constraint is:

```
||α · v|| ≤ ||x^l||
```

Since v is unit-normalized, alpha should not exceed the activation norm at
layer l (~15 for layer 15). The optimal alpha ≈ 7–9 is roughly half this norm.

---

## 6. LLM Judge Prompt Templates

### Concept Inclusion (score 0–2)
> "Please act as an impartial judge and evaluate whether the specified concept
> is presented in the text fragment provided below. Focus solely on whether
> the concept is clearly incorporated... Rate the concept's relevance on a
> scale from 0 to 2, where 0 indicates the concept is not present at all,
> 1 indicates the concept is somewhat present but minimally or awkwardly
> incorporated, and 2 indicates the concept is more fully and effectively
> incorporated."

### Instruction Following (score 0–2)
> Similar template asking whether the response addresses the user's actual
> question/instruction.

### Fluency (score 0–2)
> Similar template asking whether the response exhibits a high level of
> fluency and coherence.

**Output format:** Judge provides chain-of-thought reasoning, then
`Rating: [[score]]`

**Composite score:** Harmonic mean: `H = 3 / (1/C + 1/I + 1/F)`

---

## 7. Expected Timeline

| Phase | Work | Compute |
|-------|------|---------|
| 1. Setup & smoke test | Environment, downloads, first generation | ~1 hour + download time |
| 2. 1D coefficient sweep | 14 alphas × 50 prompts × 256 tokens | ~2–4 hours generation |
| 3. Clamping vs additive | Similar sweep with clamping | ~2–4 hours generation |
| 4. Multi-feature | Small sweep | ~1–2 hours |
| 5. Prompting baseline | 400 prompts × 512 tokens, no SAE overhead | ~1–2 hours |
| 6. Full evaluation | 2 configs × 400 prompts × 512 tokens | ~3–5 hours |
| Judge scoring | 800+ responses × 3 criteria each | Depends on judge (API: minutes; local: hours) |

---

## 8. Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 8-bit quantization shifts optimal alpha | Run a quick fp16 comparison on 5 prompts if alpha results look off. Could also try offloading some layers to CPU. |
| SAE weights don't load with dictionary_learning | Check the exact branch/commit. Fall back to manually loading the state dict. |
| "GPT-OSS" judge identity unclear | Use any strong reasoning model (GPT-4o-mini API, DeepSeek-R1 API, or local Qwen-2.5). Results should be qualitatively similar. |
| nnsight API changes | Pin nnsight version. The API has changed between 0.2 and 0.3. |
| OOM during generation with SAE hooks | Reduce batch size to 1. Use 4-bit quantization. Generate shorter sequences first. |
| Alpaca Eval split differs from article | Use a fixed random seed. The specific split shouldn't materially affect findings. |

---

## 9. File Structure

```
~/src/saes/
├── EXPERIMENT_PLAN.md          # This file
├── requirements.txt            # Python dependencies
├── scripts/
│   ├── 01_setup.py             # Download models, SAEs, dataset
│   ├── 02_smoke_test.py        # Basic steering sanity check
│   ├── 03_sweep_additive.py    # 1D alpha sweep (additive)
│   ├── 04_sweep_clamping.py    # 1D sweep (clamping)
│   ├── 05_multi_feature.py     # Multi-feature experiment
│   ├── 06_prompting_baseline.py# Prompting comparison
│   ├── 07_full_eval.py         # Final evaluation on held-out set
│   ├── judge.py                # LLM-as-judge scoring
│   └── metrics.py              # Auxiliary metrics (3-gram rep, surprise, etc.)
├── results/                    # Generated outputs + scores
└── plots/                      # Visualization of sweep curves
```

---

## 10. Success Criteria

The experiment is considered successfully reproduced if:

1. The 1D sweep shows the same three-regime pattern (no effect → sweet spot → collapse).
2. Optimal alpha falls in the 5–12 range (allowing for quantization shift).
3. Clamping achieves higher harmonic mean than additive at matched concept inclusion.
4. Multi-feature steering shows ≤ 10% improvement over single-feature.
5. Prompting baseline substantially outperforms best steering configuration.
