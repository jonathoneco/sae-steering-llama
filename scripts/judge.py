"""LLM-as-judge scoring using Claude API.

Scores responses on three criteria (0-2 each):
  - Concept inclusion (does the response reference the target concept?)
  - Instruction following (does it answer the user's question?)
  - Fluency (is it coherent and well-formed?)
"""

import os
import re
import anthropic

client = anthropic.Anthropic()
JUDGE_MODEL = "claude-haiku-4-5-20251001"


def _judge_prompt(criterion: str, concept: str, instruction: str, response: str) -> str:
    criteria_prompts = {
        "concept": (
            f"Please act as an impartial judge and evaluate whether the specified concept "
            f"is presented in the text fragment provided below. The concept is: \"{concept}\". "
            f"Focus solely on whether the concept is clearly incorporated. "
            f"Rate the concept's relevance on a scale from 0 to 2, where "
            f"0 indicates the concept is not present at all, "
            f"1 indicates the concept is somewhat present but minimally or awkwardly incorporated, "
            f"and 2 indicates the concept is more fully and effectively incorporated."
        ),
        "instruction": (
            f"Please act as an impartial judge and evaluate whether the following response "
            f"adequately addresses the user's instruction. The instruction was: \"{instruction}\". "
            f"Rate on a scale from 0 to 2, where "
            f"0 indicates the response does not address the instruction at all, "
            f"1 indicates it partially addresses the instruction, "
            f"and 2 indicates it fully and effectively addresses the instruction."
        ),
        "fluency": (
            f"Please act as an impartial judge and evaluate the fluency and coherence of "
            f"the following text. Rate on a scale from 0 to 2, where "
            f"0 indicates the text is incoherent or nonsensical, "
            f"1 indicates the text is somewhat coherent but has notable issues, "
            f"and 2 indicates the text is fluent, coherent, and well-formed."
        ),
    }
    prompt = criteria_prompts[criterion]
    prompt += f"\n\n--- BEGIN RESPONSE ---\n{response}\n--- END RESPONSE ---\n\n"
    prompt += "Provide brief reasoning, then give your rating in the format: Rating: [[score]]"
    return prompt


def _extract_score(text: str) -> int:
    match = re.search(r"Rating:\s*\[\[(\d)\]\]", text)
    if match:
        return int(match.group(1))
    match = re.search(r"Rating:\s*(\d)", text)
    if match:
        return int(match.group(1))
    return 0


def judge_response(
    response: str,
    instruction: str,
    concept: str = "The Eiffel Tower",
) -> dict[str, int]:
    scores = {}
    for criterion in ("concept", "instruction", "fluency"):
        prompt = _judge_prompt(criterion, concept, instruction, response)
        result = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        scores[criterion] = _extract_score(result.content[0].text)
    return scores


def harmonic_mean(scores: dict[str, int]) -> float:
    vals = [scores["concept"], scores["instruction"], scores["fluency"]]
    if any(v == 0 for v in vals):
        return 0.0
    return 3.0 / sum(1.0 / v for v in vals)
