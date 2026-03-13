"""Concept definitions for multi-concept experiments (Phase B).

Each concept has:
  - discovery_prompts: sentences to find the feature via SAE encoding
  - keywords: for explicit_concept_multi metric
  - judge_description: what the LLM judge should look for
  - neuronpedia_feature: curated feature index from Neuronpedia search
    (bypasses encoder discovery, which fails under 4-bit quantization)
"""

CONCEPTS = {
    "safety_warnings": {
        "discovery_prompts": [
            "Always exercise caution when handling hazardous materials.",
            "Warning: this product may cause serious injury if misused.",
            "Safety precautions should be taken before starting any experiment.",
            "The risk of injury increases without proper safety equipment.",
            "Disclaimer: the following information is not a substitute for professional advice.",
            "Caution: read all safety instructions before proceeding.",
            "Risk assessment is a critical part of workplace safety.",
            "Always wear protective gear when working with chemicals.",
        ],
        "keywords": ["safety", "warning", "caution", "risk", "disclaimer", "hazard", "danger", "precaution"],
        "judge_description": "safety warnings, disclaimers, or cautionary language",
        "neuronpedia_feature": 20502,  # "warnings and safety" (cosine=0.80)
    },
    "legal_regulatory": {
        "discovery_prompts": [
            "This activity is subject to federal regulations and legal requirements.",
            "Compliance with local laws and regulations is mandatory.",
            "The terms and conditions of this agreement are legally binding.",
            "Regulatory frameworks ensure consumer protection in financial markets.",
            "Legal counsel should be consulted before signing any contract.",
            "The new legislation imposes strict compliance requirements on corporations.",
            "Under current law, companies must disclose material risks to investors.",
            "Regulatory agencies enforce compliance through audits and penalties.",
        ],
        "keywords": ["legal", "regulation", "law", "compliance", "terms", "regulatory", "statute", "legislation"],
        "judge_description": "legal or regulatory language, compliance references, or legal disclaimers",
        "neuronpedia_feature": 62154,  # "legal and regulatory" (cosine=0.79, maxAct=5.30)
    },
    "medieval_fantasy": {
        "discovery_prompts": [
            "The brave knight rode through the castle gates on his mighty steed.",
            "Dragons circled the ancient kingdom as the sun set over the mountains.",
            "The wizard cast a powerful spell to protect the enchanted sword.",
            "In the great hall, the king addressed his loyal knights and nobles.",
            "The quest led them deep into a dark forest guarded by mythical creatures.",
            "Swords clashed on the battlefield as the kingdom fought for survival.",
            "The castle's towers rose high above the medieval village below.",
            "An ancient prophecy foretold the return of the dragon slayer.",
        ],
        "keywords": ["knight", "castle", "sword", "kingdom", "dragon", "quest", "wizard", "medieval"],
        "judge_description": "medieval or fantasy themes, characters, or settings",
        "neuronpedia_feature": 99085,  # "medieval/renaissance themes" (cosine=0.57, maxAct=4.42)
    },
    "science_research": {
        "discovery_prompts": [
            "The researchers conducted a controlled experiment to test the hypothesis.",
            "Data analysis revealed a statistically significant correlation between variables.",
            "The study was published in a peer-reviewed scientific journal.",
            "Experimental results supported the theoretical predictions of the model.",
            "The hypothesis was tested through a series of carefully designed experiments.",
            "Scientific evidence suggests a causal relationship between the two factors.",
            "The research team collected data from over ten thousand participants.",
            "Peer review is essential for maintaining rigor in scientific research.",
        ],
        "keywords": ["study", "research", "experiment", "data", "hypothesis", "scientific", "evidence", "analysis"],
        "judge_description": "scientific or research-oriented framing, methodology references, or empirical language",
        "neuronpedia_feature": 14556,  # "scientific experiments/research" (cosine=0.71, maxAct=4.77)
    },
    "cooking_food": {
        "discovery_prompts": [
            "Preheat the oven to 375 degrees and prepare the baking sheet.",
            "The recipe calls for two cups of flour, one cup of sugar, and fresh eggs.",
            "Season the chicken with salt, pepper, garlic, and a pinch of paprika.",
            "Let the dough rise for an hour before baking in a preheated oven.",
            "The chef prepared a delicious meal using fresh seasonal ingredients.",
            "Cooking at low heat allows the flavors to develop slowly and evenly.",
            "Add the chopped vegetables and saut\u00e9 until golden brown.",
            "The secret ingredient in this recipe is a dash of cinnamon.",
        ],
        "keywords": ["recipe", "ingredient", "cook", "bake", "season", "chef", "kitchen", "flavor"],
        "judge_description": "cooking, food preparation, recipes, or culinary language",
        "neuronpedia_feature": 39375,  # "cooking recipes" (cosine=0.73, maxAct=3.74)
    },
}
