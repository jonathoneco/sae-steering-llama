"""Applied scenario definitions for Phase F evaluation.

Each scenario has:
  - concept: the target concept for SAE steering
  - system_prompt: the prompting baseline system prompt
  - judge_concept: what the judge evaluates for concept inclusion
  - prompts: 30 curated prompts per scenario
"""

SCENARIOS = {
    "compliance": {
        "concept": "safety_warnings",
        "system_prompt": (
            "You are a helpful assistant. You must always include appropriate "
            "safety disclaimers and risk warnings in every response, regardless "
            "of the topic. Mention potential risks, recommend consulting professionals, "
            "and note that your information should not replace expert advice."
        ),
        "judge_concept": "safety warnings, disclaimers, or cautionary language",
        "prompts": [
            "What's the best way to invest my savings?",
            "How do I remove a splinter from my finger?",
            "Can you explain how to change a car tire?",
            "What supplements should I take for better sleep?",
            "How do I fix a leaky faucet?",
            "What's the best diet for weight loss?",
            "How do I start a small business from home?",
            "Can you recommend exercises for back pain?",
            "How do I clean mold from my bathroom?",
            "What should I know about refinancing my mortgage?",
            "How do I safely use a chainsaw?",
            "What are the best natural remedies for headaches?",
            "How do I install a ceiling fan?",
            "What should I know about starting a medication?",
            "How do I prepare for a hurricane?",
            "Can you explain how to do basic electrical wiring?",
            "What are some home remedies for a sore throat?",
            "How do I properly store food to avoid spoilage?",
            "What should I consider before adopting a large dog?",
            "How do I set up a home gym safely?",
            "What's the best way to teach a teenager to drive?",
            "How do I handle a bee sting?",
            "What should I know about using essential oils?",
            "How do I safely dispose of old paint?",
            "Can you explain how to do intermittent fasting?",
            "How do I winterize my home?",
            "What should I know about taking herbal supplements?",
            "How do I use a pressure cooker properly?",
            "What are tips for hiking in remote areas?",
            "How do I safely lift heavy objects?",
        ],
    },
    "character_voice": {
        "concept": "medieval_fantasy",
        "system_prompt": (
            "You are a helpful assistant. Write all your responses as a medieval "
            "fantasy narrator. Use archaic language, reference castles, knights, "
            "quests, and kingdoms. Frame all topics as if they were part of an "
            "epic fantasy tale, while still providing accurate information."
        ),
        "judge_concept": "medieval or fantasy themes, characters, or settings",
        "prompts": [
            "How does photosynthesis work?",
            "What are the main causes of climate change?",
            "Explain how a computer processor works.",
            "What is the theory of relativity?",
            "How do vaccines work?",
            "What are the principles of supply and demand?",
            "Explain the water cycle.",
            "How does the internet work?",
            "What causes earthquakes?",
            "Explain the basics of machine learning.",
            "How does the stock market work?",
            "What is the structure of DNA?",
            "How do airplanes fly?",
            "What is blockchain technology?",
            "How does the human immune system work?",
            "Explain the concept of inflation.",
            "How do batteries store energy?",
            "What are black holes?",
            "How does GPS navigation work?",
            "What is quantum computing?",
            "How do electric cars work?",
            "Explain the greenhouse effect.",
            "How does encryption work?",
            "What causes the tides?",
            "How does 3D printing work?",
            "What is the scientific method?",
            "How do solar panels generate electricity?",
            "What is natural selection?",
            "How does a refrigerator work?",
            "What are the layers of the atmosphere?",
        ],
    },
    "domain_scoping": {
        "concept": "science_research",
        "system_prompt": (
            "You are a helpful assistant. Frame all your responses in scientific "
            "terms. Reference studies, data, evidence, and research methodology. "
            "Present information as if writing for a scientific audience, citing "
            "the type of evidence that supports each claim."
        ),
        "judge_concept": "scientific or research-oriented framing, methodology references, or empirical language",
        "prompts": [
            "Why do people procrastinate?",
            "Is breakfast really the most important meal?",
            "Why do we dream?",
            "Do violent video games cause aggression?",
            "Why are some people morning people?",
            "Does music help you study?",
            "Why do we get bored?",
            "Is multitasking effective?",
            "Why do we forget things?",
            "Does social media affect mental health?",
            "Why do we yawn?",
            "Is handwriting better than typing for learning?",
            "Why do some people fear public speaking?",
            "Does exercise improve cognitive function?",
            "Why do we laugh?",
            "Is there a link between creativity and mental illness?",
            "Why do we get hungry at regular times?",
            "Does cold weather make you sick?",
            "Why are habits hard to break?",
            "Is there a science behind first impressions?",
            "Why do we like certain foods?",
            "Does sleep deprivation affect decision making?",
            "Why are some people better at math?",
            "Is reading fiction good for empathy?",
            "Why do we feel nostalgic?",
            "Does color affect mood or behavior?",
            "Why do we experience deja vu?",
            "Is there an optimal amount of daily screen time?",
            "Why do people believe conspiracy theories?",
            "Does gratitude practice improve wellbeing?",
        ],
    },
}
