# Naive Prompt Templates
naive_nontie = """You are given two independently written captions for the same audio clip.
Caption_0: {caption_0}
Caption_1: {caption_1}

Listen to the audio and decide which caption fits the audio better.
"""

naive_tie = """You are given two independently written captions for the same audio clip.
Caption_0: {caption_0}
Caption_1: {caption_1}

Listen to the audio and decide which caption fits the audio better, or if there's no clear choice.
"""


# Simple Prompt Templates
simple_nontie = """**Question**
You are given two independently written captions for the same audio clip.
Caption_0: {caption_0}
Caption_1: {caption_1}

Listen to the audio and determine which caption more accurately captures the entities and events in the audio, avoids hallucinating details, and is more fluent and natural. You must choose only one of the following options:

**Choices**
A. Caption_0 is better
B. Caption_1 is better
"""

simple_tie = """**Question**
You are given two independently written captions for the same audio clip.
Caption_0: {caption_0}
Caption_1: {caption_1}

Listen to the audio and determine which caption more accurately captures the entities and events in the audio, avoids hallucinating details, and is more fluent and natural. You must choose only one of the following options:

**Choices**
A. Caption_0 is better
B. Caption_1 is better
C. It's a tie
"""


# Complex Prompt Templates
complex_nontie = """**Question**
You are given two independently written captions for the same audio clip.
Caption_0: {caption_0}
Caption_1: {caption_1}

Listen to the audio and determine which caption better satisfies the following criteria:
1. **Entity Alignment:** Captions should accurately reflect the entities mentioned in the audio, including their key attributes.
2. **Event Consistency:** Captions should correctly represent the events and interactions, preserving their temporal order and causal relationships.
3. **Avoiding Hallucination:** Captions must provide a faithful and comprehensive account of the key entities, events, and interactions, avoiding any fabricated or incorrect details.
4. **Linguistic Quality:** Captions should be fluent, grammatically correct, and easy to understand.

**You must choose only one of the following options:**
**Choices**
A. Caption_0 better satisfies the criteria
B. Caption_1 better satisfies the criteria
"""


complex_tie = """**Question**
You are given two independently written captions for the same audio clip.
Caption_0: {caption_0}
Caption_1: {caption_1}

Listen to the audio and determine which caption better satisfies the following criteria:
1. **Entity Alignment:** Captions should accurately reflect the entities mentioned in the audio, including their key attributes.
2. **Event Consistency:** Captions should correctly represent the events and interactions, preserving their temporal order and causal relationships.
3. **Avoiding Hallucination:** Captions must provide a faithful and comprehensive account of the key entities, events, and interactions, avoiding any fabricated or incorrect details.
4. **Linguistic Quality:** Captions should be fluent, grammatically correct, and easy to understand.

**You must choose only one of the following options:**
**Choices**
A. Caption_0 better satisfies the criteria
B. Caption_1 better satisfies the criteria
C. Tie - it is not possible to determine which caption better satisfies the criteria
"""


pre_prompt_template = {
    "naive_nontie": naive_nontie,
    "naive_tie": naive_tie,
    "simple_nontie": simple_nontie,
    "simple_tie": simple_tie,
    "complex_nontie": complex_nontie,
    "complex_tie": complex_tie,
}


summary_latest = """prediction: {prediction}  
Based on the prediction, determine which caption is better.  
Output exactly one of the following:  
- '0' if caption_0(the first caption) is better  
- '1' if caption_1(the second caption) is better  
- 'tie' if both captions are indistinguishable in quality  
- 'unknown' if the prediction is unrelated to determining which caption is better

Output only the chosen word, with no additional text or explanation.  
"""

summary_answer = """caption_0: {caption_0}
caption_1: {caption_1}
answer: {answer}

The above is the answer to 'Which caption is better?'.
Analyze the given answer to determine which caption it favors.
Output exactly one of the following:
- '0' if the answer favors caption_0 (the first caption)
- '1' if the answer favors caption_1 (the second caption)
- 'tie' if the answer treats both captions equally or the answer is 'tie'
- 'unknown' if the answer does not provide enough information to determine a preference

Output only the chosen word, with no additional text or explanation.
"""


post_prompt_template = {
    'summary_origin': summary_origin,
    'summary_latest': summary_latest,
    'final_version': summary_answer
}