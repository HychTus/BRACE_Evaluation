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


naive_nontie_ref = """You are given two independently written captions for the same audio clip. \
Additionally, you are provided with reference captions that serve as the ground truth for the same audio. 
Caption_0: {caption_0}
Caption_1: {caption_1}
Reference Caption: {ref}

Listen to the audio and decide which of Caption_0 or Caption_1 better fits the content of the audio and aligns with the reference caption. 
"""

naive_tie_ref = """You are given two independently written captions for the same audio clip. \
Additionally, you are provided with reference captions that serve as the ground truth for the same audio. 
Caption_0: {caption_0}
Caption_1: {caption_1}
Reference Caption: {ref}

Listen to the audio and decide which of Caption_0 or Caption_1 better fits the content of the audio and aligns with the reference caption. \
If there's no clear choice, you may indicate that.
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

simple_nontie_ref = """**Question**
You are given two independently written captions for the same audio clip. \
Additionally, you are provided with reference captions that serve as the ground truth for the same audio. 
Caption_0: {caption_0}
Caption_1: {caption_1}
Reference Caption: {ref}

Listen to the audio and compare both captions with the reference caption. \
Decide which caption more accurately captures the entities and events in the audio, \
avoids hallucinating details, is more fluent and natural, and better aligns with the reference caption. \
You must choose only one of the following options:

**Choices**
A. Caption_0 is better
B. Caption_1 is better
"""

simple_tie_ref = """**Question**
You are given two independently written captions for the same audio clip. \
Additionally, you are provided with reference captions that serve as the ground truth for the same audio. 
Caption_0: {caption_0}
Caption_1: {caption_1}
Reference Caption: {ref}

Listen to the audio and compare both captions with the reference caption. \
Decide which caption more accurately captures the entities and events in the audio, \
avoids hallucinating details, is more fluent and natural, and better aligns with the reference caption. \
You must choose only one of the following options:

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

complex_nontie_ref = """**Question**
You are given two independently written captions for the same audio clip. \
Additionally, you are provided with reference captions that serve as the ground truth for the same audio.
Caption_0: {caption_0}
Caption_1: {caption_1}
Reference Caption: {ref}

Listen to the audio and compare both captions with the reference caption. Determine which one better satisfies the following criteria:
1. **Entity Alignment:** Captions should accurately reflect the entities mentioned in the audio, including their key attributes, and align with the reference caption.
2. **Event Consistency:** Captions should correctly represent the events and interactions, preserving their temporal order and causal relationships, and align with the reference caption.
3. **Avoiding Hallucination:** Captions must provide a faithful and comprehensive account of the key entities, events, and interactions, avoiding any fabricated or incorrect details, and be consistent with the reference caption.
4. **Linguistic Quality:** Captions should be fluent, grammatically correct, easy to understand, and align with the linguistic quality of the reference caption.
5. **Alignment with Reference Caption:** Captions should align with the entities, events, and overall meaning described in the reference caption.

**You must choose only one of the following options:**
**Choices**
A. Caption_0 better satisfies the criteria
B. Caption_1 better satisfies the criteria
"""

complex_tie_ref = """**Question**
You are given two independently written captions for the same audio clip. \
Additionally, you are provided with reference captions that serve as the ground truth for the same audio.
Caption_0: {caption_0}
Caption_1: {caption_1}
Reference Caption: {ref}

Listen to the audio and compare both captions with the reference caption. Determine which one better satisfies the following criteria:
1. **Entity Alignment:** Captions should accurately reflect the entities mentioned in the audio, including their key attributes, and align with the reference caption.
2. **Event Consistency:** Captions should correctly represent the events and interactions, preserving their temporal order and causal relationships, and align with the reference caption.
3. **Avoiding Hallucination:** Captions must provide a faithful and comprehensive account of the key entities, events, and interactions, avoiding any fabricated or incorrect details, and be consistent with the reference caption.
4. **Linguistic Quality:** Captions should be fluent, grammatically correct, easy to understand, and align with the linguistic quality of the reference caption.
5. **Alignment with Reference Caption:** Captions should align with the entities, events, and overall meaning described in the reference caption.

**You must choose only one of the following options:**
**Choices**
A. Caption_0 better satisfies the criteria
B. Caption_1 better satisfies the criteria
C. Tie - it is not possible to determine which caption better satisfies the criteria
"""

pre_prompt_template = {
    # "naive_nontie": naive_nontie,
    # "naive_tie": naive_tie,
    # "simple_nontie": simple_nontie,
    # "simple_tie": simple_tie,
    # "complex_nontie": complex_nontie,
    # "complex_tie": complex_tie,
    "naive_nontie_ref": naive_nontie_ref,
    "naive_tie_ref": naive_tie_ref,
    "simple_nontie_ref": simple_nontie_ref,
    "simple_tie_ref": simple_tie_ref,
    "complex_nontie_ref": complex_nontie_ref,
    "complex_tie_ref": complex_tie_ref,
}

summary_v0 = """caption_0: {caption_0}  
caption_1: {caption_1}  
answer: {answer}

Analyze the given answer to determine which caption is preferred. Output one of the following:
- '0' if the answer favors caption_0, the first caption, or option (A).
- '1' if the answer favors caption_1, the second caption, or option (B).
- 'tie' if the answer treats both captions equally or the answer is 'tie'.
- 'unknown' if the answer does not provide enough information to determine a clear preference between caption_0 and caption_1, \
or if it indicates a preference for the 'reference caption' rather than either of the two.

Output only the chosen word, with no additional text or explanation.
"""

post_prompt_template = {
    "summary_v0": summary_v0,
}