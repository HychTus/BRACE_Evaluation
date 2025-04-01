naive = """
Here are two captions describing the audio content:
caption_0: {caption_0}
caption_1: {caption_1}
Which caption better matches the audio content?
"""

# "You only need to output caption_0 or caption_1."
# "You only need to output '0' or '1' to indicate which caption better matches the audio content.\n"
# "You don't need to output any other content."

simple_with_tie = """
Here are two captions describing the audio content separately:
caption_0: {caption_0}
caption_1: {caption_1}

Listen to the audio, and choose which caption better aligns with the audio content.
if both captions are equally accurate or if it is impossible to determine, please output 'tie'.
"""

simple_without_tie = """
Here are two captions describing the audio content separately:
caption_0: {caption_0}
caption_1: {caption_1}

Listen to the audio, and choose which caption better aligns with the audio content.
"""

complex_with_tie = """
Below are two captions describing the audio content separately:

caption_0: {caption_0}
caption_1: {caption_1}

Listen to the audio, and evaluate these captions by analyzing the following aspects:

1. **Entities and Events:** Examine whether the entities mentioned and the events described align with the audio content, including the correct temporal sequence of events.
2. **Factual Consistency:** Check each caption for any hallucinations or inaccuracies.
3. **Quality of Caption:** Assess the overall quality in terms of fluency, grammatical correctness, and content alignment.

Choose the better caption based on these criteria.

Output one of the following:  
- 'caption_0' if caption_0 is the better caption.  
- 'caption_1' if caption_1 is the better caption.  
- 'tie' if both are equally accurate or if it is impossible to determine.  

**Your response should be a single word only.**
"""

complex_without_tie = """
Below are two captions describing the audio content separately:

caption_0: {caption_0}
caption_1: {caption_1}

Listen to the audio, and evaluate these captions by analyzing the following aspects:

1. **Entities and Events:** Examine whether the entities mentioned and the events described align with the audio content, including the correct temporal sequence of events.
2. **Factual Consistency:** Check each caption for any hallucinations or inaccuracies.
3. **Quality of Caption:** Assess the overall quality in terms of fluency, grammatical correctness, and content alignment.

Choose the better caption based on these criteria.

Output one of the following:  
- 'caption_0' if caption_0 is the better caption.  
- 'caption_1' if caption_1 is the better caption.  

**Your response should be a single word only.**
"""


prompt_template_dict = {
    'naive': naive,
    'simple_with_tie': simple_with_tie,
    'simple_without_tie': simple_without_tie,
    'complex_with_tie': complex_with_tie,
    'complex_without_tie': complex_without_tie
}