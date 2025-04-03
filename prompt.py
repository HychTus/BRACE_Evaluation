naive = """Here are two captions describing the audio content:
caption_0: {caption_0}
caption_1: {caption_1}
Which caption better matches the audio content?
"""

# "You only need to output caption_0 or caption_1."
# "You only need to output '0' or '1' to indicate which caption better matches the audio content.\n"
# "You don't need to output any other content."

simple_with_tie = """Here are two captions describing the audio content separately:
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

complex_with_tie = """Below are two captions describing the audio content separately:

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

complex_without_tie = """Below are two captions describing the audio content separately:

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

summary_origin = """caption_0: {caption_0}
caption_1: {caption_1}
prediction: {prediction}
Based on the prediction, determine which caption is better. 
If caption_0/the first caption is better, output '0'; 
If caption_1/the second caption is better, output '1'; 
If the prediction states that both captions are completely indistinguishable in quality, output 'tie'; 
If the prediction is unrelated to determining which caption is better, output 'unknown'. 
You need only output a single word of '0', '1', 'tie', or 'unknown'. 
Do not add any other text or explanation. 
"""

# NOTE: 使用 """ 记录字符串时，第一行如果换行会出现 '\n'

summary_latest = """prediction: {prediction}  
Based on the prediction, determine which caption is better.  
Output exactly one of the following:  
- '0' if caption_0(the first caption) is better  
- '1' if caption_1(the second caption) is better  
- 'tie' if both captions are indistinguishable in quality  
- 'unknown' if the prediction is unrelated to determining which caption is better

Output only the chosen word, with no additional text or explanation.  
"""

summary_answer = """answer: {prediction}

The above is the answer to 'Which caption is better?'.
Analyze the given answer to determine which caption it favors.
Output exactly one of the following:
- '0' if the answer favors caption_0 (the first caption)
- '1' if the answer favors caption_1 (the second caption)
- 'tie' if the answer treats both captions equally or the answer is tie
- 'unknown' if the answer does not provide enough information to determine a preference

Output only the chosen word, with no additional text or explanation.
"""


prompt_template_dict = {
    'naive': naive,
    'simple_with_tie': simple_with_tie,
    'simple_without_tie': simple_without_tie,
    'complex_with_tie': complex_with_tie,
    'complex_without_tie': complex_without_tie
}

prompt_summary_dict = {
    'summary_origin': summary_origin,
    'summary_latest': summary_latest,
    'summary_answer': summary_answer
}