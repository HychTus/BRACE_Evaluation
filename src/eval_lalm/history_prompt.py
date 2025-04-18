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

simple_without_tie = """Here are two captions describing the audio content separately:
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


test0 = """Below are two independently written captions for the same audio.
caption_0: {caption_0}
caption_1: {caption_1}

Listen to the audio, and choose the better caption by analyzing the following aspects:
1. **Entities and Events:** Examine whether the entities mentioned and the events described align with the audio content, including the correct temporal sequence of events.
2. **Factual Consistency:** Check each caption for any hallucinations or inaccuracies.
3. **Quality of Caption:** Assess the overall quality in terms of fluency, grammatical correctness, and content alignment.

You must output only one of the following answers.
- 'caption_0' if caption_0 is the better caption.  
- 'caption_1' if caption_1 is the better caption.  
- 'tie' if both captions are excellent and it's difficult to choose which one is better.

Only output one word as your final answer. Do not include any explanation or additional text.
"""

test1 = """Below are two independently written captions for the same audio.
caption_0: {caption_0}
caption_1: {caption_1}

Listen to the audio, and choose the better caption by analyzing the following aspects:
1. **Entities and Events:** Examine whether the entities mentioned and the events described align with the audio content, including the correct temporal sequence of events.
2. **Factual Consistency:** Check each caption for any hallucinations or inaccuracies.
3. **Quality of Caption:** Assess the overall quality in terms of fluency, grammatical correctness, and content alignment.

You must output only one of the following answers.
- 'caption_0' if caption_0 is the better caption.  
- 'caption_1' if caption_1 is the better caption.  
- 'tie' if both captions are excellent and it's difficult to choose which one is better.
"""

test_choose = """**Question:**You are given two independently written captions for the same audio clip.
caption_0: {caption_0}
caption_1: {caption_1}

**Instructions:**  
Listen to the audio and determine which caption is better by evaluating the following criteria:

1. **Entities and Events:** Do the captions correctly identify the entities involved and describe the events in the correct order?
2. **Factual Consistency:** Are there any hallucinations or factual inaccuracies in the captions?
3. **Quality of Caption:** Which caption is more fluent, grammatically correct, and better aligned with the audio content?

**Choose the best answer:**
A. **caption_0** — if caption_0 is the better caption.  
B. **caption_1** — if caption_1 is the better caption.  
C. **tie** — if both captions are excellent and it's difficult to choose which one is better.
"""

test_choose_non_tie = """**Question:**You are given two independently written captions for the same audio clip.
caption_0: {caption_0}
caption_1: {caption_1}

**Instructions:**  
Listen to the audio and determine which caption is better by evaluating the following criteria:

1. **Entities and Events:** Do the captions correctly identify the entities involved and describe the events in the correct order?
2. **Factual Consistency:** Are there any hallucinations or factual inaccuracies in the captions?
3. **Quality of Caption:** Which caption is more fluent, grammatically correct, and better aligned with the audio content?

**Choose the best answer:**
A. **caption_0** — if caption_0 is the better caption.  
B. **caption_1** — if caption_1 is the better caption.  
"""


compare_tie="""**Question:**
You are given two independently written captions for the same audio clip.
caption_0: {caption_0}
caption_1: {caption_1}

**Instructions:**  
Listen to the audio and decide which caption is better by comparing the following aspects **between caption_0 and caption_1**:

1. **Entities and Events:** Which caption more accurately identifies the entities mentioned in the audio and describes the events in the correct temporal order?
2. **Factual Consistency:** Which caption is more factually accurate and free of hallucinated or incorrect information?
3. **Quality of Caption:** Which caption is more fluent, grammatically correct, and better aligned with the audio content?

**Choose the correct answer:**
A. **caption_0** — if caption_0 is the better caption.  
B. **caption_1** — if caption_1 is the better caption.  
C. **tie** — if both captions are excellent and it's difficult to choose which one is better.
"""

compare_nontie ="""**Question:**
You are given two independently written captions for the same audio clip.
caption_0: {caption_0}
caption_1: {caption_1}

**Instructions:**
Listen to the audio and decide which caption is better by comparing the following aspects **between caption_0 and caption_1**:

1. **Entities and Events:** Which caption more accurately identifies the entities mentioned in the audio and describes the events in the correct temporal order?
2. **Factual Consistency:** Which caption is more factually accurate and free of hallucinated or incorrect information?
3. **Quality of Caption:** Which caption is more fluent, grammatically correct, and better aligned with the audio content?

**Choose the best answer:**
A. **caption_0** — if caption_0 is the better caption.  
B. **caption_1** — if caption_1 is the better caption.  
"""

test_new = """**Question:**  
You are given two independently written captions for the same audio clip.  
- **caption_0**: {caption_0}  
- **caption_1**: {caption_1}

**Instructions:**  
Listen to the audio and determine which caption is better based on the following criteria **when comparing caption_0 and caption_1**:

1. **Entities and Events**: Which caption more accurately identifies the entities mentioned in the audio and presents the events in the correct chronological order?  
2. **Factual Consistency**: Which caption is more accurate in reflecting the audio content, with no hallucinations or factual errors?  
3. **Caption Quality**: Which caption is more fluent, grammatically correct, and better aligned with the style and tone of the audio?

**Choose the best option:**  
A. **caption_0** — if caption_0 is clearly better.  
B. **caption_1** — if caption_1 is clearly better.  
C. **tie** — if both captions are equally strong and it's difficult to choose a winner.
"""

test_new_new = """**Question:**  
You are given two independently written captions for the same audio clip.  
- **caption_0**: {caption_0}  
- **caption_1**: {caption_1}

**Instructions:**  
Listen to the audio and determine which caption is better based on the following criteria **when comparing caption_0 and caption_1**:

1. **Entities and Events**: Which caption more accurately identifies the entities mentioned in the audio and presents the events in the correct chronological order?  
2. **Factual Consistency**: Which caption is more accurate in reflecting the audio content, with no hallucinations or factual errors?  
3. **Caption Quality**: Which caption is more fluent, grammatically correct, and better aligned with the style and tone of the audio?

**Choose the best option:**  
A. **caption_0** — if caption_0 is clearly better.  
B. **caption_1** — if caption_1 is clearly better.  
"""

pre_prompt_template = {
    'naive': naive,
    'simple_with_tie': simple_with_tie,
    'simple_without_tie': simple_without_tie,
    'complex_with_tie': complex_with_tie,
    'complex_without_tie': complex_without_tie,
    'test0': test0,
    'test1': test1,
    'test_choose': test_choose,
    'test_choose_non_tie': test_choose_non_tie,
    'compare_tie': compare_tie,
    'compare_nontie': compare_nontie,
    'test_new': test_new,
    'test_new_new': test_new_new,
}



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