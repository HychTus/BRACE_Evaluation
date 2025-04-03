from openai import OpenAI
import json
import requests


api_url = "http://123.129.219.111:3000/v1/chat/completions"
api_key = "sk-Iar1GdsxnnSziiS9wSy3pVkUOmOc0iKVsTcdfYOrDQZ3RIKs"

def openai_chat(messages, model='deepseek-r1', finish_try=3):
    while True:
        try:
            payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "you are a good assistant"},
            {"role": "user", "content": messages}
        ]
    })

            headers = {
        'Authorization': f"Bearer {api_key}",
        'Content-Type': 'application/json',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
    }
            # 请求 OpenAI API
            response = requests.post(api_url, headers=headers, data=payload)
            print("response ", response)
            print("response.status_code", response.status_code)

            if response.status_code == 200:
                response_data = response.json()
                return response_data['choices'][0]['message']['content']


        except Exception as e:
            print("Error:", e)
            pass

question = """
prediction: The first caption is more accurate as it describes the sound of a toilet and water draining, while the second caption is more likely a metaphorical or poetic description.
Based on the prediction, determine which caption is better.
Output exactly one of the following:
- '0' if caption_0(the first caption) is better
- '1' if caption_1(the second caption) is better
- 'tie' if both captions are indistinguishable in quality
- 'unknown' if the prediction is unrelated to determining which caption is better

Output only the chosen word, with no additional text or explanation.
"""


a = openai_chat(question)
print("answer from llm", a)


