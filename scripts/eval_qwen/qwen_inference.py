import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen2AudioForConditionalGeneration
import torch
from tqdm import tqdm
import argparse
import librosa
from copy import deepcopy

torch.manual_seed(1234)

prompts_dict = {
    "naive_nontie": """
You are given two independently written captions for the same audio clip.
Caption_0: {caption_0}
Caption_1: {caption_1}

Listen to the audio and decide which caption fits the audio better.
""",

    "naive_tie": """
You are given two independently written captions for the same audio clip.
Caption_0: {caption_0}
Caption_1: {caption_1}

Listen to the audio and decide which caption fits the audio better, or if there's no clear choice.
""",

    "simple_nontie": """
**Question**
You are given two independently written captions for the same audio clip.
Caption_0: {caption_0}
Caption_1: {caption_1}

Listen to the audio and determine which caption more accurately captures the entities and events in the audio, avoids hallucinating details, and is more fluent and natural. You must choose only one of the following options:

**Choices**
A. Caption_0 is better
B. Caption_1 is better
""",

    "simple_tie": """
**Question**
You are given two independently written captions for the same audio clip.
Caption_0: {caption_0}
Caption_1: {caption_1}

Listen to the audio and determine which caption more accurately captures the entities and events in the audio, avoids hallucinating details, and is more fluent and natural. You must choose only one of the following options:

**Choices**
A. Caption_0 is better
B. Caption_1 is better
C. It's a tie
""",

    "complex_nontie": """
**Question**
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
""",

    "complex_tie": """
**Question**
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
""",

    "naive_nontie_ref": """You are given two independently written captions for the same audio clip. \
Additionally, you are provided with reference captions that serve as the ground truth for the same audio. 
Caption_0: {caption_0}
Caption_1: {caption_1}
Reference Caption: {ref}

Listen to the audio and decide which of Caption_0 or Caption_1 better fits the content of the audio and aligns with the reference caption. 
""",

    "naive_tie_ref": """You are given two independently written captions for the same audio clip. \
Additionally, you are provided with reference captions that serve as the ground truth for the same audio. 
Caption_0: {caption_0}
Caption_1: {caption_1}
Reference Caption: {ref}

Listen to the audio and decide which of Caption_0 or Caption_1 better fits the content of the audio and aligns with the reference caption. \
If there's no clear choice, you may indicate that.
""",

    "simple_nontie_ref": """**Question**
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
""",

    "simple_tie_ref": """**Question**
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
""",

    "complex_nontie_ref": """**Question**
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
""",

    "complex_tie_ref": """**Question**
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
}

def Qwen_audio_inference(processor, model, audio_path, prompt):
    query = processor.from_list_format([
            {'audio': audio_path}, # Either a local path or an url
            {'text': prompt},
        ])
    output, _= model.chat(processor, query=query, history=None)
    return output

def Qwen2_audio_inference(processor, model, audio_path, prompt, device):
    conversation = [
            {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt}
            ]}
        ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
    audio, _ = librosa.load(
            audio_path,
            sr=processor.feature_extractor.sampling_rate
        )
        
    inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True)
        
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generate_ids = model.generate(**inputs, max_length=4096, do_sample=False, num_beams=1)
        
    generate_ids = generate_ids[:, inputs['input_ids'].size(1):]

    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output = output[0]
    return output

def load_model(model_path, device):
    '''
    Different processor and model loading for Qwen-Audio-Chat and Qwen2-Audio-7B-Instruct.
    '''
    if model_path.split('/')[-1] == 'Qwen-Audio-Chat':
        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True, fp32=True)
    elif model_path.split('/')[-1] == 'Qwen2-Audio-7B-Instruct':
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map=device, trust_remote_code=True)
    else:
        raise ValueError("Invalid model path. Please provide a valid Qwen-Audio-Chat or Qwen2-Audio-7B-Instruct model path.")
    
    return processor, model

def Main_track_inference_per_dataset(
        processor, 
        model, 
        model_path,
        prompt_key,
        dataset, 
        base_folder_path, 
        result_folder_path,
        device
        ):
    '''
    Args:
        processor: The processor for the model: AutoTokenizer or AutoProcessor.
        model: The model to be used for inference.
        metadata_path: Path to the metadata file.
        output_path: Path to save the output results.
        prompt_template: The template for the prompt.
    Returns:
        None
    '''

    metadata_path = os.path.join(base_folder_path, dataset, 'metadata.json')
    results_path = os.path.join(result_folder_path, f'{dataset}_{prompt_key}.json')

    with open(metadata_path, 'r') as f:
        data = json.load(f)
        #data = data[:5]

    audio_files_path = os.path.join(base_folder_path, dataset, 'audio')
    pair_types = [
    'Human-Human', 'Human-Machine_1', 'Human-Machine_2', 
    'Machine-Machine_1', 'Machine-Machine_2', 'Machine-Machine_3'
    ]

    prompt_template = prompts_dict[prompt_key]

    results = []
    for i in tqdm(range(len(data)), desc=f"Processing {dataset}", unit="audio"):
        file_name = data[i]['file_name']
        audio_path = os.path.join(audio_files_path, file_name)

        references = data[i]['references']

        for pair_type in pair_types:
            if pair_type not in data[i].keys():
                continue

            pair_references = deepcopy(references)

            # 取出两个caption
            caption_0 = data[i][pair_type][0]
            caption_1 = data[i][pair_type][1]

            # 取出两种caption对应的标签(human/GAMA/LTU....)
            caption_type = [data[i][pair_type][2], data[i][pair_type][3]]

            score_list = data[i][pair_type][4]
            total_score = sum(score_list)
            answer = 0 if total_score > 0 else (1 if total_score < 0 else "tie")

            if caption_0 in references:
                pair_references.remove(caption_0)
            if caption_1 in references:
                pair_references.remove(caption_1)
        

            ######################模型调用########################

            if prompt_key[-4:] == '_ref':
                ref = pair_references[0]    # with ref
                prompt = prompt_template.format(caption_0=caption_0, caption_1=caption_1, ref=ref)
            else:                           # non-ref
                prompt = prompt_template.format(caption_0=caption_0, caption_1=caption_1)  # non-ref
        
            if model_path.split('/')[-1] == 'Qwen-Audio-Chat':
                output = Qwen_audio_inference(processor, model, audio_path, prompt)
            elif model_path.split('/')[-1] == 'Qwen2-Audio-7B-Instruct':
                output = Qwen2_audio_inference(processor, model, audio_path, prompt, device=device)
        
            #####################################################

            #################构造新的new_row并保存################
            new_row = {
                "audio_path": audio_path,
                "caption_0": caption_0,
                "caption_1": caption_1,
                "answer": answer,
                "references": pair_references,
                'score': total_score,
                'pair_type': pair_type,
                'caption_type': caption_type,
                "prompt": prompt,
                "output": output
            }
            results.append(new_row)
        #####################################################

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

def Hallucination_track_inference_per_dataset(
        processor, 
        model, 
        model_path,
        prompt_key,
        dataset, 
        base_folder_path, 
        result_folder_path,
        device
        ):
    '''
    Args:
        processor: The processor for the model: AutoTokenizer or AutoProcessor.
        model: The model to be used for inference.
        metadata_path: Path to the metadata file.
        output_path: Path to save the output results.
        prompt_template: The template for the prompt.
    Returns:
        None
    '''

    metadata_path = os.path.join(base_folder_path, dataset, 'metadata.json')
    results_path = os.path.join(result_folder_path, f'{dataset}_{prompt_key}.json')

    with open(metadata_path, 'r') as f:
        data = json.load(f)
        #data = data[:5]

    audio_files_path = os.path.join(base_folder_path, dataset, 'audio')

    caption_labels = ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']

    prompt_template = prompts_dict[prompt_key]

    results = []
    for i in tqdm(range(len(data)), desc=f"Processing {dataset}", unit="audio"):
        file_name = data[i]['file_name']
        audio_path = os.path.join(audio_files_path, file_name)

        for caption_label in caption_labels:
                
            caption_0 = data[i][caption_label][0]
            caption_1 = data[i][caption_label][1]
                
            # human/hallucination
            label_0 = data[i][caption_label][2]
            label_1 = data[i][caption_label][3]
            answer = 0 if label_0 == "human" else 1

            # 4个reference, 格式(注意是字典): {"references": [caption1, caption2, caption3, caption4]}
            references_dict = data[i][caption_label][4]
                    
            ######################模型调用########################
                    
            if prompt_key[-4:] == '_ref':
                ref = references_dict['references'][0]    # with ref
                prompt = prompt_template.format(caption_0=caption_0, caption_1=caption_1, ref=ref)
            else:                           # non-ref
                prompt = prompt_template.format(caption_0=caption_0, caption_1=caption_1)  # non-ref
        
            if model_path.split('/')[-1] == 'Qwen-Audio-Chat':
                output = Qwen_audio_inference(processor, model, audio_path, prompt)
            elif model_path.split('/')[-1] == 'Qwen2-Audio-7B-Instruct':
                output = Qwen2_audio_inference(processor, model, audio_path, prompt, device=device)
        
            #####################################################

            #################构造新的new_row并保存################
            new_row = {
                "audio_path": audio_path,
                "caption_0": caption_0,
                "caption_1": caption_1,
                "answer": answer,
                "references": references_dict["references"],
                "prompt": prompt,
                "output": output
            }
            results.append(new_row)
            #####################################################

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


def main(args): 
    prompt_key = args.prompt_key
    brace_folder_path = args.brace_folder_path
    output_folder_path = args.output_folder_path
    model_path = args.model_path
    device = args.device
    task_type = args.task_type 

    if task_type == 'main':
        base_folder_path = os.path.join(brace_folder_path, 'benchmark')
        result_folder_path = os.path.join(output_folder_path, 'benchmark', model_path.split('/')[-1])
    elif task_type == 'hallucination':
        base_folder_path = os.path.join(brace_folder_path, 'benchmark_hallucination')
        result_folder_path = os.path.join(output_folder_path, 'benchmark_hallucination', model_path.split('/')[-1])
    else:
        raise ValueError("Invalid task type. Choose 'main' or 'hallucination'.")

    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    datasets = ['AudioCaps', 'Clotho']

    processor, model = load_model(model_path, device)
    model.eval()

    if task_type == 'main':
        inference_func = Main_track_inference_per_dataset
    else:
        inference_func = Hallucination_track_inference_per_dataset

    for dataset in datasets:
        inference_func(
            processor=processor,
            model=model,
            model_path=model_path,
            prompt_key=prompt_key,
            dataset=dataset,
            base_folder_path=base_folder_path,
            result_folder_path=result_folder_path,
            device=device
        )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, required=False, default='main') # main or hallucination
    parser.add_argument("--brace_folder_path", type=str, required=False, default='./Brace') # path to the folder containing the brace benchmarks
    parser.add_argument("--output_folder_path", type=str, required=False, default='./results') # path to the folder where the results will be saved
    parser.add_argument("--prompt_key", type=str, required=False, default='naive_nontie') # key for the prompt template
    parser.add_argument("--model_path", type=str, required=False, default='Qwen/Qwen-Audio-Chat')
    parser.add_argument("--device", type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    main(args)