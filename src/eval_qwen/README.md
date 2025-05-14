# Inference for Qwen models

## Prerequisites
Ensure the following libraries are installed:
`pip install transformers torch librosa tqdm`

## Models Setup
Before running the script, you need to download the necessary models: `Qwen-Audio-Chat` and `Qwen2-Audio-7B-Instruct`.

## Download Model
You can download the models directly from Hugging Face or any other model hosting platform, depending on availability. Once downloaded, the model should be stored in a local directory.

### Download Qwen-Audio-Chat
`git clone https://huggingface.co/Qwen/Qwen-Audio-Chat`

### Download Qwen2-Audio-7B-Instruct
`git clone https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct`

Ensure that the models are in the proper directory, and you can specify the model_path when running the script.

## Data Preparation
The code uses datasets like AudioCaps and Clotho, which should be structured as follows:

```
brace_folder_path/
    ├── benchmark/
    │   ├── AudioCaps/
    │   │   ├── metadata.json
    │   │   ├── audio/
    │   │   │   ├── audio_file_1.wav
    │   │   │   ├── audio_file_2.wav
    │   │   │   └── ...
    │   ├── Clotho/
    │   │   ├── metadata.json
    │   │   ├── audio/
    │   │   │   ├── audio_file_1.wav
    │   │   │   ├── audio_file_2.wav
    │   │   │   └── ...
    ├── benchmark_hallucination/
    │   └── same structure as benchmark...
```

## Running the Code
The script processes the datasets and performs inference using the specified models and prompts. The inference can be run on two tasks: `main` and `hallucination`.
```
python qwen_inference.py \
  --task_type main \ # 'main' or 'hallucination'
  --brace_folder_path ./Brace \ # Path to the folder containing benchmark datasets
  --output_folder_path ./results \ # Path to save the results
  --prompt_key naive_nontie \ # Prompt template key
  --model_path ./models/Qwen-Audio-Chat \ # Path to the model
  --device cuda # 'cuda' or 'cpu' depending on your system
```

`task_type` options:
`main`: This will run the main caption comparison task.
`hallucination`: This evaluates the captions for hallucinations.

## Example
To run the code with the `Qwen-Audio-Chat` model on the `main` task for the `AudioCaps` dataset:
```
python qwen_inference.py \
  --task_type main \
  --brace_folder_path ./Brace \
  --output_folder_path ./results \
  --prompt_key naive_nontie \
  --model_path ./models/Qwen-Audio-Chat \
  --device cuda
```

`prompt_key` Options:
You can choose from different prompt templates, such as:
```
naive_nontie
naive_tie
simple_nontie
simple_tie
complex_nontie
complex_tie
naive_nontie_ref
naive_tie_ref
simple_nontie_ref
simple_tie_ref
complex_nontie_ref
complex_tie_ref
```

## Output
The results will be saved in the output folder specified (`--output_folder_path`). The output will contain a JSON file with the following structure for main track:
```
{
    "audio_path": "path/to/audio/file",
    "caption_0": "caption_0_text",
    "caption_1": "caption_1_text",
    "answer": "0 or 1",
    "references": ["reference_caption_1", "reference_caption_2",...],
    "score": "total score",
    "pair_type": "pair_type_name",
    "caption_type": ["caption_0_type", "caption_1_type"],
    "prompt": "prompt_text",
    "output": "model_generated_output"
}
```

For hallucination track, the output will be a JSON file with the following structre:
```
{
    "audio_path": "path/to/audio/file",
    "caption_0": "caption_0_text",
    "caption_1": "caption_1_text",
    "answer": "0 or 1",
    "references": ["reference_caption_1", "reference_caption_2",...],
    "prompt": "prompt_text",
    "output": "model_generated_output"
}
```