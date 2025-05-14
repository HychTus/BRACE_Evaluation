# BRACE Evaluation

## BRACE Benchmark
You can download the **BRACE** benchmark dataset (including metadata and raw audio) from:
[https://huggingface.co/datasets/gtysssp/audio\_benchmarks](https://huggingface.co/datasets/gtysssp/audio_benchmarks)

To make it easier to work with, you can process the metadata using `src/data.py`, which returns a PyTorch-compatible `Dataset` class.
**Note:** You may need to adjust the directory structure and filenames of the downloaded benchmark to ensure compatibility with the evaluation scripts.


### BRACE-Main Metadata Format

Each item includes file-level metadata and multiple evaluation pairs (human vs. human, human vs. machine, etc.):

```json
[
  {
    "file_name": "Glass Bottles rattle and chink.wav",
    "references": [ ... ],
    "Human-Human": [
      "caption_0",
      "caption_1",
      "caption_0_type",
      "caption_1_type",
      [1, 1, 1]
    ],
    "Human-Machine_1": [ ... ],
    ...
  }
]
```

* **Scores**: `1` = preference for `caption_0`, `-1` = preference for `caption_1`

### BRACE-Hallucination Metadata Format

```json
[
  {
    "file_name": "Glass Bottles rattle and chink.wav",
    "caption_1": [
      "caption_0",
      "caption_1",
      "caption_0_type",
      "caption_1_type",
      {
        "references": [ ... ]
      }
    ],
    "caption_2": [ ... ],
    ...
  }
]
```

### Torch Dataset Format

For use during training or evaluation:

```python
[
  {
    "audio_path": "path/to/audio/file",
    "caption_0": "caption_0",
    "caption_1": "caption_1",
    "answer": 0 or 1,
    "references": [ ... ],
    "pair_type": "Human-Human",
    "caption_type": ["human", "human"]
  },
  ...
]
```

* `answer = 0` means `caption_0` is preferred; `answer = 1` means `caption_1` is preferred.

---

## CLAP Evaluation

### Prerequisites

Follow the setup instructions in the respective CLAP model repositories to install dependencies and download pretrained weights.

> **Note:** Some models (e.g., `m2d-CLAP`) require local cloning and path modification in `src/eval_clap/clap.py`.

### Running Scoring

Run the following script to evaluate a CLAP model:

```bash
./scripts/eval_clap/eval.sh MODEL_NAME DATA_NAME DATA_TYPE
```

* `MODEL_NAME`: Format as `Scorer-CLAP`, e.g., `SIMPLE-LAION_CLAP`, `SLIDE-LAION_CLAP`
* `DATA_NAME`: Dataset name — `AudioCaps` or `Clotho`
* `DATA_TYPE`: Subset of the dataset — `Main` or `Hallu`

Optional parameters like the number of references can be adjusted in `./scripts/eval_clap/pre.sh`.

### Calculating Metrics

```bash
./scripts/eval_clap/calc.sh TARGET
```

* `TARGET`: A single JSON file or a directory containing multiple JSON results

---

## LALM Evaluation

### Prerequisites

1. Clone the relevant LALM GitHub repository.
2. Download the model weights.
3. Update local paths in `src/eval_lalm/factory.py`.

> If evaluating a new model, integrate its inference code into `src/eval_lalm/model.py`.

### Running Inference

#### Qwen Models

Refer to `src/eval_qwen/README.md`.

#### Other Models

Use this script (modify if needed for benchmark compatibility):

```bash
./scripts/eval_lalm/pre.sh MODEL_NAME DATA_NAME DATA_TYPE PROMPT
```

* `MODEL_NAME`: `LTU`, `LTU-AS`, `GAMA`, or `AF2`
* `DATA_NAME`: `AudioCaps` or `Clotho`
* `DATA_TYPE`: `Main` or `Hallu`
* `PROMPT`: Prompt template — see `src/eval_lalm/prompt.py`

You can modify parameters like the number of references in `pre.sh`.

Inference results are saved in the `logs` directory. Each JSON result should follow this format:

```json
{
  "audio_path": "path/to/audio/file",
  "caption_0": "caption_0_text",
  "caption_1": "caption_1_text",
  "answer": "0 or 1",
  "references": [ ... ],
  "score": "total score",                // Only for BRACE-Main
  "pair_type": "pair_type_name",         // Only for BRACE-Main
  "caption_type": ["type_0", "type_1"],  // Only for BRACE-Main
  "prompt": {
    "prompt_template_0": "prompt_0",
    ...
  },
  "output": {
    "prompt_template_0": "output_0",
    ...
  }
}
```

### Summarizing Results

```bash
./scripts/eval_lalm/pre.sh TARGET
```

* `TARGET`: A single JSON file or a directory of JSON files

### Calculating Metrics

```bash
./scripts/eval_lalm/calc.sh TARGET
```

* `TARGET`: A single JSON file or a directory of JSON files
