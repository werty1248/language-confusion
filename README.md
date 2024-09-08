# Language Confusion Benchmark with Generation Script

I added output generation script to the official [Language Confusion Benchmark](https://github.com/for-ai/language-confusion) to check for language confusion in the various models.

The output generation script allows for online inference using the API and offline (local) inference using [vLLM](https://github.com/vllm-project/vllm).

## Installation

```bash
git clone https://github.com/werty1248/language-confusion.git
cd language-confusion
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# View the test sets
unzip test_sets.zip
```

## Generation

### Offline(vLLM)

```bash
python generate_offline.py --model [MODEL_NAME] --task [TASK] --source [SOURCEs] --language [LANGUAGEs] --tensor-parallel-size [#GPUs] --hf-token [YOUR_HF_TOKEN] --system-prompt [SYSTEM_PROMPT]
```

- [TASK]: all(default), monolingual, crosslingual
- [SOURCEs]: all(default), aya, dolly, okapi, native
- [LANGUAGEs]: all(default), ar, de, en, es, fr, hi, id, it, ja, ko, pt, ru, tr, vi, zh

For instance, to evaluate the Korean and Japanese LPR/WPR of the [Qwen/Qwen2-7B-Instruct](Qwen/Qwen2-7B-Instruct) model using 2 GPUs with the default system prompt,

```bash
python generate_offline.py --model Qwen/Qwen2-7B-Instruct --language ko,ja --tensor-parallel-size 2 --hf-token "YOUR_HF_TOKEN" --system-prompt "You are a helpful assistant."
```

### Online(API)

```bash
python generate_online.py --url [URL] --api-key [API_KEY] --model [MODEL_NAME] --task [TASK] --source [SOURCEs] --language [LANGUAGEs] --concurrency [#_OF_CONCURRENT_QUERYS] --system-prompt [SYSTEM_PROMPT]
```

If you want to perform the same test using OpenRouter,

```bash
python generate_online.py --url https://openrouter.ai/api/v1/chat/completions --api_key "YOUR_OPENROUTER_KEY" --model qwen/qwen-2-7b-instruct --concurrency 2 --language ko,ja --system-prompt "You are a helpful assistant."
```

## Evaluation

```bash
python compute_metrics.py outputs/[MODEL_NAME].csv
```

---

# Language Confusion Benchmark

The Language Confusion Benchmark (LCB) evaluates to what extent LLMs are unable to consistently generate text in the user's desired language, i.e., their "language confusion".
The benchmark consists of English and multilingual prompts from existing and newly created sources covering 15 typologically diverse languages. Evaluation is in two settings:

- **Monolingual generation**: A user queries a model in language $l$ and expects a response in language $l$.
- **Cross-lingual generation**: A user instructs a model in language $l$ to fulfill a request in another language $l'$.

We provide further information about the data, model completions, metrics, and how to use the benchmark below.

## Data

The benchmark data is available in `test_sets` in separate `csv` files for each dataset and language. Each file contains a `prompt` column with additional metadata. We provide an overview of the datasets and languages per dataset below:

| Task          | Dataset name    | Reference                                                         | Nature of data  | $\|L\|$ | $\|D\|$ | Languages                                      | $W$ |
|---------------|-----------------|-------------------------------------------------------------------|-----------------|:-------:|:-------:|------------------------------------------------|:---:|
| Monolingual   | Aya             | [Singh et al. (2024)](https://arxiv.org/abs/2402.06619)           | Human-generated |   100   |   500   | en, tr, ar, zh, pt                             |  9  |
| Monolingual   | Dolly           | [Singh et al. (2024)](https://arxiv.org/abs/2402.06619)           | MT post-edited  |   100   |   500   | hi, ru, fr, ar, es                             |  10 |
| Monolingual   | Okapi           | [Lai et al. (2023)](https://aclanthology.org/2023.emnlp-demo.28/) | Synthetic + MT  |   100   |   1.2k  | en, fr, it, de, zh, vi, ru, es, id, pt, ar, hi |  13 |
| Monolingual   | Native prompts  | Ours                                                              | Human-generated |   100   |   400   | es, fr, ja, ko                                 |  19 |
| Cross-lingual | Okapi           | [Lai et al. (2023)](https://aclanthology.org/2023.emnlp-demo.28/) | Synthetic       |   100   |   1.5k  | $\mathcal{L}$                                  |  15 |
| Cross-lingual | ShareGPT        | [https://sharegpt.com/](https://sharegpt.com/)                    | Human-generated |   100   |   1.5k  | $\mathcal{L}$                                  |  18 |
| Cross-lingual | Complex prompts | Ours                                                              | Human-generated |    99   |   1.5k  | $\mathcal{L}$                                  | 159 |

$|D|$ is the total number of examples per dataset, $|L|$ is the number of examples per language, and $W$ is the median length in words of the prompts in each dataset. For the cross-lingual setting, the model is instructed in English to generate in the target language $l \in \mathcal{L}$ where $\mathcal{L} = \{\text{fr, de, es, pt, it, ja, ko, zh, ar, tr, hi, ru, id, vi} \}$.

## Model Completions

We provide model completions in `outputs`. Outputs are stored in a single `csv` file per model with `model`, `completion`, `task`, `source`, and `language` columns. Completions are available for the following models:

- [Command R](https://huggingface.co/CohereForAI/c4ai-command-r-v01) (35B parameters)
- [Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus) (104B parameters)

We generated at most 100 tokens per prompt using nucleus sampling with $p=0.75$ and $T=0.3$.

## Metrics

We use the following metrics for evaluation:

- **Line-level pass rate (LPR)**: percentage of model responses that pass our line-level language confusion detector without error. A response is "correct" if all lines match the user's desired language.
- **Word-level pass rate (WPR)**: percentage of model responses where all words are in the desired language (excluding responses with line-level errors). We only identify erroneous English words in non-Latin script languages.

We use fastText as line-level language ID detector and the English word list in `words` to detect word-level language confusion.

## Using the Benchmark

You can obtain completions using your preferred LLM API or use the completions we provide in `outputs`.

We provide a script to calculate LPR and WPR metrics based on output files:

```bash
git clone https://github.com/for-ai/language-confusion.git
cd language-confusion
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# View the test sets
unzip test_sets.zip

# Compute and print LPR and WPR for Command R+
python compute_metrics.py outputs/command-r-plus.csv
```

## Citation

If you are using the data or scripts in this repo, please cite:
```
@misc{marchisio2024understanding,
  Author = {Kelly Marchisio and Wei-Yin Ko and Alexandre Bérard and Théo Dehaze and Sebastian Ruder},
  Title = {Understanding and Mitigating Language Confusion in LLMs},
  Year = {2024},
  Eprint = {arXiv},
}
```
