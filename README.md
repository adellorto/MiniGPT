# MiniGPT

A comparative study of tokenization strategies and language model architectures for text generation, built on [Andrej Karpathy's miniGPT](https://github.com/karpathy/ng-video-lecture).

We implement three models of increasing complexity -- a Neural Bigram Model, a GPT Language Model with multi-head self-attention, and a Monte Carlo Dropout GPT for uncertainty quantification -- and evaluate them across three tokenization methods and two datasets.

## Project Structure

```
.
├── tokenizer.py                     # Character-level, Tiktoken, and MinBPE tokenizer classes
├── precompute_tokens.py             # Script to precompute and cache tokenized datasets
├── minbpe/                          # Byte Pair Encoding tokenizer library
│   ├── base.py                      # Base tokenizer with BPE utilities
│   ├── basic.py                     # Byte-level BPE tokenizer
│   ├── regex.py                     # Regex-based BPE (GPT-2/GPT-4 patterns)
│   └── gpt4.py                      # GPT-4 tokenizer wrapper
├── bigram_model_colab.ipynb         # Bigram model experiments (Google Colab)
├── attention_model_mc_dropout.ipynb # MC Dropout GPT experiments (Google Colab)
├── results/                         # Saved experiment outputs (plots, metrics)
│   ├── bigram/
│   └── attention/
└── requirements.txt
```

> **Note:** The notebooks are designed to run on **Google Colab** and are not included in this repository as standalone runnable files. Upload them to Colab (or your Google Drive) before running — see [Running the Experiments](#running-the-experiments) below.

## Environment Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (experiments were run on an NVIDIA Tesla T4 via Google Colab)

### Installation

```bash
git clone https://github.com/adellorto/MiniGPT.git
cd MiniGPT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation

The datasets are not included in the repository. To prepare them:

1. **Tiny Shakespeare**: download `input.txt` from [Karpathy's char-rnn](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and place it in `data/input.txt`.

2. **Text8**: downloaded automatically at runtime from HuggingFace (`roshbeed/text8-dataset`).

Once the data is in place, precompute the tokenized tensors:

```bash
python precompute_tokens.py --dataset shakespeare
python precompute_tokens.py --dataset text8
```

This saves cached `.pt` files to `data/cache/`, which the notebooks load at training time.

## Running the Experiments

The experiments are designed to run on **Google Colab** with GPU acceleration. Each notebook mounts Google Drive to access the project files and cached data.

1. Upload the repository to Google Drive.
2. Open a notebook in Colab and select a GPU runtime.
3. Run all cells. Results (plots, metrics, model checkpoints) are saved to timestamped folders.

Alternatively, the notebooks can be run locally with a CUDA GPU by adjusting the file paths in the Drive-mount cells.

### Notebooks

| Notebook | Description |
|---|---|
| `bigram_model_colab.ipynb` | Neural Bigram Model -- single embedding table, no attention |
| `attention_model_mc_dropout.ipynb` | MC Dropout GPT -- keeps dropout active at inference for uncertainty estimation |

## Models

### Tokenizers

| Tokenizer | Description | Vocab Size (Shakespeare / Text8) |
|---|---|---|
| **Character-level** | Maps each unique character to an integer | 65 / 27 |
| **Tiktoken (cl100k)** | OpenAI's BPE tokenizer with vocabulary remapping | 12,111 / -- |
| **MinBPE** | Custom BPE trained on the corpus (max 100k chars) | 807 / 794 |

Tiktoken was skipped on Text8 due to GPU memory constraints.

### Architectures

| Parameter | Bigram Model | GPT Model | MC Dropout GPT |
|---|---|---|---|
| Batch size | 64 | 128 | 128 |
| Block size (T) | 128 | 256 | 256 |
| Max iterations | 3,000 | 5,000 | 5,000 |
| Learning rate | 1e-2 | 1e-3 | 1e-3 |
| Embedding size (C) | Vocab size | 128 | 128 |
| Attention heads | -- | 4 | 4 |
| Transformer layers | -- | 4 | 4 |
| Dropout | -- | 0.4 (train only) | 0.4 (train + inference) |

## Results

### Bigram Model

**Tiny Shakespeare**

| Tokenizer | \|V\| | Val Loss | Val PPL | Norm Loss | Time (s) |
|---|---|---|---|---|---|
| CharacterLevel | 65 | 2.486 | 12.01 | 0.596 | 19.1 |
| Tiktoken (cl100k) | 12111 | 6.289 | 538.36 | 0.669 | 278.4 |
| MinBPE | 807 | 4.231 | 68.80 | 0.632 | 19.6 |

**Text8**

| Tokenizer | \|V\| | Val Loss | Val PPL | Norm Loss | Time (s) |
|---|---|---|---|---|---|
| CharacterLevel | 27 | 2.383 | 10.84 | 0.723 | 18.9 |
| MinBPE | 794 | 4.167 | 64.51 | 0.624 | 20.6 |

### GPT Language Model

**Tiny Shakespeare**

| Tokenizer | \|V\| | Val Loss | Val PPL | Norm Loss | Time (s) |
|---|---|---|---|---|---|
| CharacterLevel | 65 | 1.565 | 4.78 | 0.375 | 682.4 |
| Tiktoken (cl100k) | 12111 | 12.611 | 299731.8 | 1.341 | 1215.7 |
| MinBPE | 807 | 3.524 | 33.92 | 0.526 | 714.4 |

**Text8**

| Tokenizer | \|V\| | Val Loss | Val PPL | Norm Loss | Time (s) |
|---|---|---|---|---|---|
| CharacterLevel | 27 | 1.397 | 4.04 | 0.424 | 680.8 |
| MinBPE | 794 | 3.207 | 24.70 | 0.480 | 713.4 |

## Acknowledgements

Built on [Andrej Karpathy's miniGPT](https://github.com/karpathy/ng-video-lecture) lecture series. Tokenizer implementations adapted from [minbpe](https://github.com/karpathy/minbpe).
