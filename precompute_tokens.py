"""
Precompute tokenized tensors for the bigram model notebook.

Usage:
    python precompute_tokens.py --dataset shakespeare
    python precompute_tokens.py --dataset text8

Saves to: data/cache/{tokenizer_name}_{dataset}.pt
Each file contains:
  {
    'encoded':   torch.LongTensor,  # full encoded sequence
    'vocab_size': int,              # compact vocab size
  }

Run once per dataset; the notebook loads from cache on subsequent runs.
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))

from tokenizer import CharachterLevelTokenizer, TiktokenTokenizer, MinbpeTokenizer

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['shakespeare', 'text8'], required=True)
args = parser.parse_args()

DATASET = args.dataset
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cache')
MINBPE_MAX_CHARS = 100000

os.makedirs(CACHE_DIR, exist_ok=True)

# --- Load text ---
if DATASET == 'shakespeare':
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'input.txt')
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
elif DATASET == 'text8':
    from huggingface_hub import hf_hub_download
    import json
    sentences_path = hf_hub_download(
        repo_id="roshbeed/text8-dataset",
        filename="text8_sentences.json",
    )
    with open(sentences_path, 'r') as f:
        data = json.load(f)
    text = ' '.join(data['sentences'])

print(f"Dataset: {DATASET} | Length: {len(text):,} chars")

# --- Tokenize and cache ---
tokenizers_to_run = [
    ('char',     lambda: CharachterLevelTokenizer(text)),
    ('tiktoken', lambda: TiktokenTokenizer(text)),
    ('minbpe',   lambda: MinbpeTokenizer(text, max_chars=MINBPE_MAX_CHARS)),
]

for name, build_tokenizer in tokenizers_to_run:
    cache_path = os.path.join(CACHE_DIR, f'{name}_{DATASET}.pt')
    if os.path.exists(cache_path):
        print(f"[{name}] Cache already exists at {cache_path}, skipping.")
        continue

    print(f"[{name}] Building tokenizer and encoding...")
    tok = build_tokenizer()
    encoded = torch.tensor(tok.train_encoded, dtype=torch.long)

    torch.save({'encoded': encoded, 'vocab_size': len(tok.vocab)}, cache_path)
    print(f"[{name}] Saved {len(encoded):,} tokens, vocab={len(tok.vocab)} -> {cache_path}")

print("\nDone.")
