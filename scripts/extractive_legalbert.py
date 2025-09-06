# scripts/extractive_legalbert.py
"""
Extractive sentence selector using LegalBERT embeddings.

Reads dataset/dataset.jsonl (created earlier by build_dataset.py) OR data/input files
and writes selected sentences (ordered) for each doc to:
  data/extractive_inputs/{docid}_extract.txt

How selection works (simple, robust):
 - Split doc into sentences (nltk punkt).
 - Encode sentences with LegalBERT (mean-pooling).
 - Compute document embedding = mean of sentence embeddings.
 - Score each sentence by cosine(sim(sentence, doc_embedding)).
 - Pick top-K sentences, then order them by original position.
 - Save to file for the next step (rewrite with mT5).
"""
import os
import json
import math
import sys
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

import torch
from transformers import AutoTokenizer, AutoModel

# ----- Config -----
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"  # LegalBERT
DATASET_JSONL = Path("dataset/dataset.jsonl")   # created earlier
OUT_DIR = Path("data/extractive_inputs")
BATCH_SIZE = 16  # sentence embedding batch size
TOP_K_MIN = 3
TOP_K_MAX = 12

# ----- Utilities -----
def mean_pooling(last_hidden_state, attention_mask):
    # last_hidden_state: (B, T, H), attention_mask: (B, T)
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return (summed / counts)

def cosine_sim(a, b):
    # a: (N, D), b: (D,) or (1, D)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a_norm @ b_norm.T).squeeze(-1)

# ----- Main -----
def read_dataset(jsonl_path: Path):
    records = []
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found. Run build_dataset.py first.")
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            # rec should have 'id' and 'input' fields
            records.append({"id": rec.get("id"), "text": rec.get("input", "")})
    return records

def batch_embed(sentences: List[str], tokenizer, model, device):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i:i+BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            last = out.last_hidden_state  # (B, T, H)
            pool = mean_pooling(last, attn)  # (B, H)
            embeddings.append(pool.cpu().numpy())
    if embeddings:
        return np.vstack(embeddings)
    else:
        return np.zeros((0, model.config.hidden_size), dtype=float)

def select_top_k_sentences(text: str, tokenizer, model, device):
    # cut into sentences
    sents = sent_tokenize(text)
    if len(sents) == 0:
        return []
    # embed sentences
    embeddings = batch_embed(sents, tokenizer, model, device)  # (N, D)
    # doc embedding = mean
    doc_emb = embeddings.mean(axis=0)
    # score sentences by cosine similarity to doc embedding
    scores = cosine_sim(embeddings, doc_emb)  # (N,)
    # choose k
    k = min(TOP_K_MAX, max(TOP_K_MIN, math.ceil(len(sents) / 7)))
    # get top k indices
    top_idx = np.argsort(-scores)[:k]
    # order them by original position
    top_idx_sorted = sorted(top_idx.tolist())
    selected = [sents[i] for i in top_idx_sorted]
    return selected

def main():
    print("Loading model:", MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = read_dataset(DATASET_JSONL)
    print(f"Processing {len(records)} records...")
    for rec in tqdm(records):
        docid = rec["id"]
        text = rec["text"] or ""
        if (not text) or (len(text.strip()) == 0):
            # skip empty
            open(OUT_DIR / f"{docid}_extract.txt", "w", encoding="utf-8").write("")
            continue
        try:
            selected = select_top_k_sentences(text, tokenizer, model, device)
            # write selected sentences, each on its own line, in original order
            out_path = OUT_DIR / f"{docid}_extract.txt"
            with out_path.open("w", encoding="utf-8") as wf:
                for s in selected:
                    wf.write(s.strip() + "\n")
        except Exception as e:
            print(f"Error processing {docid}: {e}")
            open(OUT_DIR / f"{docid}_extract.txt", "w", encoding="utf-8").write("")
    print("Done. Extractive outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
