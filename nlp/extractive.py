from typing import List, Dict, Tuple
import math
import re

LEGAL_CUES = [
    'shall', 'must', 'liable', 'except', 'provided that', 'breach', 'penalty', 'cap',
    'term', 'years', 'months', 'jurisdiction', 'notice', 'return', 'destroy', 'license',
    'confidential', 'governing law', 'arbitration', 'warranty', 'indemnity', 'payment',
]


def simple_embed(sentence: str) -> List[float]:
    # Very light placeholder embedding: bag-of-cues vector
    vec = [0.0] * len(LEGAL_CUES)
    low = sentence.lower()
    for i, cue in enumerate(LEGAL_CUES):
        if cue in low:
            vec[i] = 1.0
    return vec


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def score_sentence(sentence: str) -> float:
    base = len(sentence) ** 0.25  # slight length prior
    cues = sum(1 for cue in LEGAL_CUES if cue in sentence.lower())
    return base + 0.8 * cues


def mmr_select(sentences: List[str], k: int, lam: float = 0.7) -> List[int]:
    # MMR over simple embeddings
    if not sentences:
        return []
    embs = [simple_embed(s) for s in sentences]
    # query as average cue vector
    q = [sum(col)/max(1.0, len(embs)) for col in zip(*embs)] if embs else [0.0]*len(LEGAL_CUES)

    selected: List[int] = []
    cand = list(range(len(sentences)))

    while cand and len(selected) < k:
        best_i = None
        best_score = -1e9
        for i in cand:
            sim_to_q = cosine(embs[i], q)
            sim_to_sel = 0.0
            if selected:
                sim_to_sel = max(cosine(embs[i], embs[j]) for j in selected)
            score = lam * sim_to_q - (1 - lam) * sim_to_sel
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)  # type: ignore
        cand.remove(best_i)      # type: ignore
    return selected


def select_sentences(chunk_text: str, heading: str, page: int, top_k: int = 10) -> List[Dict]:
    # naive sentence split
    sents = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
    sents = [s.strip() for s in sents if s.strip()]

    # score
    scores = [score_sentence(s) for s in sents]
    # pick more than needed, then MMR
    top_idx = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)[: min(len(sents), top_k*2)]
    top_sents = [sents[i] for i in top_idx]

    mmr_idx_rel = mmr_select(top_sents, k=min(len(top_sents), top_k), lam=0.7)
    final_idx = [top_idx[i] for i in mmr_idx_rel]

    final = []
    for i in sorted(final_idx):
        final.append({
            "sentence": sents[i],
            "heading": heading,
            "page": page
        })
    return final

