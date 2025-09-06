"""
scripts/extractive_summarizer.py

Extractive summarizer for LegalDocGPT pipeline.
- Tries to use nltk punkt_tab sentence tokenizer.
- Falls back to a regex-based sentence splitter if punkt_tab missing.
- Uses TF-IDF to score sentences and selects top-N sentences.
- Writes plain text and PDF outputs (UTF-8) into:
    data/predictions_text/{id}_extractive.txt
    data/predictions_pdf/{id}_extractive.pdf

Usage:
    python scripts/extractive_summarizer.py
"""

import os
import json
import math
import re
from pathlib import Path
from typing import List

# NLP / ML
try:
    import nltk
    from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# PDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------- Config ----------
DATASET_PATH = Path("dataset/dataset.jsonl")  # dataset created earlier
PRED_TEXT_DIR = Path("data/predictions_text")
PRED_PDF_DIR = Path("data/predictions_pdf")
PDF_FONT_NAME = "DejaVuSans"
PDF_FONT_FILE = "DejaVuSans.ttf"  # will try to use system or local fallback
MAX_SENTENCES = 8  # max bullets
MIN_SENTENCES = 3  # min bullets
# ----------------------------

def ensure_dirs():
    PRED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_PDF_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset(jsonl_path: Path):
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found. Make sure you've built the dataset.")
    records = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def try_nltk_sentence_split(text: str) -> List[str]:
    # tries to use nltk punkt_tab -> sent_tokenize(language="english") might require punkt_tab
    if not _HAS_NLTK:
        raise LookupError("nltk not available")
    # prefer 'punkt_tab' if installed; sent_tokenize will use tokenizer lookup
    try:
        sents = nltk_sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    except LookupError as e:
        # resource missing
        raise e

def regex_sentence_split(text: str) -> List[str]:
    """
    Basic regex sentence splitter:
    splits on . ? ! followed by whitespace and a capital / numeric / quote / bracket
    Keeps reasonably long pieces (filter out tiny fragments).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Protect abbreviations common in legal docs (approx): "No.", "Mr.", "Ms.", "Dr.", "etc.", "i.e.", "e.g."
    # A simple heuristic: temporarily mask common abbreviations to avoid splitting there.
    abbrevs = ["No\\.", "Mr\\.", "Ms\\.", "Mrs\\.", "Dr\\.", "Prof\\.", "Inc\\.", "Ltd\\.", "Co\\.", "i\\.e\\.", "e\\.g\\.", "etc\\."]
    mask_map = {}
    for i, ab in enumerate(abbrevs):
        token = f"__ABBR{i}__"
        mask_map[token] = ab.replace("\\", "")
        text = re.sub(ab, token, text)

    # split
    parts = [p.strip() for p in re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9"\'\(\[])', text) if p.strip()]

    # restore abbreviations
    def restore(s):
        for token, orig in mask_map.items():
            s = s.replace(token, orig)
        return s

    parts = [restore(p) for p in parts]
    # filter very short fragments
    parts = [p for p in parts if len(p) > 30]  # adjust threshold if you want shorter sentences
    if not parts:
        # fallback to newline-based chunks
        parts = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    return parts

def sentences_from_text(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Try nltk punkt_tab first (if available)
    if _HAS_NLTK:
        try:
            return try_nltk_sentence_split(text)
        except LookupError:
            # punkt_tab missing; fall through to regex fallback
            pass
        except Exception:
            pass
    # fallback
    return regex_sentence_split(text)

def choose_top_sentences(sents: List[str], top_k: int) -> List[str]:
    if not sents:
        return []
    # If few sentences, return them all (or up to top_k)
    if len(sents) <= top_k:
        return sents

    # Compute TF-IDF over sentences
    try:
        vect = TfidfVectorizer(stop_words="english", max_df=0.9)
        X = vect.fit_transform(sents)  # shape (n_sents, n_features)
        # score sentences by sum of TF-IDF weights
        scores = X.sum(axis=1).A1  # convert to 1D array
    except Exception:
        # fallback: length-based heuristic
        scores = np.array([len(s) for s in sents], dtype=float)

    # pick top_k by score, preserve original order
    top_idx = np.argsort(-scores)[:top_k]
    top_idx_sorted = sorted(top_idx.tolist())
    selected = [sents[i] for i in top_idx_sorted]
    return selected

def make_summary_points(sents: List[str]) -> List[str]:
    # Basic cleaning of extracted sentences
    points = []
    for s in sents:
        s = s.strip()
        # Remove excessive whitespace/newlines inside sentence
        s = re.sub(r'\s+', ' ', s)
        # Optionally trim long sentences to a max char length (we keep full for now)
        points.append(s)
    return points

def write_text_output(id_: str, points: List[str]):
    out_path = PRED_TEXT_DIR / f"{id_}_extractive.txt"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{id_} — Extractive Summary\n\n")
        for i, p in enumerate(points, start=1):
            fh.write(f"{i}. {p}\n\n")
    return out_path

def register_pdf_font():
    # Try to register a DejaVuSans font (commonly available). If not found, try to use built-in Helvetica (may not support all unicode).
    try:
        # try local first (project folder) then system
        candidates = [
            Path(PDF_FONT_FILE),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/Arial.ttf"),
        ]
        font_file = None
        for cand in candidates:
            if cand and cand.exists():
                font_file = str(cand)
                break
        if font_file is None:
            # try to find any TTF in Windows fonts folder as last resort
            win_fonts = Path("C:/Windows/Fonts")
            if win_fonts.exists():
                ttf_list = list(win_fonts.glob("*.ttf"))
                if ttf_list:
                    font_file = str(ttf_list[0])
        if font_file is None:
            # register default (Helvetica) — note: limited unicode coverage
            return False, None
        pdfmetrics.registerFont(TTFont(PDF_FONT_NAME, font_file))
        return True, font_file
    except Exception:
        return False, None

def write_pdf_output(id_: str, title: str, points: List[str]):
    out_path = PRED_PDF_DIR / f"{id_}_extractive.pdf"
    # register font
    ok, font_file = register_pdf_font()
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    margin = 40
    x = margin
    y = height - margin - 20
    # use font
    if ok:
        c.setFont(PDF_FONT_NAME, 14)
    else:
        c.setFont("Helvetica", 14)

    # Title
    c.drawString(x, y, title)
    y -= 24

    if ok:
        c.setFont(PDF_FONT_NAME, 11)
    else:
        c.setFont("Helvetica", 11)

    # Write pointwise bullets with simple word-wrapping
    max_width = width - 2 * margin
    from reportlab.lib.utils import simpleSplit

    for i, p in enumerate(points, start=1):
        bullet = f"{i}. "
        # wrap the paragraph
        lines = simpleSplit(p, PDF_FONT_NAME if ok else "Helvetica", 11, max_width - 20)
        if not lines:
            continue
        # draw first line with bullet
        c.drawString(x + 6, y, bullet + lines[0])
        y -= 14
        # draw remaining wrapped lines
        for ln in lines[1:]:
            # indent subsequent lines
            c.drawString(x + 20, y, ln)
            y -= 14
        y -= 6
        if y < margin + 40:
            c.showPage()
            if ok:
                c.setFont(PDF_FONT_NAME, 11)
            else:
                c.setFont("Helvetica", 11)
            y = height - margin

    c.save()
    return out_path

def process_record(rec):
    id_ = rec.get("id") or rec.get("file") or "unknown"
    input_text = rec.get("input") or rec.get("text") or rec.get("full_text") or ""
    if not input_text or not input_text.strip():
        print(f"  Skipping empty file: {id_}")
        return None

    sents = sentences_from_text(input_text)
    if not sents:
        print(f"  No sentences extracted for {id_} (empty after split).")
        return None

    # decide number of sentences to pick
    # pick between MIN_SENTENCES and MAX_SENTENCES, proportionally to length
    k = max(MIN_SENTENCES, min(MAX_SENTENCES, max(1, len(sents) // 6)))
    # ensure at least MIN_SENTENCES, at most MAX_SENTENCES
    k = min(MAX_SENTENCES, max(MIN_SENTENCES, k))

    selected = choose_top_sentences(sents, k)
    points = make_summary_points(selected)
    # title: preserve id as title; the pipeline expects original doc title - dataset may contain title field
    title = rec.get("title") or f"{id_} — Simplified Summary"
    text_path = write_text_output(id_, points)
    pdf_path = write_pdf_output(id_, title, points)
    return {"id": id_, "text": str(text_path), "pdf": str(pdf_path), "n_points": len(points)}

def main():
    ensure_dirs()
    print("Loading dataset...", end="")
    records = load_dataset(DATASET_PATH)
    print(f" done. {len(records)} records found.")
    print("Processing records...")
    results = []
    for rec in records:
        id_ = rec.get("id", "unknown")
        print(f"Processing: {id_}")
        try:
            res = process_record(rec)
            if res:
                print(f"  Wrote: {res['pdf']}")
                results.append(res)
        except Exception as e:
            print(f"  Error processing {id_} : {e}")
    print(f"Done. Wrote {len(results)} predictions to {PRED_PDF_DIR}")
    # also write a small index file
    idx_path = Path("dataset") / "extractive_predictions_index.json"
    with idx_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"Index written: {idx_path}")

if __name__ == "__main__":
    main()
