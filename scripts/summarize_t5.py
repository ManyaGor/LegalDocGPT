import os, re, textwrap, math
from typing import List
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INPUT_PATH  = "outputs/cleaned_text.txt"
POINTS_TXT  = "outputs/model_points.txt"
POINTS_PDF  = "outputs/model_summary.pdf"

MODEL_NAME  = "t5-small"   # lightweight, works on CPU
MAX_INPUT_TOKENS  = 512     # T5-small limit
MAX_SUMMARY_TOKENS = 180    # per chunk
OVERLAP_TOKENS     = 64     # chunk overlap for continuity

# ---------------------- utilities (reuse from baseline) ----------------------
def sanitize(s: str) -> str:
    """Make text safe for core PDF fonts (no Unicode)."""
    repl = {
        "•": "-", "₹": "Rs.", "–": "-", "—": "-",
        "’": "'", "‘": "'", "“": '"', "”": '"', "\u00A0": " "
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    out = out.encode("latin-1", "ignore").decode("latin-1")
    return out

def write_points_txt(points: List[str], path=POINTS_TXT):
    os.makedirs("outputs", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in points:
            f.write(f"- {p.strip()}\n")

def write_points_pdf(points: List[str], title="Simplified Summary (Model)", path=POINTS_PDF):
    os.makedirs("outputs", exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, sanitize(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", size=12)
    for p in points:
        p = sanitize(p.strip())
        wrapped = textwrap.wrap(p, width=90)
        pdf.cell(5)
        pdf.cell(0, 7, f"- {wrapped[0]}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for cont in wrapped[1:]:
            pdf.cell(10)
            pdf.cell(0, 7, cont, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
    pdf.output(path)

# ---------------------- model + chunking ----------------------
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tok, mdl

def chunk_by_tokens(text: str, tok: AutoTokenizer, max_len=MAX_INPUT_TOKENS, overlap=OVERLAP_TOKENS) -> List[str]:
    # encode full text
    ids = tok.encode(text, truncation=False)
    chunks = []
    start = 0
    stride = max_len - overlap
    while start < len(ids):
        end = min(start + max_len, len(ids))
        chunk_ids = ids[start:end]
        chunk_text = tok.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(ids): break
        start += stride
    return chunks

def summarize_chunk(prompt_text: str, tok, mdl) -> str:
    # T5 expects a task prefix like "summarize: "
    inp = "summarize: " + prompt_text
    inputs = tok(
        inp,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )
    output_ids = mdl.generate(
        **inputs,
        max_length=MAX_SUMMARY_TOKENS,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tok.decode(output_ids[0], skip_special_tokens=True)

def sentences_to_bullets(text: str) -> List[str]:
    # split at ., ;, newlines, bullets, etc. keep short, plain lines
    parts = re.split(r"(?:\n|\r|\r\n|•|-|\u2022|;|\.)(?:\s+|$)", text)
    bullets = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= 4:
            bullets.append(p)
    # dedupe while preserving order
    seen, uniq = set(), []
    for b in bullets:
        key = b.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(b)
    return uniq

def compress_bullets(points: List[str], max_points: int = 12) -> List[str]:
    # keep top n by heuristic (prefer those with amounts/dates/obligations)
    def score(s: str):
        sc = 0
        if re.search(r"(?:Rs\.?|₹)\s?\d", s): sc += 2
        if re.search(r"\b\d{1,2}\s+(?:days?|months?)\b", s): sc += 1
        if re.search(r"\bnotice|payment|rent|deposit|liability|termination|confidential|repair|jurisdiction|arbitration\b", s, re.I): sc += 2
        sc += min(len(s)//60, 2)
        return sc
    ranked = sorted(points, key=score, reverse=True)
    # trim and tidy periods
    tidy = []
    for p in ranked[:max_points]:
        p = p.rstrip(" .")
        if not p.endswith("."):
            p += "."
        tidy.append(p)
    return tidy

# ---------------------- main flow ----------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Missing {INPUT_PATH}. Run extract_text.py and preprocess_text.py first.")
        raise SystemExit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if len(raw) < 20:
        print("❌ Input is too short to summarize.")
        raise SystemExit(1)

    print("⏳ Loading model…")
    tokenizer, model = load_model()

    print("✂️  Chunking input…")
    chunks = chunk_by_tokens(raw, tokenizer, max_len=MAX_INPUT_TOKENS, overlap=OVERLAP_TOKENS)
    print(f"   → {len(chunks)} chunk(s)")

    print("🧠 Summarizing chunks…")
    partial_summaries = []
    for i, ch in enumerate(chunks, 1):
        print(f"   • Summarizing chunk {i}/{len(chunks)}")
        summ = summarize_chunk(ch, tokenizer, model)
        partial_summaries.append(summ)

    combined_summary = " ".join(partial_summaries)

    print("🪄 Converting to point-wise bullets…")
    bullets = sentences_to_bullets(combined_summary)
    bullets = compress_bullets(bullets, max_points=12)

    if not bullets:
        print("❌ No bullets produced. Try a different document.")
        raise SystemExit(1)

    write_points_txt(bullets, POINTS_TXT)
    write_points_pdf(bullets, title="Simplified Summary (T5-small)", path=POINTS_PDF)

    print(f"\n✅ Saved bullets: {POINTS_TXT}")
    print(f"✅ Saved PDF:     {POINTS_PDF}")
