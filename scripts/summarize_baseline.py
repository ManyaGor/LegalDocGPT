# scripts/summarize_baseline.py
"""
Baseline summarization runner (updated to use reportlab for Unicode-safe PDFs).
- Reads dataset/dataset.jsonl if present, otherwise reads data/input files
- Runs a small summarizer (t5-small or google/mt5-small) as configured
- Writes text predictions to data/predictions_text/
- Writes Unicode PDFs to data/predictions_pdf/
"""

import os
import json
from pathlib import Path
import textwrap

# transformers model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# reportlab for Unicode-safe PDF creation
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

ROOT = Path(".")
DATASET_JSONL = ROOT / "dataset" / "dataset.jsonl"
INPUT_DIR = ROOT / "data" / "input"
PRED_TEXT_DIR = ROOT / "data" / "predictions_text"
PRED_PDF_DIR = ROOT / "data" / "predictions_pdf"

MODEL_NAME = "google/mt5-small"   # baseline; change if you want t5-small or other

# Unicode font to embed. We'll try to use DejaVu Sans (common). If not present, we fallback to built-in
FONTS_DIR = ROOT / "assets" / "fonts"
FONT_FILE = FONTS_DIR / "DejaVuSans.ttf"
FONT_NAME = "DejaVuSans"

# Ensure directories
PRED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
PRED_PDF_DIR.mkdir(parents=True, exist_ok=True)
FONTS_DIR.mkdir(parents=True, exist_ok=True)

def register_font():
    # If the DejaVu font file is available, register it. Otherwise fall back to default (may lose some glyphs).
    if FONT_FILE.exists():
        try:
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_FILE)))
            return FONT_NAME
        except Exception as e:
            print("Warning: could not register TTF font:", e)
    # fallback: use Helvetica (limited unicode)
    return "Helvetica"

def read_dataset_or_inputs():
    records = []
    if DATASET_JSONL.exists():
        with open(DATASET_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append({"id": rec.get("id") or "unknown", "input": rec.get("input","")})
                except Exception:
                    continue
    else:
        # fallback: read all docx/pdf files in data/input (simple)
        for p in sorted(INPUT_DIR.iterdir()):
            if p.suffix.lower() in (".docx", ".pdf"):
                text = p.read_text(encoding="utf-8", errors="ignore") if p.suffix.lower() == ".txt" else ""
                records.append({"id": p.stem, "input": text})
    return records

def make_pdf_reportlab(title, bullets, out_pdf_path, fontname):
    # Create a PDF that can render unicode (if font registered)
    c = canvas.Canvas(str(out_pdf_path), pagesize=A4)
    width, height = A4
    left_margin = 20 * mm
    top = height - 20 * mm
    y = top

    # choose font (size 14 for title, 11 for body)
    try:
        c.setFont(fontname, 14)
    except Exception:
        c.setFont("Helvetica", 14)
    c.drawString(left_margin, y, title)
    y -= 12 * mm

    try:
        c.setFont(fontname, 11)
    except Exception:
        c.setFont("Helvetica", 11)

    # write bullets, wrap lines
    max_width = width - 2 * left_margin
    for b in bullets:
        wrapped = textwrap.wrap(b, width=90)  # approximate wrap
        # bullet symbol — use a plain dash if bullet char not supported
        bullet_symbol = "•"
        # draw first line with bullet
        if not wrapped:
            continue
        line = f"{bullet_symbol} {wrapped[0]}"
        c.drawString(left_margin + 4 * mm, y, line)
        y -= 6 * mm
        # subsequent wrapped lines
        for cont in wrapped[1:]:
            c.drawString(left_margin + 10 * mm, y, cont)
            y -= 6 * mm
        # check for page break
        if y < 25 * mm:
            c.showPage()
            try:
                c.setFont(fontname, 11)
            except Exception:
                c.setFont("Helvetica", 11)
            y = top

    c.save()

def run_model_on_texts(records):
    print("Loading model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

    for rec in records:
        doc_id = rec.get("id", "unknown")
        text = rec.get("input", "")
        if not text or not text.strip():
            print(f"  Skipping empty file: {doc_id}")
            continue

        # Truncate to keep runtime reasonable
        max_input_chars = 3000
        short_text = text if len(text) <= max_input_chars else text[:max_input_chars]

        prompt = short_text.replace("\n", " ")

        try:
            out = summarizer(prompt, max_length=256, min_length=40, do_sample=False)
            summary = out[0]["summary_text"]
        except Exception as e:
            print(f"  Error processing {doc_id} : {e}")
            summary = ""

        # create bullets: split sentences, keep them short and specific
        bullets = [s.strip() for s in summary.split(". ") if s.strip()]
        if not bullets and summary:
            bullets = [summary.strip()]

        # write plain text
        txt_path = PRED_TEXT_DIR / f"{doc_id}_pred.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(summary)

        # write unicode-safe PDF
        pdf_path = PRED_PDF_DIR / f"{doc_id}_pred.pdf"
        fontname = register_font()
        make_pdf_reportlab(doc_id, bullets, pdf_path, fontname)

        print(f"Processing: {doc_id}\n  Wrote: {pdf_path.relative_to(ROOT)}")

def main():
    records = read_dataset_or_inputs()
    if not records:
        print("No records found to process. Put dataset/dataset.jsonl or files under data/input.")
        return
    run_model_on_texts(records)
    print("Done. Predictions (text) in:", PRED_TEXT_DIR, "PDFs in:", PRED_PDF_DIR)

if __name__ == "__main__":
    main()
