# scripts/abstractive_rewrite_mt5.py
"""
Abstractive rewrite step using mT5 (google/mt5-small).
Reads extractive inputs from data/extractive_inputs/*.txt and writes:
 - data/predictions_text/{docid}_pred.txt
 - data/predictions_pdf/{docid}_pred.pdf

Generation settings tuned to avoid extremely short outputs.
"""

import os
from pathlib import Path
from tqdm import tqdm
import nltk
nltk.download('punkt', quiet=True)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# PDF
from fpdf import FPDF

# ----- Config -----
MODEL_NAME = "google/mt5-small"  # small mT5
EXTRACT_DIR = Path("data/extractive_inputs")
OUT_TEXT_DIR = Path("data/predictions_text")
OUT_PDF_DIR = Path("data/predictions_pdf")

# Generation hyperparams (tune these if outputs are too short or too verbose)
GEN_CFG = dict(
    max_new_tokens=280,
    num_beams=4,
    early_stopping=True,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    do_sample=False,
)

# Fonts to try for PDFs (Windows common paths)
FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\DejaVuSans.ttf",  # sometimes present
]

# ----- Helpers -----
def find_font():
    for p in FONT_PATHS:
        if os.path.exists(p):
            return p
    return None

def make_pdf_from_text(text: str, outpath: Path, font_path=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    if font_path:
        try:
            # register a TTF font (unicode)
            pdf.add_font("UserFont", "", font_path, uni=True)
            pdf.set_font("UserFont", size=12)
        except Exception as e:
            # fallback to core font
            pdf.set_font("Arial", size=12)
    else:
        # try core font (may be limited)
        pdf.set_font("Arial", size=12)
    # write text (split lines)
    for line in text.splitlines():
        # ensure no extremely long lines block layout
        pdf.multi_cell(0, 7, line)
    pdf.output(str(outpath))

def extract_title_from_text(text: str):
    # simple heuristic: first non-empty line (up to 120 chars)
    for ln in text.splitlines():
        s = ln.strip()
        if len(s) > 0:
            if len(s) > 120:
                return s[:120]
            return s
    return "Document"

def build_prompt(title: str, sentences: str):
    # Keep prompt concise. Use title and sentences.
    prompt = (
        f"Title: {title}\n\n"
        "Task: Convert the following extracted sentences into a clear, numbered, "
        "concise summary in simple English. Keep each point short (1-2 sentences). "
        "Preserve the meaning and include all important points. Output as numbered bullets.\n\n"
        "Sentences:\n"
        f"{sentences}\n\n"
        "Summary:"
    )
    return prompt

# ----- Main -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Loading tokenizer and model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PDF_DIR.mkdir(parents=True, exist_ok=True)

    font_path = find_font()
    if font_path:
        print("Found font for PDF:", font_path)
    else:
        print("No TTF font found in known locations; PDFs may not render special characters properly.")

    files = sorted(EXTRACT_DIR.glob("*_extract.txt"))
    if not files:
        print("No extractive inputs found in", EXTRACT_DIR)
        return

    for fp in tqdm(files, desc="Documents"):
        docid = fp.stem.replace("_extract", "")
        text = fp.read_text(encoding="utf-8").strip()
        # if no extracted content, skip (write empty outputs)
        if not text:
            OUT_TEXT_DIR.joinpath(f"{docid}_pred.txt").write_text("", encoding="utf-8")
            OUT_PDF_DIR.joinpath(f"{docid}_pred.pdf").write_text("", encoding="utf-8")
            continue

        # Compose prompt
        title = extract_title_from_text(text)  # safe heuristic
        prompt = build_prompt(title, text)

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, **GEN_CFG)

        decoded = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

        # Post-process: enforce numbered bullets if model didn't
        # If text already contains newline-numbering, keep. Else convert lines to numbered bullets.
        final_text = decoded
        if not any(ch.isdigit() and '.' in decoded.splitlines()[0] for ch in decoded[:30].splitlines()):
            # naive split: by newlines, filter empties
            lines = [ln.strip() for ln in decoded.splitlines() if ln.strip()]
            if len(lines) > 1:
                numbered = []
                for i, ln in enumerate(lines, start=1):
                    # ensure it is short; if it's long, keep as-is
                    numbered.append(f"{i}. {ln}")
                final_text = "\n".join(numbered)
            else:
                # fallback: break sentences using nltk
                sents = nltk.tokenize.sent_tokenize(decoded)
                numbered = [f"{i+1}. {s}" for i, s in enumerate(sents)]
                final_text = "\n".join(numbered)

        # Ensure title at top (user requested original title to appear)
        full_output = title + "\n\n" + final_text

        # Write text file
        OUT_TEXT_DIR.joinpath(f"{docid}_pred.txt").write_text(full_output, encoding="utf-8")

        # Write PDF (try registering TTF font for unicode)
        pdf_path = OUT_PDF_DIR.joinpath(f"{docid}_pred.pdf")
        try:
            make_pdf_from_text(full_output, pdf_path, font_path=font_path)
        except Exception as e:
            # fallback: try without custom font
            try:
                make_pdf_from_text(full_output, pdf_path, font_path=None)
            except Exception as e2:
                print(f"Failed to write PDF for {docid}: {e2}")

    print("Done. Text outputs:", OUT_TEXT_DIR)
    print("PDF outputs:", OUT_PDF_DIR)


if __name__ == "__main__":
    main()
