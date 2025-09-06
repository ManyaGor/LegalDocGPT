# scripts/abstractive_rewrite_improved.py
"""
Improved abstractive rewrite using proper summarization models.
Uses Flan-T5-small (better for English summarization) instead of mT5.
Includes better prompt engineering and post-processing.
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
MODEL_NAME = "google/flan-t5-small"  # Better for English summarization
EXTRACT_DIR = Path("data/extractive_inputs")
OUT_TEXT_DIR = Path("data/predictions_text")
OUT_PDF_DIR = Path("data/predictions_pdf")

# Generation hyperparams optimized for summarization
GEN_CFG = dict(
    max_new_tokens=400,  # Increased for longer summaries
    num_beams=4,
    early_stopping=True,
    length_penalty=1.2,  # Slightly favor longer outputs
    no_repeat_ngram_size=3,
    do_sample=False,
    temperature=0.7,  # Add some randomness
)

# Fonts to try for PDFs (Windows common paths)
FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\DejaVuSans.ttf",
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
            pdf.add_font("UserFont", "", font_path, uni=True)
            pdf.set_font("UserFont", size=12)
        except Exception as e:
            pdf.set_font("Arial", size=12)
    else:
        pdf.set_font("Arial", size=12)
    
    # Split text into lines and write
    for line in text.splitlines():
        if line.strip():
            pdf.multi_cell(0, 7, line.strip())
    pdf.output(str(outpath))

def extract_title_from_text(text: str):
    """Extract a meaningful title from the text."""
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if len(line) > 10 and len(line) < 100:
            # Look for common legal document patterns
            if any(keyword in line.upper() for keyword in ['AGREEMENT', 'DEED', 'WILL', 'CONTRACT', 'LEASE']):
                return line
    # Fallback to first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()[:80]
    return "Legal Document"

def build_improved_prompt(title: str, sentences: str):
    """Build a better prompt for legal document summarization."""
    prompt = f"""Summarize the following legal document in simple, clear English:

Title: {title}

Document content:
{sentences}

Please provide a structured summary with numbered bullet points covering:
1. Document type and date
2. Parties involved
3. Key terms and conditions
4. Important obligations and rights
5. Duration and termination clauses
6. Governing law and jurisdiction

Summary:"""
    return prompt

def clean_generated_text(text: str):
    """Clean and format the generated text."""
    # Remove any remaining special tokens
    text = text.replace("<extra_id_0>", "").replace("<extra_id_1>", "")
    
    # Split into lines and clean
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    if not lines:
        return "No summary generated."
    
    # If the text doesn't start with numbers, try to format it
    if not lines[0].startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
        # Try to split by sentences and number them
        import re
        sentences = []
        for line in lines:
            # Split by periods but keep the period
            parts = re.split(r'(?<=\.)\s+', line)
            sentences.extend([p.strip() for p in parts if p.strip()])
        
        # Number the sentences
        numbered_lines = []
        for i, sentence in enumerate(sentences, 1):
            if sentence and not sentence.endswith('.'):
                sentence += '.'
            numbered_lines.append(f"{i}. {sentence}")
        
        return "\n".join(numbered_lines)
    
    return "\n".join(lines)

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
        print("No TTF font found; PDFs may not render special characters properly.")

    files = sorted(EXTRACT_DIR.glob("*_extract.txt"))
    if not files:
        print("No extractive inputs found in", EXTRACT_DIR)
        return

    print(f"Processing {len(files)} documents...")
    
    for fp in tqdm(files, desc="Documents"):
        docid = fp.stem.replace("_extract", "")
        text = fp.read_text(encoding="utf-8").strip()
        
        if not text:
            OUT_TEXT_DIR.joinpath(f"{docid}_pred.txt").write_text("No content extracted.", encoding="utf-8")
            continue

        try:
            # Extract title and build prompt
            title = extract_title_from_text(text)
            prompt = build_improved_prompt(title, text)

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            
            with torch.no_grad():
                out = model.generate(**inputs, **GEN_CFG)

            decoded = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            
            # Clean and format the output
            final_text = clean_generated_text(decoded)
            
            # Ensure title at top
            full_output = f"{title}\n\n{final_text}"

            # Write text file
            OUT_TEXT_DIR.joinpath(f"{docid}_pred.txt").write_text(full_output, encoding="utf-8")

            # Write PDF
            pdf_path = OUT_PDF_DIR.joinpath(f"{docid}_pred.pdf")
            try:
                make_pdf_from_text(full_output, pdf_path, font_path=font_path)
            except Exception as e:
                print(f"Failed to write PDF for {docid}: {e}")

        except Exception as e:
            print(f"Error processing {docid}: {e}")
            OUT_TEXT_DIR.joinpath(f"{docid}_pred.txt").write_text(f"Error processing document: {str(e)}", encoding="utf-8")

    print("Done. Text outputs:", OUT_TEXT_DIR)
    print("PDF outputs:", OUT_PDF_DIR)

if __name__ == "__main__":
    main()
