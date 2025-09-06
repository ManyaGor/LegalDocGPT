# scripts/direct_summarization.py
"""
Direct summarization approach using the existing API server logic.
This bypasses the extractive step and directly summarizes the full documents.
"""

import os
import json
import re
import textwrap
from pathlib import Path
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Configuration
MODEL_NAME = "google/flan-t5-small"
DATASET_PATH = Path("dataset/dataset.jsonl")
OUT_TEXT_DIR = Path("data/predictions_text")
OUT_PDF_DIR = Path("data/predictions_pdf")

MAX_INPUT_TOKENS = 512
MAX_SUMMARY_TOKENS = 300
OVERLAP_TOKENS = 64

def sanitize(s: str) -> str:
    """Make text safe for PDF fonts."""
    repl = {
        "•": "-", "₹": "Rs.", "–": "-", "—": "-",
        "'": "'", "'": "'", """: '"', """: '"', "\u00A0": " "
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out.encode("latin-1", "ignore").decode("latin-1")

def chunk_by_tokens(text: str, tokenizer, max_len=MAX_INPUT_TOKENS, overlap=OVERLAP_TOKENS) -> List[str]:
    """Split text into overlapping chunks."""
    ids = tokenizer.encode(text, truncation=False)
    chunks, start, stride = [], 0, max_len - overlap
    while start < len(ids):
        end = min(start + max_len, len(ids))
        chunks.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if end == len(ids): 
            break
        start += stride
    return chunks

def summarize_chunk(chunk: str, tokenizer, model) -> str:
    """Summarize a single chunk."""
    prompt = f"Summarize the following legal document in simple English:\n\n{chunk}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    output_ids = model.generate(
        **inputs, 
        max_length=MAX_SUMMARY_TOKENS, 
        num_beams=4,
        length_penalty=1.2,
        early_stopping=True, 
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_document_info(text: str) -> dict:
    """Extract key information from legal document."""
    info = {
        "title": "Legal Document",
        "date": "",
        "parties": [],
        "type": "Agreement"
    }
    
    lines = text.split('\n')
    
    # Extract title (first meaningful line)
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 10 and len(line) < 100:
            if any(keyword in line.upper() for keyword in ['AGREEMENT', 'DEED', 'WILL', 'CONTRACT', 'LEASE']):
                info["title"] = line
                break
    
    # Extract date
    date_patterns = [
        r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'(\d{4}-\d{2}-\d{2})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            info["date"] = match.group(1)
            break
    
    # Extract parties (look for BETWEEN, AND patterns)
    between_match = re.search(r'BETWEEN[:\s]*(.*?)(?:AND|WHEREAS)', text, re.DOTALL | re.IGNORECASE)
    if between_match:
        parties_text = between_match.group(1)
        # Extract company/person names
        party_patterns = [
            r'([A-Z][a-zA-Z\s&.,]+(?:Pvt\.?\s+Ltd\.?|LLP|Inc\.?|Corp\.?))',
            r'(Mr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'(Ms\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        
        for pattern in party_patterns:
            matches = re.findall(pattern, parties_text)
            info["parties"].extend(matches)
    
    return info

def format_summary(summaries: List[str], doc_info: dict) -> str:
    """Format the final summary with document information."""
    # Combine all summaries
    combined = " ".join(summaries)
    
    # Clean up the text
    combined = re.sub(r'\s+', ' ', combined).strip()
    
    # Split into sentences and create bullet points
    sentences = re.split(r'[.!?]+', combined)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # Create formatted output
    output_lines = [f"{doc_info['title']} - Simplified Summary", ""]
    
    if doc_info["date"]:
        output_lines.append(f"Date: {doc_info['date']}")
    
    if doc_info["parties"]:
        output_lines.append("Parties:")
        for party in doc_info["parties"][:3]:  # Limit to first 3 parties
            output_lines.append(f"  • {party}")
        output_lines.append("")
    
    # Add numbered summary points
    output_lines.append("Key Points:")
    for i, sentence in enumerate(sentences[:12], 1):  # Limit to 12 points
        if sentence:
            output_lines.append(f"{i}. {sentence.strip()}.")
    
    return "\n".join(output_lines)

def write_pdf_output(text: str, output_path: Path):
    """Write formatted text to PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    
    lines = text.split('\n')
    for line in lines:
        if line.strip():
            if line.startswith(("Date:", "Parties:", "Key Points:")):
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, sanitize(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=11)
            elif line.startswith(("  •", "•")):
                pdf.set_font("Helvetica", size=11)
                pdf.cell(5)
                pdf.cell(0, 6, sanitize(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            elif re.match(r'^\d+\.', line):
                pdf.set_font("Helvetica", size=11)
                wrapped = textwrap.wrap(sanitize(line), width=90)
                if wrapped:
                    pdf.cell(5)
                    pdf.cell(0, 6, wrapped[0], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    for cont in wrapped[1:]:
                        pdf.cell(10)
                        pdf.cell(0, 6, cont, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 8, sanitize(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=11)
        pdf.ln(2)
    
    pdf.output(str(output_path))

def process_document(doc_id: str, text: str, tokenizer, model):
    """Process a single document."""
    print(f"Processing {doc_id}...")
    
    # Extract document information
    doc_info = extract_document_info(text)
    
    # Chunk the text
    chunks = chunk_by_tokens(text, tokenizer)
    
    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            summary = summarize_chunk(chunk, tokenizer, model)
            summaries.append(summary)
        except Exception as e:
            print(f"  Error summarizing chunk {i+1}: {e}")
    
    if not summaries:
        print(f"  No summaries generated for {doc_id}")
        return
    
    # Format the final summary
    final_summary = format_summary(summaries, doc_info)
    
    # Write outputs
    text_path = OUT_TEXT_DIR / f"{doc_id}_pred.txt"
    text_path.write_text(final_summary, encoding="utf-8")
    
    pdf_path = OUT_PDF_DIR / f"{doc_id}_pred.pdf"
    try:
        write_pdf_output(final_summary, pdf_path)
        print(f"  ✓ Generated: {text_path.name} and {pdf_path.name}")
    except Exception as e:
        print(f"  ✗ PDF generation failed: {e}")

def main():
    """Main processing function."""
    print("Loading model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Create output directories
    OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return
    
    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Processing {len(records)} documents...")
    
    for record in records:
        doc_id = record.get("id", "unknown")
        text = record.get("input", "")
        
        if text and len(text.strip()) > 50:
            process_document(doc_id, text, tokenizer, model)
        else:
            print(f"Skipping {doc_id}: insufficient text")
    
    print("Done!")

if __name__ == "__main__":
    main()
