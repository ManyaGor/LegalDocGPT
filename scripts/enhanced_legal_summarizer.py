# scripts/enhanced_legal_summarizer.py
"""
Enhanced legal document summarizer with improved post-processing.
Combines the best of direct summarization with structured formatting.
"""

import os
import json
import re
import textwrap
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Configuration
MODEL_NAME = "google/flan-t5-small"
DATASET_PATH = Path("dataset/dataset.jsonl")
OUT_TEXT_DIR = Path("data/predictions_text")
OUT_PDF_DIR = Path("data/predictions_pdf")

MAX_INPUT_TOKENS = 512
MAX_SUMMARY_TOKENS = 400
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
    """Summarize a single chunk with improved prompting."""
    prompt = f"""Summarize this legal document section in simple English. Focus on:
1. Key parties involved
2. Important dates and amounts
3. Main obligations and rights
4. Duration and termination terms
5. Governing law

Document section:
{chunk}

Summary:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    output_ids = model.generate(
        **inputs, 
        max_length=MAX_SUMMARY_TOKENS, 
        num_beams=4,
        length_penalty=1.2,
        early_stopping=True, 
        no_repeat_ngram_size=3,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_document_info(text: str) -> Dict:
    """Extract comprehensive document information."""
    info = {
        "title": "Legal Document",
        "date": "",
        "parties": [],
        "type": "Agreement",
        "amounts": [],
        "durations": [],
        "locations": []
    }
    
    lines = text.split('\n')
    
    # Extract title
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 10 and len(line) < 100:
            if any(keyword in line.upper() for keyword in ['AGREEMENT', 'DEED', 'WILL', 'CONTRACT', 'LEASE', 'AFFIDAVIT']):
                info["title"] = line
                break
    
    # Extract date patterns
    date_patterns = [
        r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            info["date"] = matches[0]
            break
    
    # Extract parties
    party_patterns = [
        r'([A-Z][a-zA-Z\s&.,]+(?:Pvt\.?\s+Ltd\.?|LLP|Inc\.?|Corp\.?|Technologies|Analytics|Properties|Ventures))',
        r'(Mr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'(Ms\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'(Mrs\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)'
    ]
    
    for pattern in party_patterns:
        matches = re.findall(pattern, text)
        info["parties"].extend(matches[:3])  # Limit to first 3 parties
    
    # Extract amounts
    amount_patterns = [
        r'(₹\s?[\d,]+(?:\.\d{2})?)',
        r'(Rs\.?\s?[\d,]+(?:\.\d{2})?)',
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:Rupees?|Lakh|Crore))'
    ]
    
    for pattern in amount_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info["amounts"].extend(matches[:5])  # Limit to first 5 amounts
    
    # Extract durations
    duration_patterns = [
        r'(\d+\s+(?:years?|months?|days?|weeks?))',
        r'(\d+\s+(?:year|month|day|week)\s+(?:period|term|duration))'
    ]
    
    for pattern in duration_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info["durations"].extend(matches[:3])  # Limit to first 3 durations
    
    # Extract locations
    location_patterns = [
        r'([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*,\s*(?:Mumbai|Delhi|Bangalore|Chennai|Kolkata|Pune|Hyderabad|Ahmedabad|Jaipur|Lucknow))',
        r'([A-Z][a-z]+\s+(?:Road|Street|Avenue|Lane|Garden|Complex|Tower|Plaza|Mall))'
    ]
    
    for pattern in location_patterns:
        matches = re.findall(pattern, text)
        info["locations"].extend(matches[:3])  # Limit to first 3 locations
    
    return info

def clean_and_structure_summary(summaries: List[str], doc_info: Dict) -> str:
    """Clean and structure the final summary."""
    # Combine all summaries
    combined = " ".join(summaries)
    
    # Clean up the text
    combined = re.sub(r'\s+', ' ', combined).strip()
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', combined)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
    
    # Create structured output
    output_lines = [f"{doc_info['title']} - Simplified Summary", ""]
    
    # Add document metadata
    if doc_info["date"]:
        output_lines.append(f"📅 Date: {doc_info['date']}")
    
    if doc_info["parties"]:
        output_lines.append("👥 Parties:")
        for party in doc_info["parties"][:3]:
            output_lines.append(f"   • {party}")
        output_lines.append("")
    
    if doc_info["amounts"]:
        output_lines.append("💰 Key Amounts:")
        for amount in doc_info["amounts"][:3]:
            output_lines.append(f"   • {amount}")
        output_lines.append("")
    
    if doc_info["durations"]:
        output_lines.append("⏰ Duration:")
        for duration in doc_info["durations"][:2]:
            output_lines.append(f"   • {duration}")
        output_lines.append("")
    
    # Add structured summary points
    output_lines.append("📋 Key Points:")
    
    # Categorize sentences by content
    categorized = {
        "parties": [],
        "obligations": [],
        "terms": [],
        "other": []
    }
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in ["party", "parties", "between", "agreement"]):
            categorized["parties"].append(sentence)
        elif any(word in sentence_lower for word in ["obligation", "duty", "responsibility", "must", "shall", "required"]):
            categorized["obligations"].append(sentence)
        elif any(word in sentence_lower for word in ["term", "condition", "duration", "period", "valid"]):
            categorized["terms"].append(sentence)
        else:
            categorized["other"].append(sentence)
    
    # Add points in order of importance
    point_num = 1
    
    # Add party information first
    for sentence in categorized["parties"][:2]:
        if sentence:
            output_lines.append(f"{point_num}. {sentence.strip()}.")
            point_num += 1
    
    # Add obligations
    for sentence in categorized["obligations"][:4]:
        if sentence:
            output_lines.append(f"{point_num}. {sentence.strip()}.")
            point_num += 1
    
    # Add terms and conditions
    for sentence in categorized["terms"][:3]:
        if sentence:
            output_lines.append(f"{point_num}. {sentence.strip()}.")
            point_num += 1
    
    # Add other important points
    for sentence in categorized["other"][:3]:
        if sentence:
            output_lines.append(f"{point_num}. {sentence.strip()}.")
            point_num += 1
    
    return "\n".join(output_lines)

def write_enhanced_pdf(text: str, output_path: Path):
    """Write enhanced formatted text to PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    lines = text.split('\n')
    for line in lines:
        if line.strip():
            if line.startswith(("📅", "👥", "💰", "⏰", "📋")):
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, sanitize(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=11)
            elif line.startswith(("   •", "•")):
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
    """Process a single document with enhanced formatting."""
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
    
    # Create enhanced structured summary
    final_summary = clean_and_structure_summary(summaries, doc_info)
    
    # Write outputs
    text_path = OUT_TEXT_DIR / f"{doc_id}_pred.txt"
    text_path.write_text(final_summary, encoding="utf-8")
    
    pdf_path = OUT_PDF_DIR / f"{doc_id}_pred.pdf"
    try:
        write_enhanced_pdf(final_summary, pdf_path)
        print(f"  ✓ Generated enhanced: {text_path.name} and {pdf_path.name}")
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
    
    print(f"Processing {len(records)} documents with enhanced formatting...")
    
    for record in records:
        doc_id = record.get("id", "unknown")
        text = record.get("input", "")
        
        if text and len(text.strip()) > 50:
            process_document(doc_id, text, tokenizer, model)
        else:
            print(f"Skipping {doc_id}: insufficient text")
    
    print("Enhanced processing complete!")

if __name__ == "__main__":
    main()
