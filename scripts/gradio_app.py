# scripts/gradio_app.py
"""
Gradio web interface for LegalDocGPT.
Provides an easy-to-use web interface for legal document summarization.
"""

import gradio as gr
import os
import tempfile
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import textwrap
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Configuration
MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_TOKENS = 512
MAX_SUMMARY_TOKENS = 400
OVERLAP_TOKENS = 64

# Global variables for model
tokenizer = None
model = None

def load_model():
    """Load the summarization model."""
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print("Model loaded successfully!")
    return tokenizer, model

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

def chunk_by_tokens(text: str, tokenizer, max_len=MAX_INPUT_TOKENS, overlap=OVERLAP_TOKENS):
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

def extract_document_info(text: str):
    """Extract comprehensive document information."""
    info = {
        "title": "Legal Document",
        "date": "",
        "parties": [],
        "amounts": [],
        "durations": []
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
        r'(\d{4}-\d{2}-\d{2})'
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
        r'(Ms\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)'
    ]
    
    for pattern in party_patterns:
        matches = re.findall(pattern, text)
        info["parties"].extend(matches[:3])
    
    # Extract amounts
    amount_patterns = [
        r'(₹\s?[\d,]+(?:\.\d{2})?)',
        r'(Rs\.?\s?[\d,]+(?:\.\d{2})?)',
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:Rupees?|Lakh|Crore))'
    ]
    
    for pattern in amount_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info["amounts"].extend(matches[:3])
    
    # Extract durations
    duration_patterns = [
        r'(\d+\s+(?:years?|months?|days?|weeks?))',
        r'(\d+\s+(?:year|month|day|week)\s+(?:period|term|duration))'
    ]
    
    for pattern in duration_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info["durations"].extend(matches[:2])
    
    return info

def create_structured_summary(summaries, doc_info):
    """Create a structured summary."""
    combined = " ".join(summaries)
    combined = re.sub(r'\s+', ' ', combined).strip()
    
    sentences = re.split(r'[.!?]+', combined)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
    
    output_lines = [f"{doc_info['title']} - Simplified Summary", ""]
    
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
    
    output_lines.append("📋 Key Points:")
    
    for i, sentence in enumerate(sentences[:8], 1):
        if sentence:
            output_lines.append(f"{i}. {sentence.strip()}.")
    
    return "\n".join(output_lines)

def summarize_legal_document(text):
    """Main summarization function."""
    if not text or len(text.strip()) < 50:
        return "Please provide a legal document with sufficient text content."
    
    try:
        # Load model
        tokenizer, model = load_model()
        
        # Extract document info
        doc_info = extract_document_info(text)
        
        # Chunk the text
        chunks = chunk_by_tokens(text, tokenizer)
        
        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            summary = summarize_chunk(chunk, tokenizer, model)
            summaries.append(summary)
        
        if not summaries:
            return "Unable to generate summary. Please check the document content."
        
        # Create structured summary
        final_summary = create_structured_summary(summaries, doc_info)
        
        return final_summary
        
    except Exception as e:
        return f"Error processing document: {str(e)}"

def create_demo():
    """Create the Gradio demo interface."""
    
    # Sample legal document text
    sample_text = """MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement (the "Agreement") is entered into on this 22nd day of August, 2025 (the "Effective Date").

BETWEEN:
InnovateNext Technologies Pvt. Ltd., a company incorporated under the Companies Act, 2013, having its registered office at 7th Floor, Tech Park One, Powai, Mumbai, Maharashtra 400076, India (hereinafter referred to as "InnovateNext") of the FIRST PART;

AND
DataWise Analytics LLP, a Limited Liability Partnership registered under the Limited Liability Partnership Act, 2008, having its principal place of business at A-wing, Office No. 502, Corporate Avenue, Malad (East), Mumbai, Maharashtra 400097, India (hereinafter referred to as "DataWise") of the SECOND PART.

WHEREAS:
A. The Parties are interested in exploring a potential strategic partnership to develop and market an AI-driven predictive analytics platform (the "Purpose").
B. In the course of discussions concerning the Purpose, each Party may disclose to the other Party certain non-public, confidential, or proprietary information.
C. The Parties have agreed to enter into this Agreement to ensure the protection of such confidential information.

NOW, THEREFORE, THE PARTIES AGREE AS FOLLOWS:

1. Definition of Confidential Information
"Confidential Information" shall mean any and all information disclosed by one Party to the other Party, including but not limited to business plans, financial information, customer lists, technical data, trade secrets, and marketing strategies.

2. Obligations of the Receiving Party
The Receiving Party shall:
a. Hold and maintain the Confidential Information in strict confidence
b. Not disclose any Confidential Information to any third party without prior written consent
c. Use the Confidential Information solely for the Purpose

3. Term and Termination
This Agreement shall be effective for a period of three (3) years from the Effective Date.

4. Governing Law
This Agreement shall be governed by the laws of India."""

    # Create Gradio interface
    with gr.Blocks(title="LegalDocGPT - Legal Document Summarizer") as demo:
        gr.Markdown("# 📋 LegalDocGPT - Legal Document Summarizer")
        gr.Markdown("Upload or paste a legal document to get a simplified, structured summary in plain English.")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Legal Document Text",
                    placeholder="Paste your legal document text here...",
                    lines=15,
                    value=sample_text
                )
                
                summarize_btn = gr.Button("📝 Summarize Document", variant="primary")
                
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Simplified Summary",
                    lines=15,
                    interactive=False
                )
        
        # Example section
        gr.Markdown("## 📖 Example")
        gr.Markdown("The interface will automatically extract key information like parties, dates, amounts, and create a structured summary with numbered points.")
        
        # Event handlers
        summarize_btn.click(
            fn=summarize_legal_document,
            inputs=input_text,
            outputs=output_text
        )
        
        # Auto-summarize on text change (optional)
        input_text.change(
            fn=summarize_legal_document,
            inputs=input_text,
            outputs=output_text
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
