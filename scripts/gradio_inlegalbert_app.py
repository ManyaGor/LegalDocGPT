# scripts/gradio_inlegalbert_app.py
"""
Gradio web interface for LegalDocGPT with InLegalBERT integration.
Provides an easy-to-use web interface for legal document summarization using InLegalBERT.
"""

import gradio as gr
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import textwrap
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Configuration
INLEGALBERT_MODEL = "law-ai/InLegalBERT"
SUMMARIZATION_MODEL = "google/flan-t5-small"
MAX_INPUT_TOKENS = 512
MAX_SUMMARY_TOKENS = 400
OVERLAP_TOKENS = 64

# Global variables for models
inlegalbert_tokenizer = None
inlegalbert_model = None
summarizer_tokenizer = None
summarizer_model = None
device = None

def load_models():
    """Load both InLegalBERT and summarization models."""
    global inlegalbert_tokenizer, inlegalbert_model, summarizer_tokenizer, summarizer_model, device
    
    if inlegalbert_tokenizer is None:
        print("Loading InLegalBERT model...")
        inlegalbert_tokenizer = AutoTokenizer.from_pretrained(INLEGALBERT_MODEL)
        inlegalbert_model = AutoModel.from_pretrained(INLEGALBERT_MODEL)
        
        print("Loading summarization model...")
        summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
        summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inlegalbert_model = inlegalbert_model.to(device)
        summarizer_model = summarizer_model.to(device)
        
        print("Models loaded successfully!")
    
    return inlegalbert_tokenizer, inlegalbert_model, summarizer_tokenizer, summarizer_model, device

def get_legal_embeddings(text: str, tokenizer, model, device="cpu") -> np.ndarray:
    """Get legal embeddings using InLegalBERT."""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(device)
        
        outputs = model(**inputs)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings.cpu().numpy()

def extract_legal_context(text: str, tokenizer, model, device="cpu"):
    """Extract legal context using InLegalBERT."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    if not sentences:
        return {"entities": [], "concepts": [], "key_sentences": []}
    
    # Get embeddings for sentences
    sentence_embeddings = []
    for sentence in sentences:
        emb = get_legal_embeddings(sentence, tokenizer, model, device)
        sentence_embeddings.append(emb.flatten())
    
    sentence_embeddings = np.array(sentence_embeddings)
    doc_embedding = get_legal_embeddings(text[:1000], tokenizer, model, device).flatten()
    
    # Calculate similarities
    similarities = cosine_similarity(sentence_embeddings, doc_embedding.reshape(1, -1)).flatten()
    top_indices = np.argsort(similarities)[-min(5, len(sentences)):]
    key_sentences = [sentences[i] for i in top_indices]
    
    # Extract legal entities
    entities = []
    concept_patterns = [
        r'(?:Section|Article|Clause|Rule|Regulation|Act|Code)\s+\d+[A-Za-z]*(?:\s*\([^)]+\))?',
        r'(?:Supreme Court|High Court|District Court|Court of|Tribunal)',
        r'(?:Plaintiff|Defendant|Appellant|Respondent|Petitioner|Respondent)',
        r'(?:Contract|Agreement|Deed|Will|Lease|Partnership|Company)',
        r'(?:₹|Rs\.?)\s?[\d,]+(?:\.\d{2})?',
        r'\d+\s+(?:years?|months?|days?|weeks?)',
        r'(?:Mumbai|Delhi|Bangalore|Chennai|Kolkata|Pune|Hyderabad|Ahmedabad|Jaipur|Lucknow)'
    ]
    
    for pattern in concept_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.extend(matches[:3])
    
    return {
        "entities": list(set(entities)),
        "concepts": key_sentences,
        "key_sentences": key_sentences
    }

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

def summarize_chunk_with_legal_context(chunk: str, legal_context, summarizer_tokenizer, summarizer_model):
    """Summarize a chunk with legal context from InLegalBERT."""
    legal_entities = ", ".join(legal_context.get("entities", [])[:5])
    legal_concepts = ". ".join(legal_context.get("concepts", [])[:3])
    
    prompt = f"""Summarize this legal document section in simple English. Focus on:

Legal Context (from InLegalBERT):
- Key Legal Entities: {legal_entities}
- Important Legal Concepts: {legal_concepts}

Document Section:
{chunk}

Please provide a clear summary covering:
1. Key parties and their roles
2. Important legal terms and conditions
3. Financial amounts and durations
4. Obligations and rights
5. Governing law and jurisdiction

Summary:"""
    
    inputs = summarizer_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    output_ids = summarizer_model.generate(
        **inputs, 
        max_length=MAX_SUMMARY_TOKENS, 
        num_beams=4,
        length_penalty=1.2,
        early_stopping=True, 
        no_repeat_ngram_size=3,
        temperature=0.7,
        do_sample=True
    )
    return summarizer_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_document_info(text: str, legal_context):
    """Extract document information with legal context."""
    info = {
        "title": "Legal Document",
        "date": "",
        "parties": [],
        "amounts": [],
        "durations": [],
        "legal_entities": legal_context.get("entities", [])
    }
    
    lines = text.split('\n')
    
    # Extract title
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 10 and len(line) < 100:
            if any(keyword in line.upper() for keyword in ['AGREEMENT', 'DEED', 'WILL', 'CONTRACT', 'LEASE', 'AFFIDAVIT']):
                info["title"] = line
                break
    
    # Extract date
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

def create_enhanced_summary(summaries, doc_info):
    """Create enhanced legal summary."""
    combined = " ".join(summaries)
    combined = re.sub(r'\s+', ' ', combined).strip()
    
    sentences = re.split(r'[.!?]+', combined)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
    
    output_lines = [f"{doc_info['title']} - Legal Summary (InLegalBERT Enhanced)", ""]
    
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
    
    if doc_info["legal_entities"]:
        output_lines.append("⚖️ Legal Entities:")
        for entity in doc_info["legal_entities"][:5]:
            output_lines.append(f"   • {entity}")
        output_lines.append("")
    
    output_lines.append("📋 Legal Summary:")
    
    for i, sentence in enumerate(sentences[:8], 1):
        if sentence:
            output_lines.append(f"{i}. {sentence.strip()}.")
    
    return "\n".join(output_lines)

def summarize_legal_document_with_inlegalbert(text):
    """Main summarization function with InLegalBERT."""
    if not text or len(text.strip()) < 50:
        return "Please provide a legal document with sufficient text content."
    
    try:
        # Load models
        inlegalbert_tokenizer, inlegalbert_model, summarizer_tokenizer, summarizer_model, device = load_models()
        
        # Extract legal context using InLegalBERT
        legal_context = extract_legal_context(text, inlegalbert_tokenizer, inlegalbert_model, device)
        
        # Extract document info
        doc_info = extract_document_info(text, legal_context)
        
        # Chunk the text
        chunks = chunk_by_tokens(text, summarizer_tokenizer)
        
        # Summarize each chunk with legal context
        summaries = []
        for chunk in chunks:
            summary = summarize_chunk_with_legal_context(chunk, legal_context, summarizer_tokenizer, summarizer_model)
            summaries.append(summary)
        
        if not summaries:
            return "Unable to generate summary. Please check the document content."
        
        # Create enhanced summary
        final_summary = create_enhanced_summary(summaries, doc_info)
        
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
    with gr.Blocks(title="LegalDocGPT - InLegalBERT Enhanced") as demo:
        gr.Markdown("# ⚖️ LegalDocGPT - InLegalBERT Enhanced Legal Document Summarizer")
        gr.Markdown("**Powered by InLegalBERT** - Trained specifically on Indian legal documents from 1950-2019")
        gr.Markdown("Upload or paste a legal document to get a simplified, structured summary with enhanced legal understanding.")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Legal Document Text",
                    placeholder="Paste your legal document text here...",
                    lines=15,
                    value=sample_text
                )
                
                summarize_btn = gr.Button("⚖️ Summarize with InLegalBERT", variant="primary")
                
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="InLegalBERT Enhanced Summary",
                    lines=15,
                    interactive=False
                )
        
        # Features section
        gr.Markdown("## 🚀 Enhanced Features")
        gr.Markdown("""
        - **InLegalBERT Integration**: Uses domain-specific legal embeddings trained on Indian legal documents
        - **Legal Entity Extraction**: Automatically identifies legal entities, clauses, and provisions
        - **Enhanced Understanding**: Better comprehension of legal terminology and concepts
        - **Structured Output**: Organized summaries with parties, amounts, durations, and legal entities
        """)
        
        # Event handlers
        summarize_btn.click(
            fn=summarize_legal_document_with_inlegalbert,
            inputs=input_text,
            outputs=output_text
        )
        
        # Auto-summarize on text change (optional)
        input_text.change(
            fn=summarize_legal_document_with_inlegalbert,
            inputs=input_text,
            outputs=output_text
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port to avoid conflict
        share=False,
        debug=True
    )
