# scripts/inlegalbert_enhanced_summarizer.py
"""
Enhanced legal document summarizer using InLegalBERT for better legal understanding.
Integrates InLegalBERT embeddings with Flan-T5 for superior legal document processing.
"""

import os
import json
import re
import textwrap
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
INLEGALBERT_MODEL = "law-ai/InLegalBERT"
SUMMARIZATION_MODEL = "google/flan-t5-small"
DATASET_PATH = Path("dataset/dataset.jsonl")
OUT_TEXT_DIR = Path("data/predictions_text")
OUT_PDF_DIR = Path("data/predictions_pdf")

MAX_INPUT_TOKENS = 512
MAX_SUMMARY_TOKENS = 400
OVERLAP_TOKENS = 64
BATCH_SIZE = 8

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

def load_models():
    """Load both InLegalBERT and summarization models."""
    print("Loading InLegalBERT model...")
    inlegalbert_tokenizer = AutoTokenizer.from_pretrained(INLEGALBERT_MODEL)
    inlegalbert_model = AutoModel.from_pretrained(INLEGALBERT_MODEL)
    
    print("Loading summarization model...")
    summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL)
    
    return inlegalbert_tokenizer, inlegalbert_model, summarizer_tokenizer, summarizer_model

def get_legal_embeddings(text: str, tokenizer, model, device="cpu") -> np.ndarray:
    """Get legal embeddings using InLegalBERT."""
    model.eval()
    with torch.no_grad():
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(device)
        
        # Get embeddings
        outputs = model(**inputs)
        
        # Use mean pooling of last hidden states
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings.cpu().numpy()

def extract_legal_entities_and_concepts(text: str, tokenizer, model, device="cpu") -> Dict:
    """Extract legal entities and concepts using InLegalBERT."""
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    if not sentences:
        return {"entities": [], "concepts": [], "key_sentences": []}
    
    # Get embeddings for all sentences
    sentence_embeddings = []
    for sentence in sentences:
        emb = get_legal_embeddings(sentence, tokenizer, model, device)
        sentence_embeddings.append(emb.flatten())
    
    sentence_embeddings = np.array(sentence_embeddings)
    
    # Get document-level embedding
    doc_embedding = get_legal_embeddings(text[:1000], tokenizer, model, device).flatten()
    
    # Calculate similarity scores
    similarities = cosine_similarity(sentence_embeddings, doc_embedding.reshape(1, -1)).flatten()
    
    # Select top sentences based on legal relevance
    top_indices = np.argsort(similarities)[-min(8, len(sentences)):]
    key_sentences = [sentences[i] for i in top_indices]
    
    # Extract legal entities using patterns
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
        entities.extend(matches[:5])  # Limit to avoid duplicates
    
    return {
        "entities": list(set(entities)),
        "concepts": key_sentences,
        "key_sentences": key_sentences
    }

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

def summarize_chunk_with_legal_context(chunk: str, legal_context: Dict, summarizer_tokenizer, summarizer_model) -> str:
    """Summarize a chunk with legal context from InLegalBERT."""
    
    # Build enhanced prompt with legal context
    legal_entities = ", ".join(legal_context.get("entities", [])[:5])
    legal_concepts = ". ".join(legal_context.get("concepts", [])[:3])
    
    prompt = f"""Summarize this legal document section in simple English. Focus on:

Legal Context:
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

def extract_document_info_enhanced(text: str, legal_context: Dict) -> Dict:
    """Extract comprehensive document information with legal context."""
    info = {
        "title": "Legal Document",
        "date": "",
        "parties": [],
        "type": "Agreement",
        "amounts": [],
        "durations": [],
        "locations": [],
        "legal_entities": legal_context.get("entities", []),
        "legal_concepts": legal_context.get("concepts", [])
    }
    
    lines = text.split('\n')
    
    # Extract title with legal context
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 10 and len(line) < 100:
            if any(keyword in line.upper() for keyword in ['AGREEMENT', 'DEED', 'WILL', 'CONTRACT', 'LEASE', 'AFFIDAVIT']):
                info["title"] = line
                break
    
    # Enhanced date extraction
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
    
    # Enhanced party extraction using legal patterns
    party_patterns = [
        r'([A-Z][a-zA-Z\s&.,]+(?:Pvt\.?\s+Ltd\.?|LLP|Inc\.?|Corp\.?|Technologies|Analytics|Properties|Ventures|Company|Partnership))',
        r'(Mr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'(Ms\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'(Mrs\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'(Dr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)'
    ]
    
    for pattern in party_patterns:
        matches = re.findall(pattern, text)
        info["parties"].extend(matches[:3])
    
    # Enhanced amount extraction
    amount_patterns = [
        r'(₹\s?[\d,]+(?:\.\d{2})?)',
        r'(Rs\.?\s?[\d,]+(?:\.\d{2})?)',
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:Rupees?|Lakh|Crore|Thousand))',
        r'(\d+(?:\.\d+)?\s*(?:Lakh|Crore|Thousand|Million|Billion))'
    ]
    
    for pattern in amount_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info["amounts"].extend(matches[:5])
    
    # Enhanced duration extraction
    duration_patterns = [
        r'(\d+\s+(?:years?|months?|days?|weeks?))',
        r'(\d+\s+(?:year|month|day|week)\s+(?:period|term|duration))',
        r'(?:for\s+a\s+period\s+of\s+)(\d+\s+(?:years?|months?|days?))'
    ]
    
    for pattern in duration_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info["durations"].extend(matches[:3])
    
    # Enhanced location extraction
    location_patterns = [
        r'([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*,\s*(?:Mumbai|Delhi|Bangalore|Chennai|Kolkata|Pune|Hyderabad|Ahmedabad|Jaipur|Lucknow|India))',
        r'([A-Z][a-z]+\s+(?:Road|Street|Avenue|Lane|Garden|Complex|Tower|Plaza|Mall|Building))',
        r'((?:District|State|Union Territory)\s+of\s+[A-Z][a-z]+)'
    ]
    
    for pattern in location_patterns:
        matches = re.findall(pattern, text)
        info["locations"].extend(matches[:3])
    
    return info

def create_enhanced_legal_summary(summaries: List[str], doc_info: Dict) -> str:
    """Create enhanced legal summary with InLegalBERT insights."""
    combined = " ".join(summaries)
    combined = re.sub(r'\s+', ' ', combined).strip()
    
    sentences = re.split(r'[.!?]+', combined)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
    
    output_lines = [f"{doc_info['title']} - Legal Summary (InLegalBERT Enhanced)", ""]
    
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
    
    if doc_info["legal_entities"]:
        output_lines.append("⚖️ Legal Entities:")
        for entity in doc_info["legal_entities"][:5]:
            output_lines.append(f"   • {entity}")
        output_lines.append("")
    
    # Add structured summary points
    output_lines.append("📋 Legal Summary:")
    
    # Categorize sentences by legal content
    categorized = {
        "parties": [],
        "obligations": [],
        "terms": [],
        "legal_provisions": [],
        "other": []
    }
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in ["party", "parties", "between", "agreement", "contract"]):
            categorized["parties"].append(sentence)
        elif any(word in sentence_lower for word in ["obligation", "duty", "responsibility", "must", "shall", "required", "liable"]):
            categorized["obligations"].append(sentence)
        elif any(word in sentence_lower for word in ["term", "condition", "duration", "period", "valid", "expire"]):
            categorized["terms"].append(sentence)
        elif any(word in sentence_lower for word in ["section", "article", "clause", "act", "law", "statute", "regulation"]):
            categorized["legal_provisions"].append(sentence)
        else:
            categorized["other"].append(sentence)
    
    # Add points in order of legal importance
    point_num = 1
    
    # Add legal provisions first (most important)
    for sentence in categorized["legal_provisions"][:2]:
        if sentence:
            output_lines.append(f"{point_num}. {sentence.strip()}.")
            point_num += 1
    
    # Add party information
    for sentence in categorized["parties"][:2]:
        if sentence:
            output_lines.append(f"{point_num}. {sentence.strip()}.")
            point_num += 1
    
    # Add obligations
    for sentence in categorized["obligations"][:3]:
        if sentence:
            output_lines.append(f"{point_num}. {sentence.strip()}.")
            point_num += 1
    
    # Add terms and conditions
    for sentence in categorized["terms"][:2]:
        if sentence:
            output_lines.append(f"{point_num}. {sentence.strip()}.")
            point_num += 1
    
    # Add other important points
    for sentence in categorized["other"][:2]:
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
            if line.startswith(("📅", "👥", "💰", "⏰", "⚖️", "📋")):
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

def process_document_with_inlegalbert(doc_id: str, text: str, inlegalbert_tokenizer, inlegalbert_model, summarizer_tokenizer, summarizer_model, device="cpu"):
    """Process a single document using InLegalBERT enhanced approach."""
    print(f"Processing {doc_id} with InLegalBERT...")
    
    try:
        # Extract legal context using InLegalBERT
        legal_context = extract_legal_entities_and_concepts(text, inlegalbert_tokenizer, inlegalbert_model, device)
        
        # Extract document information with legal context
        doc_info = extract_document_info_enhanced(text, legal_context)
        
        # Chunk the text
        chunks = chunk_by_tokens(text, summarizer_tokenizer)
        
        # Summarize each chunk with legal context
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = summarize_chunk_with_legal_context(chunk, legal_context, summarizer_tokenizer, summarizer_model)
                summaries.append(summary)
            except Exception as e:
                print(f"  Error summarizing chunk {i+1}: {e}")
        
        if not summaries:
            print(f"  No summaries generated for {doc_id}")
            return
        
        # Create enhanced legal summary
        final_summary = create_enhanced_legal_summary(summaries, doc_info)
        
        # Write outputs
        text_path = OUT_TEXT_DIR / f"{doc_id}_pred.txt"
        text_path.write_text(final_summary, encoding="utf-8")
        
        pdf_path = OUT_PDF_DIR / f"{doc_id}_pred.pdf"
        try:
            write_enhanced_pdf(final_summary, pdf_path)
            print(f"  ✓ Generated InLegalBERT enhanced: {text_path.name} and {pdf_path.name}")
        except Exception as e:
            print(f"  ✗ PDF generation failed: {e}")
            
    except Exception as e:
        print(f"  Error processing {doc_id}: {e}")

def main():
    """Main processing function with InLegalBERT integration."""
    print("Loading InLegalBERT and summarization models...")
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    inlegalbert_tokenizer, inlegalbert_model, summarizer_tokenizer, summarizer_model = load_models()
    
    # Move models to device
    inlegalbert_model = inlegalbert_model.to(device)
    summarizer_model = summarizer_model.to(device)
    
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
    
    print(f"Processing {len(records)} documents with InLegalBERT enhancement...")
    
    for record in records:
        doc_id = record.get("id", "unknown")
        text = record.get("input", "")
        
        if text and len(text.strip()) > 50:
            process_document_with_inlegalbert(
                doc_id, text, 
                inlegalbert_tokenizer, inlegalbert_model,
                summarizer_tokenizer, summarizer_model,
                device
            )
        else:
            print(f"Skipping {doc_id}: insufficient text")
    
    print("InLegalBERT enhanced processing complete!")

if __name__ == "__main__":
    main()
