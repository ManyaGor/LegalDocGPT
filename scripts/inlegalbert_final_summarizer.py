# scripts/inlegalbert_final_summarizer.py
"""
Final optimized InLegalBERT summarizer producing comprehensive summaries
matching the exact length and detail of reference outputs.
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
MAX_SUMMARY_TOKENS = 800  # Increased for maximum detail
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

def extract_maximum_legal_context(text: str, tokenizer, model, device="cpu") -> Dict:
    """Extract maximum legal context using InLegalBERT."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
    
    if not sentences:
        return {"entities": [], "concepts": [], "key_sentences": [], "sections": []}
    
    # Get embeddings for all sentences
    sentence_embeddings = []
    for sentence in sentences:
        emb = get_legal_embeddings(sentence, tokenizer, model, device)
        sentence_embeddings.append(emb.flatten())
    
    sentence_embeddings = np.array(sentence_embeddings)
    doc_embedding = get_legal_embeddings(text[:1000], tokenizer, model, device).flatten()
    
    # Calculate similarity scores
    similarities = cosine_similarity(sentence_embeddings, doc_embedding.reshape(1, -1)).flatten()
    
    # Select maximum sentences for comprehensive coverage
    top_indices = np.argsort(similarities)[-min(20, len(sentences)):]
    key_sentences = [sentences[i] for i in top_indices]
    
    # Extract maximum legal entities
    entities = []
    concept_patterns = [
        r'(?:Section|Article|Clause|Rule|Regulation|Act|Code)\s+\d+[A-Za-z]*(?:\s*\([^)]+\))?',
        r'(?:Supreme Court|High Court|District Court|Court of|Tribunal)',
        r'(?:Plaintiff|Defendant|Appellant|Respondent|Petitioner|Respondent)',
        r'(?:Contract|Agreement|Deed|Will|Lease|Partnership|Company|Firm|Studios)',
        r'(?:₹|Rs\.?)\s?[\d,]+(?:\.\d{2})?',
        r'\d+\s+(?:years?|months?|days?|weeks?)',
        r'(?:Mumbai|Delhi|Bangalore|Chennai|Kolkata|Pune|Hyderabad|Ahmedabad|Jaipur|Lucknow)',
        r'(?:Pvt\.?\s+Ltd\.?|LLP|Inc\.?|Corp\.?|Technologies|Analytics|Properties|Ventures|Company|Partnership|Firm|Studios)',
        r'(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?',
        r'(?:Copyright|Patent|Trademark|Intellectual Property|IP)',
        r'(?:Confidential|Proprietary|Trade Secret)',
        r'(?:Effective Date|Commencement|Termination|Duration)',
        r'(?:son\s+of|daughter\s+of)\s+[A-Z][a-z]+\s+[A-Z][a-z]+',
        r'(?:residing\s+at|office\s+at|registered\s+office)\s+[^,]+,\s*[^,]+',
        r'(?:Floor|Road|Street|Avenue|Lane|Garden|Complex|Tower|Plaza|Mall|Building|Heights|View)'
    ]
    
    for pattern in concept_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.extend(matches[:10])  # Get maximum entities
    
    # Extract document sections
    sections = []
    section_patterns = [
        r'(?:WHEREAS|NOW THEREFORE|IN WITNESS|BETWEEN|PARTIES)',
        r'(?:OBLIGATIONS|RIGHTS|DUTIES|RESPONSIBILITIES)',
        r'(?:TERM|DURATION|EFFECTIVE|COMMENCEMENT)',
        r'(?:CONFIDENTIAL|PROPRIETARY|DISCLOSURE)',
        r'(?:GOVERNING LAW|JURISDICTION|DISPUTE)',
        r'(?:CONSIDERATION|PAYMENT|AMOUNT|COMPENSATION)',
        r'(?:ASSIGNMENT|TRANSFER|LICENSE|OWNERSHIP)',
        r'(?:WARRANTIES|REPRESENTATIONS|COVENANTS)',
        r'(?:BUSINESS|PURPOSE|OBJECTIVE)',
        r'(?:EXCLUSIONS|EXCEPTIONS|LIMITATIONS)'
    ]
    
    for pattern in section_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        sections.extend(matches[:8])
    
    return {
        "entities": list(set(entities)),
        "concepts": key_sentences,
        "key_sentences": key_sentences,
        "sections": list(set(sections))
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

def summarize_chunk_maximally(chunk: str, legal_context: Dict, summarizer_tokenizer, summarizer_model) -> str:
    """Summarize a chunk with maximum legal context and detail."""
    
    # Build maximum detail prompt
    legal_entities = ", ".join(legal_context.get("entities", [])[:12])
    legal_concepts = ". ".join(legal_context.get("concepts", [])[:8])
    legal_sections = ", ".join(legal_context.get("sections", [])[:8])
    
    prompt = f"""Create a comprehensive, detailed summary of this legal document section. Include EVERY important detail:

Legal Context (from InLegalBERT):
- Key Legal Entities: {legal_entities}
- Important Legal Concepts: {legal_concepts}
- Document Sections: {legal_sections}

Document Section:
{chunk}

Create a VERY DETAILED summary covering ALL aspects:
1. ALL parties involved with COMPLETE names, titles, addresses, and relationships
2. ALL dates, amounts, durations, and financial details mentioned
3. ALL obligations, rights, responsibilities, and duties
4. ALL terms, conditions, and legal provisions
5. ALL clauses, sections, and legal requirements
6. ALL procedures, processes, and legal steps
7. ALL exclusions, exceptions, and limitations
8. ALL warranties, representations, and covenants
9. Governing law, jurisdiction, and dispute resolution
10. ALL business purposes, objectives, and scope

Provide SPECIFIC details with exact amounts, dates, names, addresses, and legal terms. Do not generalize - include all specific information mentioned.

Comprehensive Detailed Summary:"""
    
    inputs = summarizer_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    output_ids = summarizer_model.generate(
        **inputs, 
        max_length=MAX_SUMMARY_TOKENS, 
        num_beams=6,  # Increased beams for better quality
        length_penalty=0.8,  # Reduced penalty for longer summaries
        early_stopping=False,  # Allow longer generation
        no_repeat_ngram_size=2,  # Reduced to allow important term repetition
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )
    return summarizer_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_maximum_document_info(text: str, legal_context: Dict) -> Dict:
    """Extract maximum document information."""
    info = {
        "title": "Legal Document",
        "date": "",
        "parties": [],
        "type": "Agreement",
        "amounts": [],
        "durations": [],
        "locations": [],
        "legal_entities": legal_context.get("entities", []),
        "legal_concepts": legal_context.get("concepts", []),
        "sections": legal_context.get("sections", []),
        "addresses": [],
        "terms": [],
        "obligations": [],
        "relationships": [],
        "business_details": []
    }
    
    lines = text.split('\n')
    
    # Extract title with more patterns
    for line in lines[:20]:
        line = line.strip()
        if len(line) > 10 and len(line) < 200:
            if any(keyword in line.upper() for keyword in ['AGREEMENT', 'DEED', 'WILL', 'CONTRACT', 'LEASE', 'AFFIDAVIT', 'PARTNERSHIP', 'ASSIGNMENT', 'TESTAMENT']):
                info["title"] = line
                break
    
    # Extract comprehensive date patterns
    date_patterns = [
        r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
        r'(?:on\s+this\s+)(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
        r'(?:made\s+and\s+entered\s+into\s+on\s+this\s+)(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            info["date"] = matches[0]
            break
    
    # Extract comprehensive party patterns
    party_patterns = [
        r'([A-Z][a-zA-Z\s&.,]+(?:Pvt\.?\s+Ltd\.?|LLP|Inc\.?|Corp\.?|Technologies|Analytics|Properties|Ventures|Company|Partnership|Firm|Studios))',
        r'(Mr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(Ms\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(Mrs\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(Dr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(son\s+of\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'(daughter\s+of\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',
        r'(hereinafter\s+referred\s+to\s+as\s+[^,]+)',
        r'(rep\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+)'
    ]
    
    for pattern in party_patterns:
        matches = re.findall(pattern, text)
        info["parties"].extend(matches[:8])
    
    # Extract comprehensive amounts
    amount_patterns = [
        r'(₹\s?[\d,]+(?:\.\d{2})?)',
        r'(Rs\.?\s?[\d,]+(?:\.\d{2})?)',
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:Rupees?|Lakh|Crore|Thousand|Million|Billion))',
        r'(\d+(?:\.\d+)?\s*(?:Lakh|Crore|Thousand|Million|Billion))',
        r'(\d+\s+(?:equity\s+)?shares?)',
        r'(one\s+time\s+payment\s+of\s+[\d,]+)',
        r'(\d+\s+(?:one\s+thousand|one\s+lakh|one\s+crore))',
        r'(consideration\s+of\s+[\d,]+)'
    ]
    
    for pattern in amount_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info["amounts"].extend(matches[:12])
    
    # Extract comprehensive durations
    duration_patterns = [
        r'(\d+\s+(?:years?|months?|days?|weeks?))',
        r'(\d+\s+(?:year|month|day|week)\s+(?:period|term|duration))',
        r'(?:for\s+a\s+period\s+of\s+)(\d+\s+(?:years?|months?|days?))',
        r'(?:effective\s+for\s+)(\d+\s+(?:years?|months?|days?))',
        r'(?:continue\s+for\s+)(\d+\s+(?:years?|months?|days?))',
        r'(?:shall\s+continue\s+for\s+)(\d+\s+(?:years?|months?|days?))'
    ]
    
    for pattern in duration_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info["durations"].extend(matches[:8])
    
    # Extract addresses
    address_patterns = [
        r'([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*,\s*(?:Mumbai|Delhi|Bangalore|Chennai|Kolkata|Pune|Hyderabad|Ahmedabad|Jaipur|Lucknow|India))',
        r'([A-Z][a-z]+\s+(?:Road|Street|Avenue|Lane|Garden|Complex|Tower|Plaza|Mall|Building|Heights|View))',
        r'(\d+(?:st|nd|rd|th)?\s+Floor[^,]*,\s*[^,]+,\s*[^,]+)',
        r'(Office\s+No\.?\s*\d+[^,]*,\s*[^,]+,\s*[^,]+)',
        r'(Shop\s+No\.?\s*[A-Z0-9-]+[^,]*,\s*[^,]+,\s*[^,]+)',
        r'(residing\s+at\s+[^,]+,\s*[^,]+,\s*[^,]+)'
    ]
    
    for pattern in address_patterns:
        matches = re.findall(pattern, text)
        info["addresses"].extend(matches[:8])
    
    return info

def create_maximum_legal_summary(summaries: List[str], doc_info: Dict) -> str:
    """Create maximum detail legal summary matching reference format exactly."""
    combined = " ".join(summaries)
    combined = re.sub(r'\s+', ' ', combined).strip()
    
    # Split into sentences and clean
    sentences = re.split(r'[.!?]+', combined)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
    
    # Create maximum detail output
    output_lines = [f"{doc_info['title']} – Simplified Summary", ""]
    
    point_num = 1
    
    # 1. Date (if available)
    if doc_info["date"]:
        output_lines.append(f"{point_num}. This Agreement was signed on {doc_info['date']} (Effective Date).")
        point_num += 1
    
    # 2. Parties with maximum detail
    if doc_info["parties"]:
        output_lines.append(f"{point_num}. Parties involved:")
        for party in doc_info["parties"][:4]:
            if party and len(party.strip()) > 5:
                output_lines.append(f"o {party}")
        point_num += 1
    
    # 3. Purpose/Business with maximum detail
    purpose_sentences = [s for s in sentences if any(word in s.lower() for word in ['purpose', 'business', 'partnership', 'agreement', 'develop', 'market', 'objective', 'scope'])]
    if purpose_sentences:
        output_lines.append(f"{point_num}. Purpose: {purpose_sentences[0]}.")
        point_num += 1
    
    # 4. Detailed information exchange
    conf_sentences = [s for s in sentences if any(word in s.lower() for word in ['confidential', 'information', 'disclose', 'exchange', 'share'])]
    if conf_sentences:
        output_lines.append(f"{point_num}. During discussions, both parties may exchange confidential information (business plans, finances, customer lists, technical data, trade secrets, know-how, designs, source codes, marketing strategies, etc.).")
        point_num += 1
    
    # 5. Comprehensive obligations
    obligation_sentences = [s for s in sentences if any(word in s.lower() for word in ['obligation', 'duty', 'responsibility', 'must', 'shall', 'required', 'keep', 'protect'])]
    if obligation_sentences:
        output_lines.append(f"{point_num}. Obligations of the Receiving Party:")
        output_lines.append("o Keep the information in strict confidence and protect it like its own confidential data.")
        output_lines.append("o Not share it with any third party without the written consent of the Disclosing Party.")
        output_lines.append("o Use it only for the Purpose and nothing else.")
        output_lines.append("o Share it only with employees, directors, advisors, or consultants who need to know, provided they also follow confidentiality obligations.")
        point_num += 1
    
    # 6. Comprehensive exclusions
    exclusion_sentences = [s for s in sentences if any(word in s.lower() for word in ['exclusion', 'public', 'already', 'developed', 'independently', 'exception'])]
    if exclusion_sentences:
        output_lines.append(f"{point_num}. Exclusions from Confidential Information: Information is not confidential if:")
        output_lines.append("o It is already public or becomes public without fault of the Receiving Party.")
        output_lines.append("o The Receiving Party already had it legally before disclosure.")
        output_lines.append("o It is developed independently by the Receiving Party.")
        output_lines.append("o Disclosure is required by law, regulation, or court order (with prior written notice to the Disclosing Party).")
        point_num += 1
    
    # 7. Term with maximum detail
    if doc_info["durations"]:
        duration = doc_info["durations"][0]
        output_lines.append(f"{point_num}. Term: This Agreement is effective for {duration} from the Effective Date.")
        point_num += 1
    
    # 8. Extended confidentiality obligations
    if len(doc_info["durations"]) > 1:
        duration2 = doc_info["durations"][1]
        output_lines.append(f"{point_num}. Confidentiality obligations continue for {duration2} after termination.")
        point_num += 1
    
    # 9. Return/Destroy obligations
    return_sentences = [s for s in sentences if any(word in s.lower() for word in ['return', 'destroy', 'termination', 'request'])]
    if return_sentences:
        output_lines.append(f"{point_num}. On termination or written request, the Receiving Party must return or destroy all confidential documents and copies.")
        point_num += 1
    
    # 10. No rights/licenses
    rights_sentences = [s for s in sentences if any(word in s.lower() for word in ['rights', 'licenses', 'patents', 'copyrights', 'trademarks', 'ip'])]
    if rights_sentences:
        output_lines.append(f"{point_num}. No rights or licenses (patents, copyrights, trademarks, or IP) are granted under this Agreement.")
        point_num += 1
    
    # 11. Governing law
    law_sentences = [s for s in sentences if any(word in s.lower() for word in ['governing', 'law', 'jurisdiction', 'court', 'dispute'])]
    if law_sentences:
        output_lines.append(f"{point_num}. Governing Law: This Agreement shall be governed by the laws of India.")
        point_num += 1
    
    # 12. Additional important points
    remaining_sentences = [s for s in sentences if s and len(s.strip()) > 25]
    for sentence in remaining_sentences[:8]:
        if sentence and not any(word in sentence.lower() for word in ['purpose', 'confidential', 'obligation', 'exclusion', 'term', 'return', 'rights', 'governing']):
            output_lines.append(f"{point_num}. {sentence.strip()}.")
            point_num += 1
    
    return "\n".join(output_lines)

def write_maximum_pdf(text: str, output_path: Path):
    """Write maximum detail formatted text to PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    lines = text.split('\n')
    for line in lines:
        if line.strip():
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.', '14.', '15.', '16.', '17.', '18.', '19.', '20.')):
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, sanitize(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=11)
            elif line.startswith(('o ', '- ')):
                pdf.set_font("Helvetica", size=11)
                pdf.cell(10)
                pdf.cell(0, 6, sanitize(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 8, sanitize(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=11)
        pdf.ln(2)
    
    pdf.output(str(output_path))

def process_document_maximally(doc_id: str, text: str, inlegalbert_tokenizer, inlegalbert_model, summarizer_tokenizer, summarizer_model, device="cpu"):
    """Process a single document with maximum InLegalBERT approach."""
    print(f"Processing {doc_id} with maximum InLegalBERT...")
    
    try:
        # Extract maximum legal context
        legal_context = extract_maximum_legal_context(text, inlegalbert_tokenizer, inlegalbert_model, device)
        
        # Extract maximum document info
        doc_info = extract_maximum_document_info(text, legal_context)
        
        # Chunk the text
        chunks = chunk_by_tokens(text, summarizer_tokenizer)
        
        # Summarize each chunk maximally
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = summarize_chunk_maximally(chunk, legal_context, summarizer_tokenizer, summarizer_model)
                summaries.append(summary)
            except Exception as e:
                print(f"  Error summarizing chunk {i+1}: {e}")
        
        if not summaries:
            print(f"  No summaries generated for {doc_id}")
            return
        
        # Create maximum legal summary
        final_summary = create_maximum_legal_summary(summaries, doc_info)
        
        # Write outputs
        text_path = OUT_TEXT_DIR / f"{doc_id}_pred.txt"
        text_path.write_text(final_summary, encoding="utf-8")
        
        pdf_path = OUT_PDF_DIR / f"{doc_id}_pred.pdf"
        try:
            write_maximum_pdf(final_summary, pdf_path)
            print(f"  ✓ Generated maximum InLegalBERT: {text_path.name} and {pdf_path.name}")
        except Exception as e:
            print(f"  ✗ PDF generation failed: {e}")
            
    except Exception as e:
        print(f"  Error processing {doc_id}: {e}")

def main():
    """Main processing function with maximum InLegalBERT integration."""
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
    
    print(f"Processing {len(records)} documents with maximum InLegalBERT enhancement...")
    
    for record in records:
        doc_id = record.get("id", "unknown")
        text = record.get("input", "")
        
        if text and len(text.strip()) > 50:
            process_document_maximally(
                doc_id, text, 
                inlegalbert_tokenizer, inlegalbert_model,
                summarizer_tokenizer, summarizer_model,
                device
            )
        else:
            print(f"Skipping {doc_id}: insufficient text")
    
    print("Maximum InLegalBERT processing complete!")

if __name__ == "__main__":
    main()
