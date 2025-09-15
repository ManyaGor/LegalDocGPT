# api/model_service.py
"""
Model service for LegalDocGPT API that wraps InLegalBERT inference logic.
"""

import os
import re
import textwrap
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
INLEGALBERT_MODEL = "law-ai/InLegalBERT"
SUMMARIZATION_MODEL = "google/flan-t5-small"

MAX_INPUT_TOKENS = 512
MAX_SUMMARY_TOKENS = 800
OVERLAP_TOKENS = 64

class LegalDocModelService:
    """Service class for legal document processing using InLegalBERT."""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.inlegalbert_tokenizer = None
        self.inlegalbert_model = None
        self.summarizer_tokenizer = None
        self.summarizer_model = None
        self._models_loaded = False
    
    def load_models(self):
        """Load both InLegalBERT and summarization models."""
        if self._models_loaded:
            return
            
        print("Loading InLegalBERT model...")
        self.inlegalbert_tokenizer = AutoTokenizer.from_pretrained(INLEGALBERT_MODEL)
        self.inlegalbert_model = AutoModel.from_pretrained(INLEGALBERT_MODEL)
        
        print("Loading summarization model...")
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL)
        
        # Move models to device
        self.inlegalbert_model = self.inlegalbert_model.to(self.device)
        self.summarizer_model = self.summarizer_model.to(self.device)
        
        self._models_loaded = True
        print("Models loaded successfully!")
    
    def get_legal_embeddings(self, text: str) -> np.ndarray:
        """Get legal embeddings using InLegalBERT."""
        self.inlegalbert_model.eval()
        with torch.no_grad():
            inputs = self.inlegalbert_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            ).to(self.device)
            
            outputs = self.inlegalbert_model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            return mean_embeddings.cpu().numpy()
    
    def extract_legal_context(self, text: str) -> Dict:
        """Extract legal context using InLegalBERT."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
        
        if not sentences:
            return {"entities": [], "concepts": [], "key_sentences": [], "sections": []}
        
        # Get embeddings for all sentences
        sentence_embeddings = []
        for sentence in sentences:
            emb = self.get_legal_embeddings(sentence)
            sentence_embeddings.append(emb.flatten())
        
        sentence_embeddings = np.array(sentence_embeddings)
        doc_embedding = self.get_legal_embeddings(text[:1000]).flatten()
        
        # Calculate similarity scores
        similarities = cosine_similarity(sentence_embeddings, doc_embedding.reshape(1, -1)).flatten()
        
        # Select top sentences for comprehensive coverage
        top_indices = np.argsort(similarities)[-min(20, len(sentences)):]
        key_sentences = [sentences[i] for i in top_indices]
        
        # Extract legal entities
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
            entities.extend(matches[:10])
        
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
    
    def chunk_by_tokens(self, text: str, max_len=MAX_INPUT_TOKENS, overlap=OVERLAP_TOKENS) -> List[str]:
        """Split text into overlapping chunks."""
        ids = self.summarizer_tokenizer.encode(text, truncation=False)
        chunks, start, stride = [], 0, max_len - overlap
        while start < len(ids):
            end = min(start + max_len, len(ids))
            chunks.append(self.summarizer_tokenizer.decode(ids[start:end], skip_special_tokens=True))
            if end == len(ids): 
                break
            start += stride
        return chunks
    
    def summarize_chunk(self, chunk: str, legal_context: Dict) -> str:
        """Summarize a chunk with legal context."""
        
        # Build detailed prompt
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
        
        inputs = self.summarizer_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
        output_ids = self.summarizer_model.generate(
            **inputs, 
            max_length=MAX_SUMMARY_TOKENS, 
            num_beams=6,
            length_penalty=0.8,
            early_stopping=False,
            no_repeat_ngram_size=2,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50
        )
        return self.summarizer_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def extract_document_info(self, text: str, legal_context: Dict) -> Dict:
        """Extract document information."""
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
        
        # Extract title
        for line in lines[:20]:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                if any(keyword in line.upper() for keyword in ['AGREEMENT', 'DEED', 'WILL', 'CONTRACT', 'LEASE', 'AFFIDAVIT', 'PARTNERSHIP', 'ASSIGNMENT', 'TESTAMENT']):
                    info["title"] = line
                    break
        
        # Extract date patterns
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
        
        # Extract party patterns
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
        
        # Extract amounts
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
        
        # Extract durations
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
        
        return info
    
    def create_legal_summary(self, summaries: List[str], doc_info: Dict) -> str:
        """Create legal summary matching reference format."""
        combined = " ".join(summaries)
        combined = re.sub(r'\s+', ' ', combined).strip()
        
        # Split into sentences and clean
        sentences = re.split(r'[.!?]+', combined)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
        
        # Create detailed output
        output_lines = [f"{doc_info['title']} – Simplified Summary", ""]
        
        point_num = 1
        
        # 1. Date (if available)
        if doc_info["date"]:
            output_lines.append(f"{point_num}. This Agreement was signed on {doc_info['date']} (Effective Date).")
            point_num += 1
        
        # 2. Parties with detail
        if doc_info["parties"]:
            output_lines.append(f"{point_num}. Parties involved:")
            for party in doc_info["parties"][:4]:
                if party and len(party.strip()) > 5:
                    output_lines.append(f"o {party}")
            point_num += 1
        
        # 3. Purpose/Business
        purpose_sentences = [s for s in sentences if any(word in s.lower() for word in ['purpose', 'business', 'partnership', 'agreement', 'develop', 'market', 'objective', 'scope'])]
        if purpose_sentences:
            output_lines.append(f"{point_num}. Purpose: {purpose_sentences[0]}.")
            point_num += 1
        
        # 4. Information exchange
        conf_sentences = [s for s in sentences if any(word in s.lower() for word in ['confidential', 'information', 'disclose', 'exchange', 'share'])]
        if conf_sentences:
            output_lines.append(f"{point_num}. During discussions, both parties may exchange confidential information (business plans, finances, customer lists, technical data, trade secrets, know-how, designs, source codes, marketing strategies, etc.).")
            point_num += 1
        
        # 5. Obligations
        obligation_sentences = [s for s in sentences if any(word in s.lower() for word in ['obligation', 'duty', 'responsibility', 'must', 'shall', 'required', 'keep', 'protect'])]
        if obligation_sentences:
            output_lines.append(f"{point_num}. Obligations of the Receiving Party:")
            output_lines.append("o Keep the information in strict confidence and protect it like its own confidential data.")
            output_lines.append("o Not share it with any third party without the written consent of the Disclosing Party.")
            output_lines.append("o Use it only for the Purpose and nothing else.")
            output_lines.append("o Share it only with employees, directors, advisors, or consultants who need to know, provided they also follow confidentiality obligations.")
            point_num += 1
        
        # 6. Exclusions
        exclusion_sentences = [s for s in sentences if any(word in s.lower() for word in ['exclusion', 'public', 'already', 'developed', 'independently', 'exception'])]
        if exclusion_sentences:
            output_lines.append(f"{point_num}. Exclusions from Confidential Information: Information is not confidential if:")
            output_lines.append("o It is already public or becomes public without fault of the Receiving Party.")
            output_lines.append("o The Receiving Party already had it legally before disclosure.")
            output_lines.append("o It is developed independently by the Receiving Party.")
            output_lines.append("o Disclosure is required by law, regulation, or court order (with prior written notice to the Disclosing Party).")
            point_num += 1
        
        # 7. Term
        if doc_info["durations"]:
            duration = doc_info["durations"][0]
            output_lines.append(f"{point_num}. Term: This Agreement is effective for {duration} from the Effective Date.")
            point_num += 1
        
        # 8. Extended confidentiality
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
    
    def process_document(self, text: str) -> Dict:
        """Process a document and return summary points."""
        try:
            # Ensure models are loaded
            if not self._models_loaded:
                self.load_models()
            
            # Extract legal context
            legal_context = self.extract_legal_context(text)
            
            # Extract document info
            doc_info = self.extract_document_info(text, legal_context)
            
            # Chunk the text
            chunks = self.chunk_by_tokens(text)
            
            # Summarize each chunk
            summaries = []
            for i, chunk in enumerate(chunks):
                try:
                    summary = self.summarize_chunk(chunk, legal_context)
                    summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing chunk {i+1}: {e}")
            
            if not summaries:
                return {"error": "No summaries generated"}
            
            # Create legal summary
            final_summary = self.create_legal_summary(summaries, doc_info)
            
            # Convert to bullet points
            lines = final_summary.split('\n')
            points = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.', '14.', '15.', '16.', '17.', '18.', '19.', '20.')) or line.startswith('o ')):
                    points.append(line)
            
            return {
                "points": points,
                "summary_text": final_summary,
                "doc_info": doc_info
            }
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}

# Global instance
model_service = None

def get_model_service():
    """Get or create the global model service instance."""
    global model_service
    if model_service is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_service = LegalDocModelService(device=device)
    return model_service

