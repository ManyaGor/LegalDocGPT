"""
Enhanced model service integrating the new NLP pipeline with NER capabilities.
"""

import os
import re
import textwrap
from typing import List, Dict, Optional
import sys
sys.path.append('..')

# Import our new NLP modules
from nlp.chunking import chunk_document
from nlp.extractive import select_sentences
from nlp.validate import compare
from nlp.citations import format_citation
from summarizer import map_summarize_chunk, reduce_summaries, initialize_llm

# Try to import NER capabilities
try:
    import spacy
    NER_AVAILABLE = True
except ImportError:
    NER_AVAILABLE = False
    print("Warning: spaCy not available, using regex fallback for NER")

try:
    from transformers import pipeline
    HF_NER_AVAILABLE = True
except ImportError:
    HF_NER_AVAILABLE = False
    print("Warning: Transformers not available for NER fallback")


class EnhancedLegalDocModelService:
    """Enhanced service with new NLP pipeline and NER capabilities."""
    
    def __init__(self):
        self._initialized = False
        self.nlp = None
        self.ner_pipeline = None
    
    def initialize(self):
        """Initialize the service with NER capabilities."""
        if self._initialized:
            return
        
        # Try spaCy first
        if NER_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy NER model loaded successfully!")
            except OSError:
                print("⚠️ spaCy model not found, trying transformers fallback...")
                self.nlp = None
        
        # Fallback to Hugging Face NER
        if not self.nlp and HF_NER_AVAILABLE:
            try:
                self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
                print("✅ Hugging Face NER model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Hugging Face NER failed: {e}")
                self.ner_pipeline = None
        
        if not self.nlp and not self.ner_pipeline:
            print("⚠️ No NER models available, using regex fallback")
        
        # Initialize LLM for generation
        print("Initializing LLM for text generation...")
        llm_success = initialize_llm()
        if llm_success:
            print("✅ LLM initialized successfully!")
        else:
            print("⚠️ LLM initialization failed, using placeholder generation")
        
        self._initialized = True
        print("Enhanced Legal Document Model Service initialized!")
    
    def extract_entities_with_ner(self, text: str) -> Dict:
        """Extract entities using available NER models."""
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': [],
            'legal_entities': [],
            'addresses': [],
            'signatures': []
        }
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities['persons'].append(ent.text)
                elif ent.label_ == "ORG":
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ("GPE", "LOC"):
                    entities['locations'].append(ent.text)
                elif ent.label_ == "DATE":
                    entities['dates'].append(ent.text)
                elif ent.label_ == "MONEY":
                    entities['money'].append(ent.text)
        
        # Use Hugging Face NER if available
        elif self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)
                for entity in ner_results:
                    label = entity.get("entity_group", "")
                    word = entity.get("word", "")
                    if label == "PER":
                        entities['persons'].append(word)
                    elif label == "ORG":
                        entities['organizations'].append(word)
                    elif label in ("LOC", "GPE"):
                        entities['locations'].append(word)
            except Exception as e:
                print(f"NER pipeline error: {e}")
        
        # Add legal-specific pattern matching
        legal_patterns = {
            'legal_entities': [
                r'(?:Pvt\.?\s+Ltd\.?|LLP|Inc\.?|Corp\.?|Company|Partnership|Firm)',
                r'(?:Supreme Court|High Court|District Court|Court of|Tribunal)',
                r'(?:Plaintiff|Defendant|Appellant|Respondent|Petitioner)'
            ],
            'addresses': [
                r'(?:Floor|Road|Street|Avenue|Lane|Garden|Complex|Tower|Plaza|Mall|Building|Heights|View)',
                r'(?:Mumbai|Delhi|Bangalore|Chennai|Kolkata|Pune|Hyderabad|Ahmedabad|Jaipur|Lucknow)'
            ],
            'signatures': [
                r'(?:For\s+[^:]+:\s*[^,]+,\s*[^,]+)',
                r'(?:Witness:\s*[^,]+,\s*[^,]+)',
                r'(?:Signature:\s*[^,]+)'
            ]
        }
        
        for category, patterns in legal_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities[category].extend(matches)
        
        return entities
    
    def extract_pdf_page_info(self, text: str) -> Dict[int, tuple]:
        """Extract page information from PDF text (placeholder implementation)."""
        # This is a simplified implementation
        # In practice, you'd use PyPDF2 or similar to get actual page breaks
        lines = text.split('\n')
        pages = {}
        current_page = 1
        
        # Simple heuristic: assume page breaks at certain patterns
        for i, line in enumerate(lines):
            if re.search(r'\f', line) or re.search(r'^Page \d+', line, re.IGNORECASE):
                current_page += 1
            if i % 50 == 0:  # Rough estimate
                pages[i // 50] = (current_page, current_page)
        
        return pages
    
    def process_document_enhanced(self, text: str, doc_title: str = "Legal Document") -> Dict:
        """Process document using the new NLP pipeline with NER."""
        try:
            # Initialize if needed
            if not self._initialized:
                self.initialize()
            
            # Step 1: Extract entities with NER
            entities = self.extract_entities_with_ner(text)
            
            # Step 2: Get page information
            pages = self.extract_pdf_page_info(text)
            
            # Step 3: Chunk the document
            chunks = chunk_document(doc_title, text, pages)
            
            # Step 4: Extract evidence from each chunk
            map_outputs = []
            for chunk in chunks:
                # Parse breadcrumb
                lines = chunk.split('\n')
                breadcrumb = lines[0] if lines else "BREADCRUMB: Unknown"
                chunk_text = '\n'.join(lines[2:]) if len(lines) > 2 else chunk
                
                # Extract heading and page from breadcrumb
                heading_match = re.search(r'▸\s*([^|]+)', breadcrumb)
                page_match = re.search(r'pages=(\d+)', breadcrumb)
                heading = heading_match.group(1).strip() if heading_match else "Unknown"
                page = int(page_match.group(1)) if page_match else 1
                
                # Select evidence sentences
                evidence = select_sentences(chunk_text, heading, page, top_k=10)
                
                # Map step: summarize chunk
                map_output = map_summarize_chunk(breadcrumb, evidence)
                map_outputs.append(map_output)
            
            # Step 5: Reduce step: combine all summaries
            final_summary = reduce_summaries(map_outputs)
            
            # Step 6: Validate
            discrepancies = compare(text, final_summary)
            if discrepancies:
                final_summary += "\n\n" + discrepancies
            
            # Step 7: Convert to bullet points
            lines = final_summary.split('\n')
            points = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('•', '-', '1.', '2.', '3.', '4.', '5.'))):
                    points.append(line)
            
            return {
                "points": points,
                "summary_text": final_summary,
                "entities": entities,
                "chunks_processed": len(chunks),
                "evidence_sentences": sum(len(select_sentences(chunk.split('\n')[2:], "Unknown", 1, 10)) for chunk in chunks)
            }
            
        except Exception as e:
            return {"error": f"Enhanced processing failed: {str(e)}"}


# Global instance
enhanced_model_service = None

def get_enhanced_model_service():
    """Get or create the global enhanced model service instance."""
    global enhanced_model_service
    if enhanced_model_service is None:
        enhanced_model_service = EnhancedLegalDocModelService()
    return enhanced_model_service
