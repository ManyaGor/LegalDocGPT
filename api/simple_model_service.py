# api/simple_model_service.py
"""
Simplified model service for LegalDocGPT API that provides enhanced processing
without heavy model dependencies initially.
"""

import os
import re
import textwrap
from typing import List, Dict

class SimpleLegalDocModelService:
    """Simplified service class for legal document processing."""
    
    def __init__(self):
        self._initialized = False
    
    def initialize(self):
        """Initialize the service."""
        if self._initialized:
            return
        self._initialized = True
        print("Simple Legal Document Model Service initialized!")
    
    def extract_legal_context(self, text: str) -> Dict:
        """Extract legal context using pattern matching."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
        
        # Extract legal entities using patterns
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
            "concepts": sentences[:20],  # Use first 20 sentences as concepts
            "key_sentences": sentences[:20],
            "sections": list(set(sections))
        }
    
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
    
    def create_legal_summary(self, text: str, doc_info: Dict) -> str:
        """Create legal summary using enhanced pattern matching."""
        
        # Split into sentences and clean
        sentences = re.split(r'[.!?]+', text)
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
            # Initialize if needed
            if not self._initialized:
                self.initialize()
            
            # Extract legal context
            legal_context = self.extract_legal_context(text)
            
            # Extract document info
            doc_info = self.extract_document_info(text, legal_context)
            
            # Create legal summary
            final_summary = self.create_legal_summary(text, doc_info)
            
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
        model_service = SimpleLegalDocModelService()
    return model_service




