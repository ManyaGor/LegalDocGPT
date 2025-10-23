#!/usr/bin/env python3
"""
Test script to demonstrate NER-based improvements in LegalDocGPT.
"""

import sys
import os
sys.path.append('api')

from simple_model_service import SimpleLegalDocModelService

def test_ner_improvements():
    """Test the NER-enhanced model."""
    
    model = SimpleLegalDocModelService()
    model.initialize()
    
    # Sample NDA text for testing
    nda_text = """
    MUTUAL NON-DISCLOSURE AGREEMENT
    
    This Mutual Non-Disclosure Agreement (the "Agreement") is entered into on this 22nd day of August, 2025 (the "Effective Date").
    
    BETWEEN:
    InnovateNext Technologies Pvt. Ltd., a company incorporated under the Companies Act, 2013, having its registered office at 7th Floor, Tech Park One, Powai, Mumbai, Maharashtra 400076, India (hereinafter referred to as "InnovateNext") of the FIRST PART;
    
    AND:
    DataWise Analytics LLP, registered under the LLP Act, 2008, having its office at A-wing, Office No. 502, Corporate Avenue, Malad (East), Mumbai, Maharashtra 400097, India (hereinafter referred to as "DataWise") of the SECOND PART;
    
    WHEREAS both parties are exploring a strategic partnership to develop and market an AI-driven predictive analytics platform for healthcare applications;
    
    NOW THEREFORE, the parties agree as follows:
    
    1. CONFIDENTIAL INFORMATION: During discussions, both parties may exchange confidential information including business plans, financial data, customer lists, technical specifications, trade secrets, proprietary methodologies, source codes, marketing strategies, and other sensitive information.
    
    2. OBLIGATIONS: The receiving party shall maintain strict confidentiality and shall not disclose any confidential information to third parties without written consent from the disclosing party.
    
    3. TERM: This agreement shall remain in effect for a period of 3 years from the effective date.
    
    4. GOVERNING LAW: This agreement shall be governed by the laws of India and courts in Mumbai shall have exclusive jurisdiction.
    
    5. DISPUTE RESOLUTION: Any disputes will be settled by arbitration under the Arbitration and Conciliation Act, 1996.
    
    IN WITNESS WHEREOF, the parties have executed this Agreement on the date first above written.
    
    For InnovateNext Technologies Pvt. Ltd.: Anjali Rao, CEO
    Witness: Vijay Kumar, C-110, Hiranandani Gardens, Powai, Mumbai 400076
    
    For DataWise Analytics LLP: Karan Desai, Designated Partner  
    Witness: Sunita Sharma, Flat 801, Royal Palms, Goregaon, Mumbai 400065
    """
    
    print("=" * 80)
    print("TESTING NER-ENHANCED LEGALDOCGPT")
    print("=" * 80)
    
    # Test NER entity extraction
    print("\n🔍 NER Entity Extraction:")
    entities = model.extract_entities_with_ner(nda_text)
    for category, items in entities.items():
        if items:
            print(f"  {category.upper()}: {items[:3]}")  # Show first 3 items
    
    # Test enhanced document info extraction
    print("\n📄 Enhanced Document Information:")
    doc_info = model.extract_enhanced_document_info(nda_text)
    print(f"  Title: {doc_info['title']}")
    print(f"  Date: {doc_info['date']}")
    print(f"  Parties: {doc_info['parties'][:2]}")
    print(f"  Organizations: {doc_info['organizations'][:2]}")
    print(f"  Locations: {doc_info['locations'][:2]}")
    print(f"  Money: {doc_info['amounts'][:2]}")
    print(f"  Durations: {doc_info['durations'][:2]}")
    
    # Test document type detection
    print(f"\n📋 Document Type: {model.detect_document_type(nda_text, doc_info)}")
    
    # Test full processing
    print("\n📝 Generated Summary:")
    result = model.process_document(nda_text)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Summary generated successfully!")
        print(f"📊 Points extracted: {len(result['points'])}")
        print("\nFirst 5 points:")
        for i, point in enumerate(result['points'][:5], 1):
            print(f"  {i}. {point}")
    
    print("\n" + "=" * 80)
    print("NER ENHANCEMENTS COMPLETED!")
    print("The model now uses spaCy NER for better entity extraction.")
    print("=" * 80)

if __name__ == "__main__":
    print("Testing NER-Enhanced LegalDocGPT Model")
    print("This demonstrates improved entity extraction and summarization")
    print()
    
    try:
        test_ner_improvements()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
