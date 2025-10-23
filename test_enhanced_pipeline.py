#!/usr/bin/env python3
"""
Test script for the enhanced NLP pipeline with NER capabilities.
"""

import sys
import os
sys.path.append('api')

from enhanced_model_service import EnhancedLegalDocModelService

def test_enhanced_pipeline():
    """Test the complete enhanced pipeline."""
    
    # Sample legal document text
    sample_text = """
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
    print("TESTING ENHANCED NLP PIPELINE WITH NER")
    print("=" * 80)
    
    # Initialize the enhanced service
    model = EnhancedLegalDocModelService()
    model.initialize()
    
    print("\n🔍 Testing NER Entity Extraction:")
    entities = model.extract_entities_with_ner(sample_text)
    for category, items in entities.items():
        if items:
            print(f"  {category.upper()}: {items[:3]}")  # Show first 3 items
    
    print("\n📄 Testing Enhanced Document Processing:")
    result = model.process_document_enhanced(sample_text, "MUTUAL NON-DISCLOSURE AGREEMENT")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Enhanced processing completed successfully!")
        print(f"📊 Chunks processed: {result.get('chunks_processed', 0)}")
        print(f"📊 Evidence sentences: {result.get('evidence_sentences', 0)}")
        print(f"📊 Points generated: {len(result.get('points', []))}")
        
        print("\n📝 Generated Summary (first 5 points):")
        for i, point in enumerate(result.get('points', [])[:5], 1):
            print(f"  {i}. {point}")
        
        print("\n🔍 NER Entities Found:")
        for category, items in result.get('entities', {}).items():
            if items:
                print(f"  {category.upper()}: {items[:3]}")
    
    print("\n" + "=" * 80)
    print("ENHANCED PIPELINE TEST COMPLETED!")
    print("The system now uses:")
    print("✅ Structure-aware chunking")
    print("✅ NER-based entity extraction")
    print("✅ Extractive evidence selection")
    print("✅ Map→reduce summarization")
    print("✅ Factual validation")
    print("✅ Citation formatting")
    print("=" * 80)

if __name__ == "__main__":
    print("Testing Enhanced LegalDocGPT Pipeline")
    print("This demonstrates the complete NLP pipeline with NER")
    print()
    
    try:
        test_enhanced_pipeline()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

