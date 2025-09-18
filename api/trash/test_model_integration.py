# test_model_integration.py
"""
Test script to verify the model integration is working.
"""

import requests
import json

def test_process_endpoint():
    """Test the /process endpoint with a sample document."""
    
    # Sample legal document text
    sample_text = """
    CONFIDENTIALITY AGREEMENT
    
    This Confidentiality Agreement ("Agreement") is made and entered into on this 15th day of March, 2024, between:
    
    ABC Technologies Pvt. Ltd., a company incorporated under the Companies Act, 2013, having its registered office at 123 Tech Park, Bangalore, Karnataka, India (hereinafter referred to as "Company A")
    
    AND
    
    XYZ Analytics LLP, a limited liability partnership firm registered under the Limited Liability Partnership Act, 2008, having its registered office at 456 Business Center, Mumbai, Maharashtra, India (hereinafter referred to as "Company B")
    
    WHEREAS, the parties wish to explore potential business opportunities and may need to share confidential information;
    
    NOW THEREFORE, the parties agree as follows:
    
    1. PURPOSE: The purpose of this Agreement is to facilitate discussions regarding potential partnership opportunities in the field of artificial intelligence and data analytics.
    
    2. CONFIDENTIAL INFORMATION: Confidential Information includes but is not limited to business plans, financial information, customer lists, technical data, trade secrets, know-how, designs, source codes, marketing strategies, and any other proprietary information.
    
    3. OBLIGATIONS: The Receiving Party shall:
       a) Keep all Confidential Information in strict confidence
       b) Not disclose any Confidential Information to third parties without written consent
       c) Use Confidential Information solely for the Purpose stated herein
       d) Return or destroy all Confidential Information upon termination
    
    4. TERM: This Agreement shall remain in effect for a period of two (2) years from the Effective Date.
    
    5. GOVERNING LAW: This Agreement shall be governed by the laws of India and any disputes shall be subject to the jurisdiction of the courts in Mumbai.
    
    IN WITNESS WHEREOF, the parties have executed this Agreement on the date first written above.
    """
    
    # Create a temporary text file
    with open("test_document.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # Test the endpoint
    url = "http://127.0.0.1:8001/process"
    
    try:
        with open("test_document.txt", "rb") as f:
            files = {"file": ("test_document.txt", f, "text/plain")}
            response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Model integration is working.")
            print(f"Number of summary points: {len(result.get('points', []))}")
            print("\nFirst few summary points:")
            for i, point in enumerate(result.get('points', [])[:5]):
                print(f"{i+1}. {point}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    finally:
        # Clean up
        import os
        if os.path.exists("test_document.txt"):
            os.remove("test_document.txt")

if __name__ == "__main__":
    test_process_endpoint()




