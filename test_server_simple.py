#!/usr/bin/env python3
"""
Simple test to verify the server works without getting stuck.
"""

import requests
import json
import os

def test_server():
    """Test the server with a simple document."""
    
    # Sample legal text
    sample_text = """
    CONFIDENTIALITY AGREEMENT
    
    This Agreement is between Company A and Company B.
    
    The parties agree to keep confidential information secret for 2 years.
    
    This agreement is governed by the laws of California.
    """
    
    # Create a simple text file for testing
    with open("test_doc.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    print("Testing server with simple document...")
    
    try:
        # Test the server endpoint
        url = "http://127.0.0.1:8001/process"
        
        with open("test_doc.txt", "rb") as f:
            files = {"file": ("test_doc.txt", f, "text/plain")}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Server responded successfully!")
            print(f"📊 Points generated: {len(result.get('points', []))}")
            print("\n📝 Sample output:")
            for i, point in enumerate(result.get('points', [])[:3], 1):
                print(f"  {i}. {point}")
        else:
            print(f"❌ Server error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - server might be stuck")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server - is it running?")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up
        if os.path.exists("test_doc.txt"):
            os.remove("test_doc.txt")

if __name__ == "__main__":
    test_server()

