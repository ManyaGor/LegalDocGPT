#!/usr/bin/env python3
"""
Setup script to install spaCy and download the English model for NER.
Run this before starting the server to enable NER capabilities.
"""

import subprocess
import sys
import os

def install_spacy():
    """Install spaCy package."""
    print("Installing spaCy...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy==3.7.2"])
        print("✅ spaCy installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing spaCy: {e}")
        return False

def download_english_model():
    """Download the English language model for spaCy."""
    print("Downloading English language model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ English model downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading English model: {e}")
        return False

def test_ner():
    """Test NER functionality."""
    print("Testing NER functionality...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        # Test with sample text
        test_text = "John Smith works at Microsoft Corporation in Seattle, Washington."
        doc = nlp(test_text)
        
        entities = []
        for ent in doc.ents:
            entities.append(f"{ent.text} ({ent.label_})")
        
        print(f"✅ NER test successful! Found entities: {entities}")
        return True
    except Exception as e:
        print(f"❌ NER test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up NER for LegalDocGPT...")
    print("=" * 50)
    
    # Install spaCy
    if not install_spacy():
        print("❌ Failed to install spaCy. Please install manually: pip install spacy==3.7.2")
        return False
    
    # Download English model
    if not download_english_model():
        print("❌ Failed to download English model. Please install manually: python -m spacy download en_core_web_sm")
        return False
    
    # Test NER
    if not test_ner():
        print("❌ NER test failed. Please check your installation.")
        return False
    
    print("=" * 50)
    print("🎉 NER setup completed successfully!")
    print("You can now start the LegalDocGPT server with enhanced NER capabilities.")
    print("\nTo start the server:")
    print("cd api")
    print("python server.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
