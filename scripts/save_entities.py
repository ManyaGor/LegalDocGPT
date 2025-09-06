import spacy
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Input cleaned text
input_path = "outputs/cleaned_text.txt"
output_path = "outputs/ner_results.txt"

if not os.path.exists(input_path):
    print("❌ Cleaned text not found. Run preprocess_text.py first.")
else:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)

    with open(output_path, "w", encoding="utf-8") as f:
        for ent in doc.ents:
            f.write(f"{ent.text}\t({ent.label_})\n")

    print(f"✅ Entities saved to {output_path}")
