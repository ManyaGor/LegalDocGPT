import spacy
import os

def perform_ner(text):
    nlp = spacy.load("en_core_web_sm")  # Small English model
    doc = nlp(text)

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

if __name__ == "__main__":
    file_path = input("Enter the path of the cleaned text file: ")

    if not os.path.exists(file_path):
        print("File not found. Please check the path again.")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            cleaned_text = f.read()

        entities = perform_ner(cleaned_text)

        # Save to outputs
        output_path = "outputs/entities.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for ent, label in entities:
                f.write(f"{ent} --> {label}\n")

        print("\n--- Named Entities ---\n")
        for ent, label in entities:
            print(f"{ent} --> {label}")

        print(f"\nEntities saved to {output_path}")
