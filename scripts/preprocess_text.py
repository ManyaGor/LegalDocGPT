import re
import os

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    file_path = input("Enter the path of the extracted text file: ")

    if not os.path.exists(file_path):
        print("File not found. Please check the path again.")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned_text = preprocess_text(raw_text)

        # Save to outputs
        output_path = "outputs/cleaned_text.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print("\n--- Cleaned Text ---\n")
        print(cleaned_text)
        print(f"\nCleaned text saved to {output_path}")
