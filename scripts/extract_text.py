import os
import docx
import PyPDF2

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

if __name__ == "__main__":
    file_path = input("Enter the path of the document (PDF/DOCX): ").strip()

    if not os.path.exists(file_path):
        print("File not found. Please check the path again.")
        exit()

    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        print("Unsupported file format. Please provide PDF or DOCX.")
        exit()

    print("\n--- Extracted Text ---\n")
    print(text)

    # Save extracted text into outputs folder
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "extracted_text.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\nExtracted text saved to {output_path}")
