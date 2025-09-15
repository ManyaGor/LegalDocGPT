"""
Extract plain text from PDFs under data/expected output into .txt files
so we can inspect and match the exact layout/wording.
"""

from pathlib import Path
import pdfplumber

SRC_DIR = Path("data/expected output")
OUT_DIR = SRC_DIR / "_text"


def main():
    if not SRC_DIR.exists():
        print(f"Missing: {SRC_DIR}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(SRC_DIR.glob("*.pdf")):
        try:
            texts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    texts.append(page.extract_text() or "")
            content = "\n\n".join(texts)
            out_txt = OUT_DIR / (pdf_path.stem + ".txt")
            out_txt.write_text(content, encoding="utf-8")
            print(f"✓ Wrote {out_txt.relative_to(OUT_DIR.parent)}")
        except Exception as e:
            print(f"✗ Failed {pdf_path.name}: {e}")


if __name__ == "__main__":
    main()



