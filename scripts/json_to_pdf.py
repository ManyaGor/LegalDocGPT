"""
Convert unified JSON outputs in data/predictions_json to PDFs in data/predictions_pdf.
Works with fpdf2.
"""

import json
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos

PRED_JSON_DIR = Path("data/predictions_json")
PRED_PDF_DIR = Path("data/predictions_pdf")


def sanitize(s: str) -> str:
    repl = {
        "•": "-", "₹": "Rs.", "–": "-", "—": "-",
        "'": "'", "\u00A0": " "
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out.encode("latin-1", "ignore").decode("latin-1")


def write_pdf_from_json(doc: dict, output_path: Path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def h1(text: str):
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=11)
        pdf.ln(2)

    def h2(text: str):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=11)

    def li(text: str, indent: int = 5):
        pdf.cell(indent)
        pdf.cell(0, 6, sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    doc_id = doc.get("id", "document")
    h1(f"Summary – {doc_id}")

    # Enhanced block
    h2("Enhanced Summary")
    enh = doc.get("enhanced", {})
    enh_info = enh.get("info", {})
    if enh.get("summary"):
        li(enh.get("summary"))
    # Key fields
    for label, key in [
        ("Title", "title"), ("Date", "date"), ("Type", "type")
    ]:
        val = enh_info.get(key)
        if val:
            li(f"{label}: {val}")
    for label, key in [
        ("Parties", "parties"), ("Amounts", "amounts"), ("Durations", "durations"),
        ("Locations", "locations"), ("Legal Entities", "legal_entities")
    ]:
        items = enh_info.get(key, [])
        if items:
            h2(label)
            for it in items[:10]:
                li(f"• {it}")

    # Final block
    h2("Final Summary")
    fin = doc.get("final", {})
    fin_info = fin.get("info", {})
    if fin.get("summary"):
        li(fin.get("summary"))
    for label, key in [
        ("Title", "title"), ("Date", "date"), ("Type", "type")
    ]:
        val = fin_info.get(key)
        if val:
            li(f"{label}: {val}")
    for label, key in [
        ("Parties", "parties"), ("Amounts", "amounts"), ("Durations", "durations"),
        ("Locations", "locations"), ("Sections", "sections"), ("Addresses", "addresses"),
        ("Terms", "terms"), ("Obligations", "obligations"), ("Legal Entities", "legal_entities")
    ]:
        items = fin_info.get(key, [])
        if items:
            h2(label)
            for it in items[:15]:
                li(f"• {it}")

    pdf.output(str(output_path))


def main():
    if not PRED_JSON_DIR.exists():
        print(f"JSON dir not found: {PRED_JSON_DIR}")
        return
    PRED_PDF_DIR.mkdir(parents=True, exist_ok=True)

    for json_file in sorted(PRED_JSON_DIR.glob("*_pred.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            out_pdf = PRED_PDF_DIR / f"{data.get('id','document')}_pred.pdf"
            write_pdf_from_json(data, out_pdf)
            print(f"✓ Wrote {out_pdf.name}")
        except Exception as e:
            print(f"✗ Failed {json_file.name}: {e}")


if __name__ == "__main__":
    main()



