# scripts/build_dataset.py
"""
Robust dataset builder:
- Reads only .docx and .pdf from data\input and data\output
- Pairs by base name:  doc1.(docx/pdf)  <->  doc1_output.(docx/pdf)
- Skips unrelated files (desktop.ini, thumbs.db, etc.)
- Writes dataset/dataset.jsonl and dataset/dataset.csv
"""

import os
import json
import csv
from pathlib import Path

try:
    from docx import Document
except Exception:
    Document = None
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

ROOT = Path(".")
INPUT_DIR = ROOT / "data" / "input"
OUTPUT_DIR = ROOT / "data" / "output"
OUT_DIR = ROOT / "dataset"
OUT_JSONL = OUT_DIR / "dataset.jsonl"
OUT_CSV = OUT_DIR / "dataset.csv"

ALLOWED_INPUT_EXT = {".docx", ".pdf"}
ALLOWED_OUTPUT_EXT = {".docx", ".pdf"}

def read_docx(path: Path):
    if Document is None:
        raise RuntimeError("python-docx is required. Install: pip install python-docx")
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras).strip()

def read_pdf(path: Path):
    if PdfReader is None:
        raise RuntimeError("PyPDF2 is required. Install: pip install PyPDF2")
    reader = PdfReader(str(path))
    texts = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            texts.append(t)
    return "\n".join(texts).strip()

def read_file(path: Path):
    s = path.suffix.lower()
    if s == ".docx":
        return read_docx(path)
    if s == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")

def collect_files(folder: Path, allowed_exts):
    files = {}
    if not folder.exists():
        return files
    for p in sorted(folder.iterdir()):
        if not p.is_file(): 
            continue
        if p.suffix.lower() not in allowed_exts:
            # skip hidden/system files like desktop.ini
            continue
        files[p.name] = p
    return files

def base_name_for_output(filename: str):
    # Expect outputs like: doc1_output.docx or doc1-output.pdf or doc1_output_v1.pdf
    name = Path(filename).stem
    # remove common suffixes like _output, -output, _summary, -summary, _out
    for suf in ["_output", "-output", "_summary", "-summary", "_out", "-out"]:
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    # if it contains '_output' anywhere, try split
    if "_output" in name:
        return name.split("_output")[0]
    if "-output" in name:
        return name.split("-output")[0]
    # fallback: if name starts with docN_output pattern we may have doc1_output_v2
    parts = name.split("_")
    if parts and parts[0].lower().startswith("doc"):
        return parts[0]
    # otherwise return full stem so it won't match typical input names
    return name

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = collect_files(INPUT_DIR, ALLOWED_INPUT_EXT)
    outputs = collect_files(OUTPUT_DIR, ALLOWED_OUTPUT_EXT)

    if not inputs:
        print(f"No input files found in {INPUT_DIR.resolve()}. Place doc1.docx etc. there.")
        return
    if not outputs:
        print(f"No output files found in {OUTPUT_DIR.resolve()}. Place doc1_output.docx/pdf there.")
        return

    # map outputs by derived base name
    outputs_by_base = {}
    for fname, path in outputs.items():
        base = base_name_for_output(fname)
        outputs_by_base.setdefault(base.lower(), []).append(path)

    pairs = []
    missing_outputs = []
    for infname, inpath in inputs.items():
        instem = Path(infname).stem  # e.g., "doc1"
        key = instem.lower()
        # prefer exact base match
        if key in outputs_by_base and outputs_by_base[key]:
            # pick the first matching output (prefer .docx if multiple)
            candidates = sorted(outputs_by_base[key], key=lambda p: (p.suffix.lower() != ".docx", str(p)))
            outpath = candidates[0]
            pairs.append((inpath, outpath))
            # remove used candidate so we don't reuse it
            outputs_by_base[key].remove(outpath)
            continue
        # fallback: try to find any output whose base contains instem
        found = False
        for base, lst in outputs_by_base.items():
            if instem.lower() in base and lst:
                outpath = lst[0]
                pairs.append((inpath, outpath))
                outputs_by_base[base].remove(outpath)
                found = True
                break
        if not found:
            missing_outputs.append(infname)

    if missing_outputs:
        print("Warning: no matching output files found for these inputs:")
        for m in missing_outputs:
            print("  -", m)
        print("Proceeding with pairs found. You can fix filenames and re-run if needed.")

    print(f"Building dataset from {len(pairs)} pairs...")

    records = []
    for inpath, outpath in pairs:
        try:
            inp_text = read_file(inpath)
        except Exception as e:
            print(f"Error reading input {inpath}: {e}")
            continue
        try:
            out_text = read_file(outpath)
        except Exception as e:
            print(f"Error reading output {outpath}: {e}")
            continue
        records.append({
            "id": Path(inpath).stem,
            "input": inp_text,
            "target": out_text
        })

    if not records:
        print("No records were read successfully. Exiting.")
        return

    # write jsonl
    with open(OUT_JSONL, "w", encoding="utf-8") as jf:
        for r in records:
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")

    # write csv safely
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "input", "target"])
        for r in records:
            writer.writerow([r["id"], r["input"], r["target"]])

    print(f"✅ Wrote {len(records)} records to:")
    print("  -", OUT_JSONL.resolve())
    print("  -", OUT_CSV.resolve())
    print("\nPreview (first record id and first 300 chars of input/target):\n")
    first = records[0]
    print("ID:", first["id"])
    print("INPUT preview:", (first["input"][:300].replace("\n"," ") + ("..." if len(first["input"])>300 else "")))
    print("TARGET preview:", (first["target"][:300].replace("\n"," ") + ("..." if len(first["target"])>300 else "")))
    print("\nDone.")
    
if __name__ == "__main__":
    main()
