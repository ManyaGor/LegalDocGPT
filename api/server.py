# api/server.py
import os, re, shutil, tempfile, textwrap
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fpdf import FPDF

# --- CORS (allow Next.js on localhost:3000) ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model (load once) ---
MODEL_NAME = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

MAX_INPUT_TOKENS  = 512
MAX_SUMMARY_TOKENS = 180
OVERLAP_TOKENS     = 64

# --- Utilities reused from your pipeline ---
def sanitize(s: str) -> str:
    repl = {
        "•": "-", "₹": "Rs.", "–": "-", "—": "-",
        "’": "'", "‘": "'", "“": '"', "”": '"', "\u00A0": " "
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out.encode("latin-1", "ignore").decode("latin-1")

def pdf_write(points: List[str], pdf_path: str, title="Simplified Summary"):
    from fpdf.enums import XPos, YPos
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, sanitize(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", size=12)
    for p in points:
        p = sanitize(p.strip())
        wrapped = textwrap.wrap(p, width=90)
        pdf.cell(5)
        pdf.cell(0, 7, f"- {wrapped[0]}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for cont in wrapped[1:]:
            pdf.cell(10)
            pdf.cell(0, 7, cont, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
    pdf.output(pdf_path)

def extract_text_pdf(path: str) -> str:
    import PyPDF2
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text() or ""
            text += t + "\n"
    return text

def extract_text_docx(path: str) -> str:
    import docx
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def preprocess_text(text: str) -> str:
    # keep light so we don't destroy legal content
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # fix hyphen breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_by_tokens(text: str, max_len=MAX_INPUT_TOKENS, overlap=OVERLAP_TOKENS) -> List[str]:
    ids = tokenizer.encode(text, truncation=False)
    chunks, start, stride = [], 0, max_len - overlap
    while start < len(ids):
        end = min(start + max_len, len(ids))
        chunks.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if end == len(ids): break
        start += stride
    return chunks

def summarize_chunk(t: str) -> str:
    inp = "summarize: " + t
    inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    out_ids = model.generate(
        **inputs, max_length=MAX_SUMMARY_TOKENS, num_beams=4,
        length_penalty=1.0, early_stopping=True, no_repeat_ngram_size=3
    )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

def to_bullets(text: str) -> List[str]:
    parts = re.split(r"(?:\n|\r|\r\n|•|-|\u2022|;|\.)(?:\s+|$)", text)
    points, seen = [], set()
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= 4:
            low = p.lower()
            if low not in seen:
                seen.add(low)
                points.append(p.rstrip(".") + ".")
    return points[:12]

@app.post("/process")
async def process(file: UploadFile = File(...)):
    # create tmp working dir
    tmpdir = tempfile.mkdtemp(prefix="legal-")
    try:
        in_path = os.path.join(tmpdir, file.filename)
        with open(in_path, "wb") as f:
            f.write(await file.read())

        # extract
        if in_path.lower().endswith(".pdf"):
            raw = extract_text_pdf(in_path)
        elif in_path.lower().endswith(".docx"):
            raw = extract_text_docx(in_path)
        else:
            return JSONResponse({"error": "Please upload a PDF or DOCX file."}, status_code=400)

        if len(raw.strip()) < 20:
            return JSONResponse({"error": "Could not extract enough text from the document."}, status_code=400)

        # preprocess
        cleaned = preprocess_text(raw)

        # summarize
        chunks = chunk_by_tokens(cleaned)
        partials = [summarize_chunk(ch) for ch in chunks]
        combined = " ".join(partials)
        bullets = to_bullets(combined)
        if not bullets:
            return JSONResponse({"error": "No summary points produced."}, status_code=500)

        # write PDF
        pdf_path = os.path.join(tmpdir, "simplified_summary.pdf")
        pdf_write(bullets, pdf_path, title="Simplified Summary")

        # move PDF into project outputs for convenience
        os.makedirs("outputs", exist_ok=True)
        final_pdf = os.path.join("outputs", "simplified_summary.pdf")
        shutil.copyfile(pdf_path, final_pdf)

        # return both: bullets (json) + downloadable PDF
        return {
            "points": bullets,
            "pdf_path": "/download"  # frontend will call this to download
        }
    finally:
        # keep tmp dir for this run; (comment next line if you want to inspect)
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.get("/download")
def download():
    path = os.path.join("outputs", "simplified_summary.pdf")
    if not os.path.exists(path):
        return JSONResponse({"error": "No PDF yet. Run /process first."}, status_code=404)
    return FileResponse(path, media_type="application/pdf", filename="simplified_summary.pdf")
