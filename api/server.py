# api/server.py
import os, re, shutil, tempfile, textwrap
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fpdf import FPDF
from simple_model_service import get_model_service

# --- CORS (allow Next.js on localhost:3000) ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model service will be loaded lazily ---

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

# Old summarization functions removed - now using model service

@app.post("/process")
async def process(file: UploadFile = File(...)):
    # create tmp working dir
    tmpdir = tempfile.mkdtemp(prefix="legal-")
    try:
        in_path = os.path.join(tmpdir, file.filename)
        with open(in_path, "wb") as f:
            f.write(await file.read())

        # extract text
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

        # Get model service and process document
        model_service = get_model_service()
        result = model_service.process_document(cleaned)
        
        if "error" in result:
            return JSONResponse({"error": result["error"]}, status_code=500)

        # Extract points from result
        points = result.get("points", [])
        summary_text = result.get("summary_text", "")
        
        if not points:
            return JSONResponse({"error": "No summary points produced."}, status_code=500)

        # write PDF
        pdf_path = os.path.join(tmpdir, "simplified_summary.pdf")
        pdf_write(points, pdf_path, title="Simplified Summary")

        # move PDF into project outputs for convenience
        os.makedirs("outputs", exist_ok=True)
        final_pdf = os.path.join("outputs", "simplified_summary.pdf")
        shutil.copyfile(pdf_path, final_pdf)

        # return both: bullets (json) + downloadable PDF
        return {
            "points": points,
            "pdf_path": "/download",  # frontend will call this to download
            "summary_text": summary_text,
            "doc_info": result.get("doc_info", {})
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

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "LegalDocGPT Backend is running!"}

if __name__ == "__main__":
    import uvicorn
    print("Starting LegalDocGPT Backend Server...")
    uvicorn.run(app, host="127.0.0.1", port=8001)