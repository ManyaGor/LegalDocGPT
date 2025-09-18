from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "LegalDocGPT Backend is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "LegalDocGPT Backend is running!"}

@app.post("/process")
async def process(file: UploadFile = File(...)):
    try:
        # Create a simple response for testing
        return {
            "points": [
                "This is a test summary point 1.",
                "This is a test summary point 2.",
                "This is a test summary point 3.",
                "The document has been processed successfully.",
                "This is a demonstration of the LegalDocGPT system."
            ],
            "pdf_path": "/download"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/download")
def download():
    return JSONResponse({"message": "PDF download endpoint - file processing not implemented yet"}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    print("Starting Simple LegalDocGPT Backend Server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
