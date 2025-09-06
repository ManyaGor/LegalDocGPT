"use client";
import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [points, setPoints] = useState<string[]>([]);
  const [error, setError] = useState<string>("");
  const [serverFile, setServerFile] = useState<string>("");
  const [docTitle, setDocTitle] = useState<string>(""); // title from backend

  const handleUpload = async () => {
    if (!file) { setError("Please choose a PDF or DOCX file."); return; }
    setError(""); setLoading(true); setPoints([]); setServerFile(""); setDocTitle("");

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch("http://127.0.0.1:8000/process", {
        method: "POST",
        body: form,
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data?.error || "Failed to process document.");
        return;
      }

      setPoints(Array.isArray(data.points) ? data.points : []);
      setServerFile(typeof data.file === "string" ? data.file : "");
      setDocTitle(typeof data.title === "string" ? data.title : "");
    } catch (e) {
      setError("Could not reach backend. Is it running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  const downloadPDF = async () => {
    const fallback = "simplified_summary.pdf";
    const name = serverFile || (file?.name ? `simplified_${file.name.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/\.(pdf|docx)$/i, "")}.pdf` : fallback);

    try {
      const url = `http://127.0.0.1:8000/download?file=${encodeURIComponent(serverFile || name)}`;
      const res = await fetch(url);
      if (!res.ok) { setError("PDF not ready yet. Click 'Upload & Simplify' first."); return; }
      const blob = await res.blob();
      const objUrl = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objUrl; a.download = name;
      document.body.appendChild(a); a.click(); a.remove();
      window.URL.revokeObjectURL(objUrl);
    } catch {
      setError("Could not download the PDF.");
    }
  };

  const canDownload = points.length > 0 || !!serverFile;

  return (
    <main style={{
      minHeight:"100vh",
      background:"linear-gradient(180deg, #f7fafc 0%, #ffffff 100%)",
      padding:"24px",
      display:"flex", flexDirection:"column", alignItems:"center", gap:"16px"
    }}>
      <div style={{textAlign:"center", marginBottom:8}}>
        <h1 style={{fontSize:28, fontWeight:800, letterSpacing:0.2}}>Legal Document Simplifier (MVP)</h1>
        <p style={{color:"#475569"}}>Upload a legal PDF/DOCX → get a point-wise, plain-English summary.</p>
      </div>

      <div style={{width:"100%", maxWidth:860, display:"grid", gridTemplateColumns:"1fr", gap:16}}>
        <section style={{border:"1px solid #e5e7eb", borderRadius:16, padding:16, background:"#fff", boxShadow:"0 2px 10px rgba(0,0,0,0.03)"}}>
          <div style={{display:"grid", gap:12}}>
            <input
              type="file"
              accept=".pdf,.docx"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            {file && <div style={{color:"#334155"}}>Selected file: <b>{file.name}</b></div>}
            <div style={{display:"flex", gap:12, flexWrap:"wrap"}}>
              <button
                onClick={handleUpload}
                disabled={loading || !file}
                style={{
                  padding:"10px 16px", borderRadius:10, background:"#0f172a", color:"#fff",
                  border:"1px solid #0f172a", cursor:(loading||!file)?"not-allowed":"pointer", opacity:(loading||!file)?0.65:1
                }}
              >
                {loading ? "Processing…" : "Upload & Simplify"}
              </button>
              <button
                onClick={downloadPDF}
                disabled={!canDownload}
                style={{
                  padding:"10px 16px", borderRadius:10, background:"#fff", color:"#0f172a",
                  border:"1px solid #0f172a", cursor:canDownload?"pointer":"not-allowed", opacity:canDownload?1:0.5
                }}
                title={canDownload ? "Download simplified PDF" : "Process a document first"}
              >
                Download PDF
              </button>
            </div>
            {loading && <div style={{color:"#0f172a"}}>⏳ Summarizing… please wait.</div>}
            {error && <div style={{color:"#b91c1c"}}>{error}</div>}
          </div>
        </section>

        {points.length > 0 && (
          <section style={{border:"1px solid #e5e7eb", borderRadius:16, padding:16, background:"#fff", boxShadow:"0 2px 10px rgba(0,0,0,0.03)"}}>
            <div style={{marginBottom:10}}>
              <div style={{fontSize:18, fontWeight:700}}>
                {docTitle ? docTitle : "Simplified Summary"}
              </div>
              <div style={{color:"#64748b", fontSize:13}}>Point-wise explanation in plain language</div>
            </div>
            <ul style={{paddingLeft:20, display:"grid", gap:6}}>
              {points.map((p, i) => (<li key={i}>{p}</li>))}
            </ul>
          </section>
        )}
      </div>
    </main>
  );
}
