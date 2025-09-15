"use client";
import { useState, useRef } from "react";
import { Button } from "../components/ui/Button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "../components/ui/Card";
import { Alert, AlertDescription } from "../components/ui/Alert";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [points, setPoints] = useState<string[]>([]);
  const [error, setError] = useState<string>("");
  const [serverFile, setServerFile] = useState<string>("");
  const [docTitle, setDocTitle] = useState<string>("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    if (!file) { 
      setError("Please choose a PDF or DOCX file."); 
      return; 
    }
    
    setError(""); 
    setLoading(true); 
    setPoints([]); 
    setServerFile(""); 
    setDocTitle("");
    setUploadProgress(0);

    try {
      const form = new FormData();
      form.append("file", file);

      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const res = await fetch("http://127.0.0.1:8001/process", {
        method: "POST",
        body: form,
      });
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      const data = await res.json();

      if (!res.ok) {
        setError(data?.error || "Failed to process document.");
        return;
      }

      setPoints(Array.isArray(data.points) ? data.points : []);
      setServerFile(typeof data.file === "string" ? data.file : "");
      setDocTitle(typeof data.title === "string" ? data.title : "");
    } catch (e) {
      setError("Could not reach backend. Is it running on port 8001?");
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const downloadPDF = async () => {
    const fallback = "simplified_summary.pdf";
    const name = serverFile || (file?.name ? `simplified_${file.name.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/\.(pdf|docx)$/i, "")}.pdf` : fallback);

    try {
      const url = `http://127.0.0.1:8001/download?file=${encodeURIComponent(serverFile || name)}`;
      const res = await fetch(url);
      if (!res.ok) { 
        setError("PDF not ready yet. Click 'Upload & Simplify' first."); 
        return; 
      }
      const blob = await res.blob();
      const objUrl = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objUrl; 
      a.download = name;
      document.body.appendChild(a); 
      a.click(); 
      a.remove();
      window.URL.revokeObjectURL(objUrl);
    } catch {
      setError("Could not download the PDF.");
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === "application/pdf" || 
          droppedFile.name.toLowerCase().endsWith('.docx')) {
        setFile(droppedFile);
        setError("");
      } else {
        setError("Please upload only PDF or DOCX files.");
      }
    }
  };

  const canDownload = points.length > 0 || !!serverFile;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">LegalDocGPT</h1>
                <p className="text-sm text-slate-600">AI-Powered Legal Document Analysis</p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-4 text-sm text-slate-600">
              <span className="flex items-center">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                Backend Connected
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-slate-900 mb-4">
            Transform Complex Legal Documents
          </h2>
          <p className="text-xl text-slate-600 mb-8 max-w-2xl mx-auto">
            Upload your legal PDFs or DOCX files and get instant, easy-to-understand summaries 
            powered by advanced AI technology.
          </p>
          
          {/* Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-white/60 backdrop-blur-sm rounded-xl p-6 border border-slate-200">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="font-semibold text-slate-900 mb-2">Lightning Fast</h3>
              <p className="text-sm text-slate-600">Process documents in seconds with our optimized AI models</p>
            </div>
            
            <div className="bg-white/60 backdrop-blur-sm rounded-xl p-6 border border-slate-200">
              <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="font-semibold text-slate-900 mb-2">Accurate Analysis</h3>
              <p className="text-sm text-slate-600">Advanced legal AI trained on thousands of documents</p>
            </div>
            
            <div className="bg-white/60 backdrop-blur-sm rounded-xl p-6 border border-slate-200">
              <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <h3 className="font-semibold text-slate-900 mb-2">Secure & Private</h3>
              <p className="text-sm text-slate-600">Your documents are processed securely and never stored</p>
            </div>
          </div>
        </div>

        {/* Upload Section */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Upload Your Document</CardTitle>
            <CardDescription>Drag and drop or click to select your PDF or DOCX file</CardDescription>
          </CardHeader>

          {/* Drag & Drop Area */}
          <div
            className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
              dragActive 
                ? 'border-blue-400 bg-blue-50' 
                : file 
                  ? 'border-green-400 bg-green-50' 
                  : 'border-slate-300 hover:border-slate-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx"
              onChange={(e) => {
                const selectedFile = e.target.files?.[0] || null;
                setFile(selectedFile);
                setError("");
              }}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            
            <div className="space-y-4">
              <div className="w-12 h-12 mx-auto bg-slate-100 rounded-full flex items-center justify-center">
                <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              
              {file ? (
                <div>
                  <p className="text-lg font-medium text-green-700 mb-2">✓ File Selected</p>
                  <p className="text-slate-600">{file.name}</p>
                  <p className="text-sm text-slate-500 mt-1">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-lg font-medium text-slate-700 mb-2">
                    {dragActive ? 'Drop your file here' : 'Choose a file or drag it here'}
                  </p>
                  <p className="text-slate-500">Supports PDF and DOCX files up to 50MB</p>
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 mt-6">
            <button
              onClick={handleUpload}
              disabled={loading || !file}
              className={`flex-1 py-3 px-6 rounded-xl font-semibold transition-all duration-200 ${
                loading || !file
                  ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
              }`}
            >
              {loading ? (
                <div className="flex items-center justify-center space-x-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Processing Document...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Analyze Document</span>
                </div>
              )}
            </button>
            
            <button
              onClick={downloadPDF}
              disabled={!canDownload}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 ${
                canDownload
                  ? 'bg-white text-slate-700 border-2 border-slate-300 hover:border-slate-400 hover:bg-slate-50'
                  : 'bg-slate-100 text-slate-400 cursor-not-allowed'
              }`}
            >
              <div className="flex items-center space-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>Download PDF</span>
              </div>
            </button>
          </div>

          {/* Progress Bar */}
          {loading && uploadProgress > 0 && (
            <div className="mt-4">
              <div className="flex justify-between text-sm text-slate-600 mb-2">
                <span>Processing...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-4">
              <Alert type="error" title="Error">
                {error}
              </Alert>
            </div>
          )}
        </Card>

        {/* Results Section */}
        {points.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between mb-4">
                <CardTitle>
                  {docTitle || "Document Analysis Results"}
                </CardTitle>
                <div className="flex items-center space-x-2 text-sm text-green-600">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Analysis Complete</span>
                </div>
              </div>
              <CardDescription>
                Here's your simplified summary with key points extracted from the document:
              </CardDescription>
            </CardHeader>

            <div className="space-y-4">
              {points.map((point, index) => (
                <div key={index} className="flex items-start space-x-3 p-4 bg-slate-50 rounded-xl border border-slate-200">
                  <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                    {index + 1}
                  </div>
                  <p className="text-slate-700 leading-relaxed">{point}</p>
                </div>
              ))}
            </div>

            <div className="mt-6 p-4 bg-blue-50 rounded-xl border border-blue-200">
              <div className="flex items-center space-x-2 mb-2">
                <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="font-semibold text-blue-800">Analysis Summary</span>
              </div>
              <p className="text-blue-700 text-sm">
                This analysis was generated using advanced AI models trained specifically on legal documents. 
                The summary provides a clear, point-by-point breakdown of the key information in your document.
              </p>
            </div>
          </Card>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white/80 backdrop-blur-sm border-t border-slate-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-slate-600">
            <p className="mb-2">© 2024 LegalDocGPT. All rights reserved.</p>
            <p className="text-sm">Powered by advanced AI technology for legal document analysis.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
