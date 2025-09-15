#!/usr/bin/env python3

import http.server
import socketserver
import json

PORT = 8001

class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"status": "healthy", "message": "Server is running!"}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/process':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "points": [
                    "This is a test summary point 1.",
                    "This is a test summary point 2.", 
                    "This is a test summary point 3.",
                    "The document has been processed successfully.",
                    "This demonstrates the LegalDocGPT system working."
                ],
                "pdf_path": "/download"
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"status": "healthy", "message": "Server is running!"}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/download':
            # Create a simple PDF content for download
            self.send_response(200)
            self.send_header('Content-Type', 'application/pdf')
            self.send_header('Content-Disposition', 'attachment; filename="simplified_summary.pdf"')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Simple PDF content (minimal PDF structure)
            pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj

4 0 obj
<<
/Length 200
>>
stream
BT
/F1 12 Tf
72 720 Td
(LegalDocGPT Summary) Tj
0 -20 Td
/F1 10 Tf
(This is a test summary point 1.) Tj
0 -15 Td
(This is a test summary point 2.) Tj
0 -15 Td
(This is a test summary point 3.) Tj
0 -15 Td
(The document has been processed successfully.) Tj
0 -15 Td
(This demonstrates the LegalDocGPT system working.) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000274 00000 n 
0000000525 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
612
%%EOF"""
            
            self.wfile.write(pdf_content)
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    print(f"Starting server on port {PORT}...")
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()
