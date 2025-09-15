import http.server
import socketserver
import json
import sys

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
    
    def log_message(self, format, *args):
        print(f"Request: {format % args}")

if __name__ == "__main__":
    print(f"Starting server on port {PORT}...")
    try:
        with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
            print(f"Server running at http://localhost:{PORT}")
            print("Press Ctrl+C to stop")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)



