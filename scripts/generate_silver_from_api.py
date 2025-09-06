# scripts/generate_silver_from_api.py
# Usage: python scripts\generate_silver_from_api.py
import os, json, time
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "silver")
API_URL = "http://127.0.0.1:8000/process"  # backend endpoint

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith((".pdf", ".docx"))])
if not files:
    print("No PDF/DOCX files found in", RAW_DIR)
    raise SystemExit(1)

for fname in files:
    path = os.path.join(RAW_DIR, fname)
    print("\nProcessing:", fname)
    try:
        with open(path, "rb") as fh:
            files_payload = {"file": (fname, fh, "application/octet-stream")}
            r = requests.post(API_URL, files=files_payload, timeout=300)
        if r.status_code != 200:
            print("  ERROR from API:", r.status_code, r.text)
            continue
        data = r.json()
        rec = {
            "id": fname,
            "title": data.get("title") or "",
            "input_path": path,
            "target": "\n".join(data.get("points", [])),
            "points_list": data.get("points", [])
        }
        out_path = os.path.join(OUT_DIR, fname + ".jsonl")
        with open(out_path, "w", encoding="utf-8") as w:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("  Wrote silver file:", out_path)
        time.sleep(0.6)
    except Exception as e:
        print("  Exception:", str(e))
