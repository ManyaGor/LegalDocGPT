"""
Run both InLegalBERT summarizers concurrently per document and emit
consistent JSON outputs (no PDF/TXT).

Outputs per document (under data/predictions_json):
{
  "id": "doc1",
  "enhanced": { "summary": "...", "info": { ... } },
  "final": { "summary": "...", "info": { ... } }
}
"""

import json
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import logging as hf_logging

# Import the two summarizer modules (robust to direct or module execution)
try:
    from . import inlegalbert_enhanced_summarizer as enh  # type: ignore
    from . import inlegalbert_final_summarizer as fin  # type: ignore
except Exception:
    # Allow running via: python LegalDocGPT/scripts/run_both_summarizers.py
    current_dir = Path(__file__).resolve().parent
    pkg_root = current_dir.parent
    sys.path.insert(0, str(pkg_root))
    sys.path.insert(0, str(pkg_root.parent))
    import inlegalbert_enhanced_summarizer as enh  # type: ignore
    import inlegalbert_final_summarizer as fin  # type: ignore


DATASET_PATH = Path("dataset/dataset.jsonl")
OUT_JSON_DIR = Path("data/predictions_json")

# Silence HF warnings
hf_logging.set_verbosity_error()


def summarize_with_enhanced(doc_id: str, text: str, device: str,
                            inlegalbert_tokenizer, inlegalbert_model,
                            summarizer_tokenizer, summarizer_model):
    """Run the enhanced pipeline and return (summary, info)."""
    legal_context = enh.extract_legal_entities_and_concepts(
        text, inlegalbert_tokenizer, inlegalbert_model, device
    )
    doc_info = enh.extract_document_info_enhanced(text, legal_context)

    chunks = enh.chunk_by_tokens(text, summarizer_tokenizer)
    summaries = []
    for chunk in chunks:
        summaries.append(
            enh.summarize_chunk_with_legal_context(
                chunk, legal_context, summarizer_tokenizer, summarizer_model
            )
        )
    final_summary = enh.create_enhanced_legal_summary(summaries, doc_info)
    # Normalize lists (dedupe, keep order where possible)
    for k in [
        "parties","amounts","durations","locations","legal_entities","legal_concepts","key_sentences"
    ]:
        if k in doc_info and isinstance(doc_info[k], list):
            seen = set()
            normalized = []
            for v in doc_info[k]:
                if v not in seen:
                    seen.add(v)
                    normalized.append(v)
            doc_info[k] = normalized
    return final_summary, doc_info


def summarize_with_final(doc_id: str, text: str, device: str,
                         inlegalbert_tokenizer, inlegalbert_model,
                         summarizer_tokenizer, summarizer_model):
    """Run the final pipeline and return (summary, info)."""
    legal_context = fin.extract_maximum_legal_context(
        text, inlegalbert_tokenizer, inlegalbert_model, device
    )
    doc_info = fin.extract_maximum_document_info(text, legal_context)

    chunks = fin.chunk_by_tokens(text, summarizer_tokenizer)
    summaries = []
    for chunk in chunks:
        summaries.append(
            fin.summarize_chunk_maximally(
                chunk, legal_context, summarizer_tokenizer, summarizer_model
            )
        )
    final_summary = fin.create_maximum_legal_summary(summaries, doc_info)
    # Normalize lists
    for k in [
        "parties","amounts","durations","locations","legal_entities","legal_concepts","key_sentences","sections","addresses","terms","obligations","relationships","business_details"
    ]:
        if k in doc_info and isinstance(doc_info[k], list):
            seen = set()
            normalized = []
            for v in doc_info[k]:
                if v not in seen:
                    seen.add(v)
                    normalized.append(v)
            doc_info[k] = normalized
    return final_summary, doc_info


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return

    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Running both summarizers concurrently on {len(records)} documents (device={device})...")

    # Preload models once (separate instances per pipeline to allow safe parallelism)
    # Enhanced models
    enh_inl_tok, enh_inl_model, enh_sum_tok, enh_sum_model = enh.load_models()
    enh_inl_model = enh_inl_model.to(device)
    enh_sum_model = enh_sum_model.to(device)

    # Final models (separate instances)
    fin_inl_tok, fin_inl_model, fin_sum_tok, fin_sum_model = fin.load_models()
    fin_inl_model = fin_inl_model.to(device)
    fin_sum_model = fin_sum_model.to(device)

    for record in records:
        doc_id = record.get("id", "unknown")
        text = record.get("input", "")
        if not text or len(text.strip()) < 50:
            print(f"Skipping {doc_id}: insufficient text")
            continue

        # Run enhanced and final in parallel for this doc
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    summarize_with_enhanced, doc_id, text, device,
                    enh_inl_tok, enh_inl_model, enh_sum_tok, enh_sum_model
                ): "enhanced",
                executor.submit(
                    summarize_with_final, doc_id, text, device,
                    fin_inl_tok, fin_inl_model, fin_sum_tok, fin_sum_model
                ): "final",
            }

            results = {}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    summary, info = fut.result()
                    results[key] = {"summary": summary, "info": info}
                except Exception as e:
                    results[key] = {"error": str(e)}

        # Write unified JSON
        # Ensure deterministic key order and content
        output = {
            "id": doc_id,
            "enhanced": {
                "summary": results.get("enhanced", {}).get("summary", ""),
                "info": results.get("enhanced", {}).get("info", {}),
            },
            "final": {
                "summary": results.get("final", {}).get("summary", ""),
                "info": results.get("final", {}).get("info", {}),
            },
        }
        (OUT_JSON_DIR / f"{doc_id}_pred.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"✓ Wrote JSON for {doc_id}")

    print("Done.")


if __name__ == "__main__":
    main()


