# scripts/test_fine_tuned_model.py
"""
Test the fine-tuned legal summarizer model.
"""

import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# Configuration
MODEL_PATH = Path("models/legal_summarizer")
DATASET_PATH = Path("dataset/dataset.jsonl")
OUTPUT_DIR = Path("data/predictions_finetuned")

def load_dataset(jsonl_path: Path):
    """Load the legal document dataset."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def prepare_prompt(text: str) -> str:
    """Prepare the input prompt for summarization."""
    return f"Summarize the following legal document in simple English:\n\n{text}"

def test_model():
    """Test the fine-tuned model."""
    if not MODEL_PATH.exists():
        print(f"Fine-tuned model not found at {MODEL_PATH}")
        return
    
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    
    # Load dataset
    records = load_dataset(DATASET_PATH)
    print(f"Testing on {len(records)} documents...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    total_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    for i, record in enumerate(records):
        doc_id = record["id"]
        input_text = record["input"]
        target_text = record["target"]
        
        print(f"Processing {doc_id}...")
        
        # Prepare input
        prompt = prepare_prompt(input_text)
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        outputs = model.generate(
            **inputs,
            max_length=300,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        # Decode output
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate ROUGE scores
        scores = scorer.score(target_text, summary)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure
        
        total_scores['rouge1'] += rouge1
        total_scores['rouge2'] += rouge2
        total_scores['rougeL'] += rougeL
        
        # Save output
        output_path = OUTPUT_DIR / f"{doc_id}_finetuned.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"{doc_id} - Fine-tuned Summary\n\n")
            f.write(summary)
        
        print(f"  ROUGE-1: {rouge1:.3f}, ROUGE-2: {rouge2:.3f}, ROUGE-L: {rougeL:.3f}")
    
    # Calculate average scores
    avg_scores = {
        'rouge1': total_scores['rouge1'] / len(records),
        'rouge2': total_scores['rouge2'] / len(records),
        'rougeL': total_scores['rougeL'] / len(records)
    }
    
    print("\nAverage ROUGE scores:")
    print(f"ROUGE-1: {avg_scores['rouge1']:.3f}")
    print(f"ROUGE-2: {avg_scores['rouge2']:.3f}")
    print(f"ROUGE-L: {avg_scores['rougeL']:.3f}")
    
    print(f"\nFine-tuned predictions saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    test_model()
