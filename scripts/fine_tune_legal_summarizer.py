# scripts/fine_tune_legal_summarizer.py
"""
Fine-tune Flan-T5-small on the legal document dataset for specialized summarization.
This creates a custom model trained specifically on legal document simplification.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import numpy as np
from rouge_score import rouge_scorer

# Configuration
MODEL_NAME = "google/flan-t5-small"
DATASET_PATH = Path("dataset/dataset.jsonl")
OUTPUT_DIR = Path("models/legal_summarizer")
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 300

def load_dataset(jsonl_path: Path) -> List[Dict]:
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

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset for training."""
    # Prepare inputs
    inputs = [prepare_prompt(text) for text in examples["input"]]
    targets = examples["target"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=True
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    """Compute ROUGE metrics during evaluation."""
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(label, pred)
        rouge_scores.append({
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        })
    
    # Average scores
    avg_scores = {
        'rouge1': np.mean([s['rouge1'] for s in rouge_scores]),
        'rouge2': np.mean([s['rouge2'] for s in rouge_scores]),
        'rougeL': np.mean([s['rougeL'] for s in rouge_scores])
    }
    
    return avg_scores

def main():
    """Main fine-tuning function."""
    print("Loading dataset...")
    records = load_dataset(DATASET_PATH)
    print(f"Loaded {len(records)} records")
    
    # Create dataset
    dataset_dict = {
        "input": [record["input"] for record in records],
        "target": [record["target"] for record in records]
    }
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=10,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=5e-5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        report_to=None,  # Disable wandb
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluate on the full dataset
    print("Evaluating on full dataset...")
    eval_results = trainer.evaluate()
    print("Final evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"Fine-tuned model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
