# scripts/eval_rouge.py
"""
Compute ROUGE between predictions (data/predictions_text/*_pred.txt) and targets in dataset/dataset.jsonl.
Shows per-doc preview and computes average ROUGE scores.
"""
import os
import json
from pathlib import Path
from rouge_score import rouge_scorer

ROOT = Path(".")
DATASET_JSONL = ROOT / "dataset" / "dataset.jsonl"
PRED_TEXT_DIR = ROOT / "data" / "predictions_text"

if not DATASET_JSONL.exists():
    print("ERROR: dataset/dataset.jsonl not found. Run build_dataset.py first.")
    raise SystemExit(1)

# load targets into dict by id
targets = {}
with open(DATASET_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        id_ = rec.get("id")
        # common keys fallback
        tgt = rec.get("target") or rec.get("simplified") or rec.get("output") or rec.get("summary") or rec.get("target_text") or ""
        if not tgt:
            # attempt other keys
            for k in rec.keys():
                if isinstance(rec[k], str) and len(rec[k]) > 10:
                    tgt = rec[k]
                    break
        targets[id_] = tgt.strip()

# load predictions
preds = {}
if not PRED_TEXT_DIR.exists():
    print(f"Predictions folder not found: {PRED_TEXT_DIR}")
    raise SystemExit(1)

for p in sorted(PRED_TEXT_DIR.glob("*_pred.txt")):
    id_ = p.stem.replace("_pred","")
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    preds[id_] = txt

# find overlap
common_ids = sorted([i for i in targets.keys() if i in preds])
if not common_ids:
    print("No overlapping ids between dataset targets and predictions.")
    print("Targets ids sample:", sorted(list(targets.keys()))[:10])
    print("Prediction ids sample:", sorted(list(preds.keys()))[:10])
    raise SystemExit(1)

scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

sum_r1 = sum_r2 = sum_rL = 0.0
count = 0

print(f"Evaluating {len(common_ids)} files...")

for idx, id_ in enumerate(common_ids):
    ref = targets[id_]
    hyp = preds[id_]
    if not hyp.strip():
        r1 = r2 = rL = 0.0
    else:
        scores = scorer.score(ref, hyp)
        r1 = scores["rouge1"].fmeasure
        r2 = scores["rouge2"].fmeasure
        rL = scores["rougeL"].fmeasure
    sum_r1 += r1
    sum_r2 += r2
    sum_rL += rL
    count += 1

    # print preview for first 6 docs
    if idx < 6:
        print("\n---", id_, "---")
        print("REF (first 300 chars):")
        print(ref[:300].replace("\n"," "))
        print("\nPRED (first 300 chars):")
        print(hyp[:300].replace("\n"," "))
        print("\nrouge1/2/L(f):", f"{r1:.3f}", f"{r2:.3f}", f"{rL:.3f}")

avg_r1 = sum_r1 / count if count else 0.0
avg_r2 = sum_r2 / count if count else 0.0
avg_rL = sum_rL / count if count else 0.0

print("\nAverage ROUGE scores across documents:")
print(f"ROUGE-1 (F1) avg: {avg_r1:.3f}")
print(f"ROUGE-2 (F1) avg: {avg_r2:.3f}")
print(f"ROUGE-L (F1) avg: {avg_rL:.3f}")

print("\nDocuments where predicted length << reference length (may be missing points):")
for id_ in common_ids:
    ref_len = len(targets[id_].split())
    pred_len = len(preds[id_].split())
    if ref_len > 0 and pred_len < 0.5 * ref_len:
        print(f" - {id_}: pred {pred_len} words, ref {ref_len} words")

print("\nDone.")
