import argparse
import re
from typing import List


def flesch_kincaid_grade(text: str) -> float:
    # Lightweight readability proxy (very rough)
    sentences = max(1, len(re.split(r"[.!?]+", text)))
    words = max(1, len(text.split()))
    syllables = max(1, sum(len(re.findall(r"[aeiouy]+", w, re.I)) for w in text.split()))
    return 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovered", type=str, help="Path to discovered items text file", required=False)
    parser.add_argument("--summary", type=str, help="Path to final summary text file", required=True)
    args = parser.parse_args()

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = f.read()

    # Simple metrics
    grade = flesch_kincaid_grade(summary)

    # Citation hit-rate: count bullets with [ ... ]
    bullets = [b for b in summary.splitlines() if b.strip().startswith("•") or b.strip().startswith("-")]
    cited = [b for b in bullets if "[" in b and "]" in b]
    hit_rate = (len(cited) / max(1, len(bullets))) * 100

    print("Evaluation Results")
    print("-------------------")
    print(f"Readability (Flesch–Kincaid Grade): {grade:.2f} (target ≤ 8)")
    print(f"Citation hit-rate: {hit_rate:.1f}%")

if __name__ == "__main__":
    main()

