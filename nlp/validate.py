import re
from typing import Dict, List

CRITICAL_PATTERNS = {
    'dates': [r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
              r"\b\d{1,2}/\d{1,2}/\d{4}\b"],
    'durations': [r"\b\d+\s+(years?|months?|days?)\b"],
    'amounts': [r"\b(?:₹|Rs\.?|USD|INR)\s?[\d,]+(?:\.\d{2})?\b"],
    'jurisdiction': [r"\bjurisdiction\b", r"\bcourts?\b", r"\bgoverning law\b", r"\barbitration\b"],
}


def extract_from_text(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {k: [] for k in CRITICAL_PATTERNS}
    for k, pats in CRITICAL_PATTERNS.items():
        for p in pats:
            out[k].extend(re.findall(p, text, flags=re.IGNORECASE))
    return out


def compare(source_text: str, final_summary: str) -> str:
    """Return discrepancy report string if mismatches or omissions are found."""
    src = extract_from_text(source_text)
    summ = extract_from_text(final_summary)

    discrepancies: List[str] = []

    # Simple checks: if something exists in source but absent in summary
    for key in src:
        if src[key] and not summ[key]:
            # report first instance
            sample = src[key][0] if isinstance(src[key][0], str) else str(src[key][0])
            discrepancies.append(f"- Found '{sample}' in source, but missing in summary ({key}).")

    if not discrepancies:
        return ""

    return "Possible discrepancies:\n" + "\n".join(discrepancies)

