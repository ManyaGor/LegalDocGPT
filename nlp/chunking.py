import re
from typing import List, Dict, Tuple

HEADING_PATTERNS = [
    re.compile(r"^\s*(\d+(?:\.\d+)*)[\.)-]\s+"),  # numbered headings: 1., 2.1), 3- etc
    re.compile(r"^\s*[A-Z][A-Z\s/&-]{3,}$"),           # ALL-CAPS headings
]

SECTION_KEYWORDS = [
    "definitions", "term", "termination", "confidentiality", "dispute", "governing law",
    "arbitration", "liability", "warranties", "indemnity", "payment", "consideration",
    "scope", "purpose", "license", "jurisdiction", "notices"
]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def detect_heading(line: str) -> bool:
    text = line.strip()
    if any(p.search(text) for p in HEADING_PATTERNS):
        return True
    lower = text.lower()
    if any(k in lower for k in SECTION_KEYWORDS) and len(text) <= 120:
        return True
    return False


def split_into_sections(text: str) -> List[Tuple[str, List[str]]]:
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Preamble"
    current_lines: List[str] = []
    for line in lines:
        if detect_heading(line):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return sections


def sentences_from_text(text: str) -> List[str]:
    # simple sentence splitting, preserve periods in abbreviations lightly
    parts = SENTENCE_SPLIT.split(text.strip())
    # ensure we keep punctuation
    sentences: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.endswith(('.', '!', '?')):
            p += '.'
        sentences.append(p)
    return sentences


def chunk_document(doc_title: str, text: str, pages: Dict[int, Tuple[int, int]],
                   max_tokens: int = 1000, overlap_tokens: int = 120) -> List[str]:
    """
    Create structure-aware chunks:
    - split by headings first
    - then by sentences, packing ~800-1200 tokens with 100-150 overlap
    pages: optional map of section index -> (start_page, end_page)
    Returns list of chunk strings, each starting with a BREADCRUMB line.
    """
    # naive tokenization by whitespace count
    def token_count(s: str) -> int:
        return max(1, len(s.split()))

    sections = split_into_sections(text)
    chunks: List[str] = []

    for idx, (heading, lines) in enumerate(sections):
        section_text = "\n".join(lines).strip()
        sents = sentences_from_text(section_text)

        start_page, end_page = pages.get(idx, (1, 1)) if pages else (1, 1)
        breadcrumb = f"BREADCRUMB: {doc_title} ▸ {heading} | pages={start_page}-{end_page}".strip()

        # pack sentences into chunks with overlap
        pack: List[str] = []
        pack_tokens = 0
        i = 0
        while i < len(sents):
            sent = sents[i]
            t = token_count(sent)
            if pack_tokens + t > max_tokens and pack:
                body = " ".join(pack).strip()
                chunks.append(f"{breadcrumb}\n\n{body}")
                # overlap
                overlap_text = []
                overlap_tokens_acc = 0
                j = len(pack) - 1
                while j >= 0 and overlap_tokens_acc < overlap_tokens:
                    overlap_text.insert(0, pack[j])
                    overlap_tokens_acc += token_count(pack[j])
                    j -= 1
                pack = overlap_text
                pack_tokens = sum(token_count(s) for s in pack)
            else:
                pack.append(sent)
                pack_tokens += t
                i += 1
        if pack:
            body = " ".join(pack).strip()
            chunks.append(f"{breadcrumb}\n\n{body}")

    return chunks

