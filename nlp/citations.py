from typing import Optional

def format_citation(heading: Optional[str], page: Optional[int]) -> str:
    if heading and page:
        return f"[{heading}, p.{page}]"
    if page:
        return f"[p.{page}]"
    if heading:
        return f"[{heading}]"
    return "[Excerpt]"

