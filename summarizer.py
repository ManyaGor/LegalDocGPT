from typing import List, Dict
import sys
import os

# Try to import transformers for actual LLM calls
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: Transformers not available, using placeholder generation")

MAP_SYSTEM_PROMPT = (
    "You are summarizing a legal document for a smart non-lawyer.\n"
    "Goals: Be clear, concise, and simple. Use layman-friendly words while keeping all facts (names, dates, durations, numbers) exact.\n"
    "Include any exceptions or conditions that significantly affect meaning.\n"
    "Constraints:\n"
    "- Use only information from the provided evidence.\n"
    "- Add short inline citations like [Heading, p.X] after factual statements.\n"
    "- Never invent or interpret beyond the text.\n"
)

REDUCE_SYSTEM_PROMPT = (
    "Combine the summaries from all chunks into a single, coherent summary in plain, everyday English.\n"
    "- Merge overlapping or duplicate points.\n"
    "- Keep exact names, numbers, and dates.\n"
    "- Include critical conditions, exceptions, or penalties if they change meaning.\n"
    "- Add [Heading, p.X] citations after each factual statement.\n"
    "- If two parts conflict, include a note like ‘Possible conflict: …’ with both citations.\n"
    "Write 10–18 concise bullets for a non-lawyer.\n"
)


def format_map_input(breadcrumb: str, evidence: List[Dict]) -> str:
    lines = [breadcrumb, "", "EVIDENCE:"]
    for e in evidence:
        lines.append(f"- {e['sentence']} [{e['heading']}, p.{e['page']}]")
    lines.append("")
    lines.append("TASK:\nStep 1: Identify the key items as discovery bullets (each ends with a citation).\n"
                 "Step 2: Write 2–5 plain-English summary bullets with citations.")
    return "\n".join(lines)


# Global LLM pipeline
llm_pipeline = None

def initialize_llm():
    """Initialize the LLM pipeline."""
    global llm_pipeline
    if not LLM_AVAILABLE:
        return False
    
    try:
        # Use a much smaller, faster model for real-time processing
        llm_pipeline = pipeline(
            "text-generation",
            model="distilgpt2",  # Much smaller and faster
            max_length=256,
            temperature=0.3,
            do_sample=True,
            pad_token_id=50256,
            truncation=True
        )
        return True
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return False

def generate_with_llm(prompt: str, temperature: float = 0.2, max_new_tokens: int = 600) -> str:
    """Generate text using the LLM pipeline."""
    if not LLM_AVAILABLE or not llm_pipeline:
        return placeholder_generate(prompt, temperature, max_new_tokens)
    
    try:
        # Truncate prompt if too long
        max_input_length = 1024
        if len(prompt) > max_input_length:
            prompt = prompt[:max_input_length]
        
        result = llm_pipeline(
            prompt,
            max_length=min(len(prompt.split()) + max_new_tokens, 512),
            temperature=temperature,
            do_sample=True,
            pad_token_id=50256
        )
        
        generated_text = result[0]['generated_text']
        # Extract only the new generated part
        if len(generated_text) > len(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return placeholder_generate(prompt, temperature, max_new_tokens)

def placeholder_generate(prompt: str, temperature: float = 0.2, max_new_tokens: int = 600) -> str:
    # Placeholder LLM call – integrate actual model here (e.g., HF Inference)
    return (
        "DISCOVERED:\n"
        "• [placeholder discovered item 1]\n"
        "• [placeholder discovered item 2]\n\n"
        "SUMMARY:\n"
        "• [placeholder summary 1] [Heading, p.X]\n"
        "• [placeholder summary 2] [Heading, p.X]"
    )


def map_summarize_chunk(breadcrumb: str, evidence: List[Dict]) -> str:
    user = format_map_input(breadcrumb, evidence)
    prompt = MAP_SYSTEM_PROMPT + "\n\n" + user
    return generate_with_llm(prompt, temperature=0.2, max_new_tokens=600)


def reduce_summaries(map_outputs: List[str]) -> str:
    # Combine map outputs into a reduce prompt
    joined = "\n\n".join(map_outputs)
    prompt = REDUCE_SYSTEM_PROMPT + "\n\n" + joined
    return generate_with_llm(prompt, temperature=0.2, max_new_tokens=1000)
