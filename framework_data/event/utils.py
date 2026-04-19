import os
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from dataclasses import asdict
import re

from openai import OpenAI

# ======================
# LLM client helper
# ======================

def get_openai_client():
    """
    Create an OpenAI client. Requires OPENAI_API_KEY in environment.

    You can also pass organization/project if needed by reading:
    - OPENAI_ORG
    - OPENAI_PROJECT
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")
    # The SDK reads the API key from env by default, so we just construct the client.
    client = OpenAI()
    return client


def make_api_request(
    messages: List[Dict[str, str]],
    model: str,
    response_format_json: bool = True,
    temperature: float = 0.7,
) -> Tuple[str, int, int, int]:
    """
    Make API request and return content and token usage.
    Includes retry mechanism: up to 3 retries with 3s delay between retries.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        response_format_json: Whether to request JSON format
        temperature: Temperature parameter
    
    Returns:
        Tuple of (content, prompt_tokens, completion_tokens, total_tokens)
    
    Raises:
        RuntimeError: If all retry attempts fail
    """
    max_retries = 3
    retry_delay = 3  # seconds
    
    last_exception = None
    
    for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (total 4 attempts)
        try:
            client = get_openai_client()
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format_json:
                kwargs["response_format"] = {"type": "json_object"}
            
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            if not content:
                raise RuntimeError("Empty content returned from the model.")
            
            # Extract token usage
            prompt_tokens = getattr(completion.usage, "prompt_tokens", 0) if hasattr(completion, "usage") and completion.usage else 0
            completion_tokens = getattr(completion.usage, "completion_tokens", 0) if hasattr(completion, "usage") and completion.usage else 0
            total_tokens = getattr(completion.usage, "total_tokens", 0) if hasattr(completion, "usage") and completion.usage else 0
            
            return content, prompt_tokens, completion_tokens, total_tokens
        
        except Exception as e:
            last_exception = e
            # If this is the last attempt, don't sleep
            if attempt < max_retries:
                time.sleep(retry_delay)
            # If this is the last attempt, break and raise
            if attempt == max_retries:
                break
    
    # If we get here, all retries failed
    raise RuntimeError(f"API request failed after {max_retries + 1} attempts. Last error: {last_exception}") from last_exception


# ======================
# Save utilities
# ======================

def _ensure_output_dir(output_dir: str = "output") -> Path:
    """Ensure output directory exists and return Path object."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_generation_results(
    output_dir: str,
    domain: str,
    events: List,
    interactions: List[Dict[str, Any]],
    run_id: int,
    token_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    """
    Save generation results to files:
    1. events.json: All generated events
    2. interactions.json: Complete interaction history (input and output for each step)
    3. stats.json: Token usage statistics
    
    Args:
        output_dir: Base output directory
        domain: Domain name (used as subdirectory)
        events: List of generated events
        interactions: List of interaction records (each with input messages and raw output)
        run_id: Run identifier (used as folder name)
        token_stats: Token usage statistics
    
    Returns:
        Dict with paths to saved files.
    """
    output_path = _ensure_output_dir(output_dir)
    
    # Create domain subdirectory
    domain_dir = output_path / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run subdirectory with run_id
    run_dir = domain_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save all events
    events_file = run_dir / "events.json"
    events_data = [asdict(ev) for ev in events]
    with open(events_file, "w", encoding="utf-8") as f:
        json.dump(events_data, f, ensure_ascii=False, indent=2, default=str)
    saved_files["events"] = str(events_file)
    
    # Save complete interaction history
    interactions_file = run_dir / "interactions.json"
    with open(interactions_file, "w", encoding="utf-8") as f:
        json.dump(interactions, f, ensure_ascii=False, indent=2, default=str)
    saved_files["interactions"] = str(interactions_file)
    
    # Save token statistics
    if token_stats:
        stats_file = run_dir / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(token_stats, f, ensure_ascii=False, indent=2, default=str)
        saved_files["stats"] = str(stats_file)
    
    return saved_files


# ======================
# JSON extraction utilities
# ======================

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from text that may be wrapped in code blocks or have extra text.
    
    Handles cases like:
    - Pure JSON: {"key": "value"}
    - Code block: ```json\n{"key": "value"}\n```
    - Code block: ```\n{"key": "value"}\n```
    - Text with JSON: Some text {"key": "value"} more text
    
    Args:
        text: Input text that may contain JSON
        
    Returns:
        Parsed JSON as a dictionary
        
    Raises:
        ValueError: If no valid JSON can be found or parsed
    """
    if not text or not text.strip():
        raise ValueError("Empty text provided")
    
    # Try direct JSON parsing first (fastest path)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from code blocks
    # Pattern 1: ```json ... ```
    json_block_pattern = r'```json\s*\n?(.*?)\n?```'
    match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Pattern 2: ``` ... ``` (generic code block)
    code_block_pattern = r'```\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in the text (look for {...})
    # Find the first { and try to match balanced braces
    brace_start = text.find('{')
    if brace_start != -1:
        brace_count = 0
        brace_end = -1
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    brace_end = i + 1
                    break
        
        if brace_end != -1:
            json_candidate = text[brace_start:brace_end]
            try:
                return json.loads(json_candidate.strip())
            except json.JSONDecodeError:
                pass
    
    # Try to find JSON array in the text (look for [...])
    bracket_start = text.find('[')
    if bracket_start != -1:
        bracket_count = 0
        bracket_end = -1
        for i in range(bracket_start, len(text)):
            if text[i] == '[':
                bracket_count += 1
            elif text[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    bracket_end = i + 1
                    break
        
        if bracket_end != -1:
            json_candidate = text[bracket_start:bracket_end]
            try:
                parsed = json.loads(json_candidate.strip())
                # If it's an array, we expect a dict, so this is a fallback
                return parsed
            except json.JSONDecodeError:
                pass
    
    # Last resort: try to clean and parse the entire text
    cleaned = text.strip()
    # Remove common prefixes/suffixes
    cleaned = re.sub(r'^[^{[]*', '', cleaned)  # Remove text before { or [
    cleaned = re.sub(r'[^}\]]*$', '', cleaned)  # Remove text after } or ]
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not extract valid JSON from text. "
            f"JSONDecodeError: {e}\n"
            f"Text preview: {text}..."
        )



if __name__ == "__main__":
    client = get_openai_client()