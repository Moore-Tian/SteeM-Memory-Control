# concept_generator.py

import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import make_api_request, extract_json_from_text
from concept_prompt import build_concept_prompt


def load_events(events_file: Path) -> List[Dict[str, Any]]:
    """Load events from a JSON file."""
    with open(events_file, 'r', encoding='utf-8') as f:
        events = json.load(f)
    return events


def generate_concepts_for_timeline(
    events: List[Dict[str, Any]],
    domain: str = "research",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    sleep_s: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate concepts for a single timeline.
    
    Args:
        events: List of event dictionaries
        domain: Domain type, either "research" or "tutoring"
        model: Model name to use
        temperature: Temperature parameter
        sleep_s: Sleep seconds between API calls
    
    Returns:
        Tuple of (concepts_list, interaction_info)
        concepts_list: List of concept dictionaries with event_id, concept, and reason
        interaction_info: Dict containing input_messages, raw_output, and tokens
    """
    # Build prompt
    prompt = build_concept_prompt(events, domain=domain)
    
    # Prepare messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Make API request
    try:
        content, prompt_tokens, completion_tokens, total_tokens = make_api_request(
            messages=messages,
            model=model,
            response_format_json=True,
            temperature=temperature,
        )
        
        # Extract JSON from response
        concepts_data = extract_json_from_text(content)
        
        # Validate and format the response
        if isinstance(concepts_data, list):
            concepts = concepts_data
        elif isinstance(concepts_data, dict) and "concepts" in concepts_data:
            concepts = concepts_data["concepts"]
        else:
            raise ValueError(f"Unexpected response format: {type(concepts_data)}")
        
        # Get all valid event_ids from the events
        valid_event_ids = {event.get("event_id") for event in events if event.get("event_id")}
        
        # Validate each concept has required fields and valid event_id
        validated_concepts = []
        seen_event_ids = set()
        invalid_event_ids = []
        
        for concept in concepts:
            if not isinstance(concept, dict):
                continue
            if "event_id" in concept and "concept" in concept and "reason" in concept:
                event_id = concept["event_id"]
                
                # Validate event_id exists in the events
                if event_id not in valid_event_ids:
                    invalid_event_ids.append(event_id)
                    continue
                
                seen_event_ids.add(event_id)
                validated_concepts.append({
                    "event_id": event_id,
                    "concept": concept["concept"],
                    "reason": concept.get("reason", "")
                })
        
        # Report validation issues
        if invalid_event_ids:
            print(f"Warning: Found {len(invalid_event_ids)} concepts with invalid event_ids: {invalid_event_ids}")
        if len(validated_concepts) < 5:
            print(f"Warning: Only {len(validated_concepts)} valid concepts generated, expected 5")
        if len(seen_event_ids) < len(validated_concepts):
            print(f"Note: Some concepts share the same event_id (this is acceptable but not ideal)")

        # print("="*50)
        # print("="*50)
        # print(messages[0]["content"])
        # print("="*50)
        # print(content)
        
        # Prepare interaction info
        interaction_info = {
            "input_messages": messages,
            "raw_output": content,
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        
        # Sleep to avoid rate limiting
        if sleep_s > 0:
            time.sleep(sleep_s)
        
        return validated_concepts, interaction_info
    
    except Exception as e:
        print(f"Error generating concepts: {e}")
        return [], {
            "input_messages": messages,
            "raw_output": None,
            "error": str(e),
            "tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


def _run_single(
    subdir: Path,
    domain: str,
    model: str,
    temperature: float,
    sleep_s: float,
    skip_existing: bool,
    output_dir: Optional[Path],
    lock: threading.Lock,
) -> Dict[str, Any]:
    """Process a single subdirectory to generate concepts."""
    try:
        events_file = subdir / "events.json"
        
        # Determine save directory
        if output_dir is not None:
            save_dir = output_dir / domain / subdir.name
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = subdir
        
        concepts_file = save_dir / "concepts.json"
        stats_file = save_dir / "concept_stats.json"
        interactions_file = save_dir / "concept_interactions.json"
        
        # Check if events file exists
        if not events_file.exists():
            return {
                "run_dir": str(subdir),
                "success": False,
                "skipped": True,
                "reason": "events.json not found",
            }
        
        # Check if concepts already exist
        if skip_existing and concepts_file.exists():
            with lock:
                print(f"[SKIP] {subdir.name}: concepts.json already exists.", flush=True)
            return {
                "run_dir": str(subdir),
                "success": True,
                "skipped": True,
            }
        
        # Load events
        try:
            events = load_events(events_file)
            if not events:
                return {
                    "run_dir": str(subdir),
                    "success": False,
                    "skipped": True,
                    "reason": "no events found",
                }
        except Exception as e:
            return {
                "run_dir": str(subdir),
                "success": False,
                "error": f"Error loading events: {e}",
            }
        
        # Generate concepts
        concepts, interaction_info = generate_concepts_for_timeline(
            events=events,
            domain=domain,
            model=model,
            temperature=temperature,
            sleep_s=sleep_s,
        )
        
        if not concepts:
            return {
                "run_dir": str(subdir),
                "success": False,
                "error": "No concepts generated",
            }
        
        # Get token statistics
        tokens = interaction_info.get("tokens", {})
        
        # Record interaction
        interaction_record = {
            "run_dir": str(save_dir),
            "input_messages": interaction_info.get("input_messages"),
            "raw_output": interaction_info.get("raw_output"),
            "tokens": tokens,
        }
        if "error" in interaction_info:
            interaction_record["error"] = interaction_info["error"]
        
        # Save concepts
        try:
            with open(concepts_file, 'w', encoding='utf-8') as f:
                json.dump(concepts, f, indent=2, ensure_ascii=False)
            
            # Save stats for this run
            run_stats = {
                "prompt_tokens": tokens.get("prompt_tokens", 0),
                "completion_tokens": tokens.get("completion_tokens", 0),
                "total_tokens": tokens.get("total_tokens", 0),
                "num_concepts": len(concepts),
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(run_stats, f, indent=2, ensure_ascii=False)
            
            # Save interactions for this run
            with open(interactions_file, 'w', encoding='utf-8') as f:
                json.dump([interaction_record], f, indent=2, ensure_ascii=False)
            
            with lock:
                print(
                    f"[OK] {subdir.name}: concepts generated. Concepts={len(concepts)}, Tokens={tokens.get('total_tokens', 0)}",
                    flush=True,
                )
            
            return {
                "run_dir": str(subdir),
                "success": True,
                "skipped": False,
                "num_concepts": len(concepts),
                "tokens": tokens,
            }
        except Exception as e:
            return {
                "run_dir": str(subdir),
                "success": False,
                "error": f"Error saving files: {e}",
            }
    
    except Exception as e:
        with lock:
            print(f"[ERR] {subdir.name}: {e}", flush=True)
        return {
            "run_dir": str(subdir),
            "success": False,
            "error": str(e),
        }


def process_directory(
    base_dir: Path,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    sleep_s: float = 0.5,
    skip_existing: bool = True,
    verbose: bool = False,
    start_from: int = 0,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    max_workers: int = 4,
):
    """
    Process all timelines in a directory to generate concepts.
    
    Args:
        base_dir: Base directory containing subdirectories with events.json files
        model: Model name to use
        temperature: Temperature parameter
        sleep_s: Sleep seconds between API calls
        skip_existing: If True, skip directories that already have concepts.json
        verbose: If True, print progress for each directory
        start_from: Index in the sorted run dirs to start from (0-based)
        limit: Limit of run dirs to process from start_from
        output_dir: Output directory. If specified, saves to output_dir/{domain}/{subdir_name}/. If None, saves to original subdir.
        max_workers: Maximum number of concurrent threads
    """
    # Extract domain from base_dir (research or tutoring)
    domain = base_dir.name
    
    # Get all numeric subdirectories and sort by numeric value
    all_subdirs = []
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            all_subdirs.append(d)
    all_subdirs.sort(key=lambda d: int(d.name))
    
    if not all_subdirs:
        print(f"Warning: No numeric subdirectories found in {base_dir}")
        return
    
    # Apply start_from and limit
    if limit is not None:
        subdirs = all_subdirs[start_from:start_from + limit]
    else:
        subdirs = all_subdirs[start_from:]
    
    print(f"Found {len(all_subdirs)} numeric subdirectories in {base_dir}")
    print(f"Processing {len(subdirs)} subdirectories (from index {start_from})")
    if output_dir:
        print(f"Output directory: {output_dir}")
    else:
        print(f"Output directory: (same as input directory)")
    print(f"Max workers: {max_workers}")
    print(f"Skip existing: {skip_existing}")
    print()
    
    lock = threading.Lock()
    results: List[Dict[str, Any]] = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_single,
                subdir=subdir,
                domain=domain,
                model=model,
                temperature=temperature,
                sleep_s=sleep_s,
                skip_existing=skip_existing,
                output_dir=output_dir,
                lock=lock,
            )
            for subdir in subdirs
        ]
        
        for fut in as_completed(futures):
            results.append(fut.result())
    
    # Calculate statistics
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    skipped = [r for r in successful if r.get("skipped")]
    
    # Token statistics
    total_prompt_tokens = sum(r.get("tokens", {}).get("prompt_tokens", 0) for r in successful if not r.get("skipped"))
    total_completion_tokens = sum(r.get("tokens", {}).get("completion_tokens", 0) for r in successful if not r.get("skipped"))
    total_tokens = sum(r.get("tokens", {}).get("total_tokens", 0) for r in successful if not r.get("skipped"))
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total subdirectories found: {len(all_subdirs)}")
    print(f"Subdirectories to process: {len(subdirs)}")
    print(f"Successful: {len(successful)} (skipped: {len(skipped)})")
    print(f"Failed: {len(failed)}")
    print(f"\nToken Statistics:")
    print(f"  Total prompt tokens: {total_prompt_tokens:,}")
    print(f"  Total completion tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    
    if failed:
        print(f"\nFailed runs:")
        for r in failed:
            print(f"  - {Path(r['run_dir']).name}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate concepts from timelines")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing subdirectories with events.json files (e.g., examples/mix/research)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--sleep_s",
        type=float,
        default=0.5,
        help="Sleep seconds between API calls.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skip runs that already have concepts.json.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-directory progress.",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Index in the sorted run dirs to start from (0-based).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit of run dirs to process from start_from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for concepts. If not specified, concepts will be saved in the same directory as events.json. Structure: output_dir/{domain}/{subdir_name}/",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of concurrent threads.",
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.input_dir)
    if not base_dir.exists():
        raise ValueError(f"Input directory does not exist: {base_dir}")
    
    # Handle output_dir: if relative path, resolve relative to script directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        # If relative path, resolve relative to script's directory
        if not output_dir.is_absolute():
            script_dir = Path(__file__).parent
            output_dir = script_dir / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    process_directory(
        base_dir=base_dir,
        model=args.model,
        temperature=args.temperature,
        sleep_s=args.sleep_s,
        skip_existing=args.skip_existing,
        verbose=args.verbose,
        start_from=args.start_from,
        limit=args.limit,
        output_dir=output_dir,
        max_workers=args.max_workers,
    )
