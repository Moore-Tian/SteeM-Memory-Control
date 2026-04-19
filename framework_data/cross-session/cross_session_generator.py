# cross_session_generator.py

import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import make_api_request, extract_json_from_text
from cross_session_prompt import build_cross_session_prompt, build_task_description


def load_events(events_file: Path) -> List[Dict[str, Any]]:
    """Load events from a JSON file."""
    with open(events_file, 'r', encoding='utf-8') as f:
        events = json.load(f)
    return events


def load_all_pref_regimes(regimes_file: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load all preference regimes from a JSON file."""
    with open(regimes_file, 'r', encoding='utf-8') as f:
        regimes = json.load(f)
    return regimes


def load_cross_session_topics(topics_file: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load cross-session topics from a JSON file."""
    with open(topics_file, 'r', encoding='utf-8') as f:
        topics = json.load(f)
    return topics


def get_all_task_regime_combinations(domain: str, regimes_data: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Get all task-regime combinations for a domain.
    
    Args:
        domain: Domain type ("research" or "tutoring")
        regimes_data: Loaded regimes data from all_pref_regimes.json
    
    Returns:
        List of (task_name, regime_dict) tuples
    """
    if domain not in regimes_data:
        raise ValueError(f"Domain {domain} not found in regimes data")
    
    domain_regimes = regimes_data[domain]
    combinations = []
    
    for task_name, regimes_list in domain_regimes.items():
        for regime in regimes_list:
            combinations.append((task_name, regime))
    
    return combinations


def generate_cross_session_summary(
    events: List[Dict[str, Any]],
    domain: str,
    task_name: str,
    task_description: str,
    preference_summary: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    sleep_s: float = 0.5,
    num_interactions: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate cross-session summary for a single timeline.
    
    Args:
        events: List of event dictionaries
        domain: Domain type, either "research" or "tutoring"
        task_name: Name of the task
        task_description: Description of the task
        preference_summary: Summary of user preference
        model: Model name to use
        temperature: Temperature parameter
        sleep_s: Sleep seconds between API calls
        num_interactions: Number of interactions to generate
    
    Returns:
        Tuple of (result_dict, interaction_info)
        result_dict: Dict containing interactions and summary
        interaction_info: Dict containing input_messages, raw_output, and tokens
    """
    # Build prompt
    prompt = build_cross_session_prompt(
        events=events,
        domain=domain,
        task_name=task_name,
        task_description=task_description,
        preference_summary=preference_summary,
        num_interactions=num_interactions,
    )
    
    # Prepare messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Retry logic: up to 3 attempts
    max_retries = 3
    retry_delay = 2  # seconds between retries
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Make API request
            content, prompt_tokens, completion_tokens, total_tokens = make_api_request(
                messages=messages,
                model=model,
                response_format_json=True,
                temperature=temperature,
            )
            
            # Extract JSON from response
            result_data = extract_json_from_text(content)
            
            # Validate the response
            if not isinstance(result_data, dict):
                raise ValueError(f"Unexpected response format: expected dict, got {type(result_data)}")
            
            # Validate required fields
            if "interactions" not in result_data:
                raise ValueError("Missing 'interactions' field in response")
            if "summary" not in result_data:
                raise ValueError("Missing 'summary' field in response")
            
            # Validate interactions
            interactions = result_data.get("interactions", [])
            if not isinstance(interactions, list):
                raise ValueError("'interactions' must be a list")
            
            # Extract all valid event_ids from the timeline
            valid_event_ids = {event.get("event_id") for event in events if event.get("event_id")}
            
            # Validate each interaction
            validated_interactions = []
            for interaction in interactions:
                if not isinstance(interaction, dict):
                    continue
                if "event_id" in interaction and "user_request" in interaction and "assistant_response" in interaction:
                    event_id = interaction["event_id"]
                    # Validate event_id exists in the events
                    if event_id not in valid_event_ids:
                        print(f"Warning: Interaction has invalid event_id: {event_id}")
                        continue
                    validated_interactions.append(interaction)
            
            if len(validated_interactions) < num_interactions:
                print(f"Warning: Only {len(validated_interactions)} valid interactions generated, expected {num_interactions}")
            
            result = {
                "interactions": validated_interactions,
                "summary": result_data.get("summary", ""),
            }
            
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
            
            return result, interaction_info
        
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Not the last attempt, wait and retry
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(retry_delay)
            else:
                # Last attempt failed
                print(f"Error generating cross-session summary after {max_retries} attempts: {e}")
                return {
                    "interactions": [],
                    "summary": "",
                }, {
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
    topic_info: Dict[str, Any],
    task_name: str,
    regime: Dict[str, Any],
    domain: str,
    mix_dir: Path,
    output_dir: Path,
    model: str,
    temperature: float,
    sleep_s: float,
    skip_existing: bool,
    verbose: bool,
    lock: threading.Lock,
    num_interactions: int = 5,
) -> Dict[str, Any]:
    """
    Process a single topic to generate cross-session summary.
    """
    try:
        topic = topic_info.get("topic", "")
        subject = topic_info.get("subject", "")
        directory_index = topic_info.get("directory_index", "")
        
        # Load events from mix directory
        events_file = mix_dir / domain / directory_index / "events.json"
        if not events_file.exists():
            with lock:
                print(f"[SKIP] {topic} ({directory_index}) - {task_name} R{regime.get('regime_id', '?')}: events.json not found", flush=True)
            return {
                "topic": topic,
                "directory_index": directory_index,
                "task_name": task_name,
                "regime_id": regime.get("regime_id", ""),
                "success": True,
                "skipped": True,
                "error": "events.json not found",
            }
        
        events = load_events(events_file)
        if not events:
            with lock:
                print(f"[SKIP] {topic} ({directory_index}): no events found", flush=True)
            return {
                "topic": topic,
                "directory_index": directory_index,
                "task_name": task_name,
                "regime_id": regime.get("regime_id", ""),
                "success": True,
                "skipped": True,
                "error": "no events found",
            }
        
        # Extract regime information
        preference_summary = regime.get("preference_summary", "")
        regime_id = regime.get("regime_id", "")
        regime_label = regime.get("regime_label", "")
        
        # Build task description
        task_description = build_task_description(domain, task_name)
        
        # Determine save directory structure:
        # output_dir/domain/directory_index/task_subdir/
        base_save_dir = output_dir / domain / directory_index
        base_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create task subdirectory (lowercase, sanitized)
        safe_task = "".join(c if c.isalnum() or c in (' ', '-', '_', '&') else '_' for c in task_name)
        safe_task = safe_task.lower().replace(' ', '_').replace('&', 'and')
        task_subdir = base_save_dir / safe_task
        task_subdir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames with regime suffix
        output_file = task_subdir / f"cross_session_r{regime_id}.json"
        stats_file = task_subdir / f"cross_session_r{regime_id}_stats.json"
        interactions_file = task_subdir / f"cross_session_r{regime_id}_interactions.json"
        
        # Check if already exists
        if skip_existing and output_file.exists():
            with lock:
                print(f"[SKIP] {topic} ({directory_index}) - {task_name} R{regime_id}: {output_file.name} already exists", flush=True)
            return {
                "topic": topic,
                "directory_index": directory_index,
                "task_name": task_name,
                "regime_id": regime_id,
                "success": True,
                "skipped": True,
            }
        
        # Generate cross-session summary
        if verbose:
            with lock:
                print(f"Processing {topic} ({directory_index})...", flush=True)
        
        result, interaction_info = generate_cross_session_summary(
            events=events,
            domain=domain,
            task_name=task_name,
            task_description=task_description,
            preference_summary=preference_summary,
            model=model,
            temperature=temperature,
            sleep_s=sleep_s,
            num_interactions=num_interactions,
        )
        
        if not result.get("interactions") or not result.get("summary"):
            with lock:
                print(f"[ERR] {topic} ({directory_index}) - {task_name} R{regime_id}: No interactions or summary generated", flush=True)
            return {
                "topic": topic,
                "directory_index": directory_index,
                "task_name": task_name,
                "regime_id": regime_id,
                "success": False,
                "error": "No interactions or summary generated",
            }
        
        # Add metadata to result
        result["metadata"] = {
            "topic": topic,
            "subject": subject,
            "directory_index": directory_index,
            "domain": domain,
            "task_name": task_name,
            "regime_id": regime_id,
            "regime_label": regime_label,
            "preference_summary": preference_summary,
            "num_interactions": len(result["interactions"]),
        }
        
        # Update token statistics
        tokens = interaction_info.get("tokens", {})
        
        # Save result
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Save stats for this run
            run_stats = {
                "prompt_tokens": tokens.get("prompt_tokens", 0),
                "completion_tokens": tokens.get("completion_tokens", 0),
                "total_tokens": tokens.get("total_tokens", 0),
                "num_interactions": len(result["interactions"]),
                "task_name": task_name,
                "regime_id": regime_id,
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(run_stats, f, indent=2, ensure_ascii=False)
            
            # Save interactions for this run
            with open(interactions_file, 'w', encoding='utf-8') as f:
                json.dump([interaction_info], f, indent=2, ensure_ascii=False)
            
            with lock:
                print(
                    f"[OK] {topic} ({directory_index}): Generated. Interactions={len(result['interactions'])}, Tokens={tokens.get('total_tokens', 0)}, Task={task_name}, Regime={regime_id}",
                    flush=True,
                )
            
            return {
                "topic": topic,
                "directory_index": directory_index,
                "task_name": task_name,
                "regime_id": regime_id,
                "success": True,
                "skipped": False,
                "num_interactions": len(result["interactions"]),
                "tokens": tokens,
            }
        except Exception as e:
            with lock:
                print(f"[ERR] {topic} ({directory_index}) - {task_name} R{regime_id}: Error saving files: {e}", flush=True)
            return {
                "topic": topic,
                "directory_index": directory_index,
                "task_name": task_name,
                "regime_id": regime_id,
                "success": False,
                "error": f"Error saving files: {e}",
            }
    
    except Exception as e:
            with lock:
                print(f"[ERR] {topic_info.get('topic', 'Unknown')} - {task_name} R{regime.get('regime_id', '?')}: {e}", flush=True)
            return {
                "topic": topic_info.get("topic", "Unknown"),
                "directory_index": topic_info.get("directory_index", ""),
                "task_name": task_name,
                "regime_id": regime.get("regime_id", ""),
                "success": False,
                "error": str(e),
            }


def process_domain(
    domain: str,
    topics_file: Path,
    mix_dir: Path,
    regimes_file: Path,
    output_dir: Path,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    sleep_s: float = 0.5,
    skip_existing: bool = True,
    verbose: bool = False,
    start_from: int = 0,
    limit: Optional[int] = None,
    max_workers: int = 4,
    num_interactions: int = 5,
):
    """
    Process all topics in a domain to generate cross-session summaries.
    For each topic, generates summaries for all task-regime combinations.
    
    Args:
        domain: Domain name ("research" or "tutoring")
        topics_file: Path to cross-session topics JSON file
        mix_dir: Path to mix directory containing events
        regimes_file: Path to all_pref_regimes.json file
        output_dir: Output directory for results
        model: Model name to use
        temperature: Temperature parameter
        sleep_s: Sleep seconds between API calls
        skip_existing: If True, skip topics that already have output files
        verbose: If True, print progress for each topic
        start_from: Index in the sorted topics to start from (0-based)
        limit: Limit of topics to process from start_from
        max_workers: Maximum number of concurrent threads
        num_interactions: Number of interactions to generate per topic-task-regime combination
    """
    # Load topics
    all_topics_dict = load_cross_session_topics(topics_file)
    
    # Flatten topics into a list
    all_topics = []
    for subject, topic_list in all_topics_dict.items():
        for topic_info in topic_list:
            topic_info["subject"] = subject  # Ensure subject is set
            all_topics.append(topic_info)
    
    if not all_topics:
        print(f"Warning: No topics found in {topics_file}")
        return
    
    # Apply start_from and limit
    if limit is not None:
        topics = all_topics[start_from:start_from + limit]
    else:
        topics = all_topics[start_from:]
    
    # Load all preference regimes
    all_regimes = load_all_pref_regimes(regimes_file)
    
    # Get all task-regime combinations for this domain
    task_regime_combinations = get_all_task_regime_combinations(domain, all_regimes)
    
    # Calculate total API calls
    total_combinations = len(topics) * len(task_regime_combinations)
    total_api_calls = total_combinations  # Each combination requires one API call
    
    print(f"Found {len(all_topics)} total topics in {topics_file}")
    print(f"Processing {len(topics)} topics (from index {start_from})")
    print(f"Found {len(task_regime_combinations)} task-regime combinations for {domain} domain")
    print(f"Total combinations to process: {total_combinations}")
    print(f"Estimated total API calls: {total_api_calls:,}")
    if skip_existing:
        print(f"  (Note: Some calls may be skipped if output files already exist)")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model}")
    print(f"Max workers: {max_workers}")
    print(f"Skip existing: {skip_existing}")
    print(f"Num interactions per combination: {num_interactions}")
    print()
    
    lock = threading.Lock()
    results: List[Dict[str, Any]] = []
    
    # Create all combinations: (topic_info, task_name, regime)
    all_combinations = []
    for topic_info in topics:
        for task_name, regime in task_regime_combinations:
            all_combinations.append((topic_info, task_name, regime))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_single,
                topic_info=topic_info,
                task_name=task_name,
                regime=regime,
                domain=domain,
                mix_dir=mix_dir,
                output_dir=output_dir,
                model=model,
                temperature=temperature,
                sleep_s=sleep_s,
                skip_existing=skip_existing,
                verbose=verbose,
                lock=lock,
                num_interactions=num_interactions,
            )
            for topic_info, task_name, regime in all_combinations
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
    print(f"Total topics found: {len(all_topics)}")
    print(f"Topics to process: {len(topics)}")
    print(f"Successful: {len(successful)} (skipped: {len(skipped)})")
    print(f"Failed: {len(failed)}")
    print(f"\nToken Statistics:")
    print(f"  Total prompt tokens: {total_prompt_tokens:,}")
    print(f"  Total completion tokens: {total_completion_tokens:,}")
    print(f"  Total tokens: {total_tokens:,}")
    
    if failed:
        print(f"\nFailed runs:")
        for r in failed:
            print(f"  - {r.get('topic', 'Unknown')} ({r.get('directory_index', 'N/A')}) - {r.get('task_name', 'N/A')} R{r.get('regime_id', 'N/A')}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate cross-session context summaries.")
    parser.add_argument(
        "--domain",
        type=str,
        choices=["research", "tutoring"],
        required=True,
        help="Domain type: research or tutoring",
    )
    parser.add_argument(
        "--topics_file",
        type=str,
        required=True,
        help="Path to cross-session topics JSON file (e.g., cross_session_topics_research.json)",
    )
    parser.add_argument(
        "--mix_dir",
        type=str,
        default="examples/mix",
        help="Path to mix directory containing events",
    )
    parser.add_argument(
        "--regimes_file",
        type=str,
        required=True,
        help="Path to all_pref_regimes.json file (e.g., all_pref_regimes.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for results. If relative, will be created relative to script directory.",
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
        help="Sleep seconds between API calls (per worker).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skip topics that already have output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-topic progress.",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Index in the sorted topics to start from (0-based).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit of topics to process from start_from.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of concurrent threads.",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=5,
        help="Number of interactions to generate per topic.",
    )
    
    args = parser.parse_args()
    
    topics_file = Path(args.topics_file)
    if not topics_file.exists():
        raise ValueError(f"Topics file does not exist: {topics_file}")
    
    mix_dir = Path(args.mix_dir)
    if not mix_dir.exists():
        raise ValueError(f"Mix directory does not exist: {mix_dir}")
    
    regimes_file = Path(args.regimes_file)
    if not regimes_file.exists():
        raise ValueError(f"Regimes file does not exist: {regimes_file}")
    
    # Resolve output_dir if provided
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        # If relative, make it relative to the script's directory
        script_dir = Path(__file__).parent
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    process_domain(
        domain=args.domain,
        topics_file=topics_file,
        mix_dir=mix_dir,
        regimes_file=regimes_file,
        output_dir=output_dir,
        model=args.model,
        temperature=args.temperature,
        sleep_s=args.sleep_s,
        skip_existing=args.skip_existing,
        verbose=args.verbose,
        start_from=args.start_from,
        limit=args.limit,
        max_workers=args.max_workers,
        num_interactions=args.num_interactions,
    )

