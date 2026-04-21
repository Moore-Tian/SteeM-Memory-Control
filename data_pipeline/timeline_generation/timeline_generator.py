"""
timeline_generator.py

End-to-end logic for automatically generating long-term task timelines
for multiple domains (research, engineering, academic writing, teaching).

- Defines domain configs (event_types + required artifacts).
- Calls OpenAI chat.completions API with a JSON-oriented prompt.
- Parses the resulting list of events into Python objects.
- Provides simple validation utilities.

You can adapt this as the first step of your data generation pipeline.
"""

from __future__ import annotations

import json
import os
import re
import hmac
import hashlib
import time
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import random
from tqdm import tqdm

from timeline_prompts import build_research_next_event_prompt, build_tutoring_next_event_prompt
from all_topics import RESEARCH_TOPIC, TUTORING_TOPIC
from utils import make_api_request, save_generation_results, extract_json_from_text



# ======================
# Data structures
# ======================

@dataclass
class Event:
    event_id: str
    time_index: int
    domain: str
    topic: str
    subject: str
    event_type: str
    description: str
    required_artifacts: List[str]
    generated_artifacts: List[str]
    reason: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Event":
        # Basic robustness: provide defaults if missing
        return Event(
            event_id=d.get("event_id", ""),
            time_index=int(d.get("time_index", 0)),
            domain=d.get("domain", ""),
            topic=d.get("topic", ""),
            subject=d.get("subject", ""),
            event_type=d.get("event_type", ""),
            description=d.get("description", ""),
            required_artifacts=d.get("required_artifacts", []) or [],
            generated_artifacts=d.get("generated_artifacts", []) or [],
            reason=d.get("reason", ""),
        )

    @staticmethod
    def print_event(ev: "Event") -> None:
        print(f"[{ev.time_index:02d}] {ev.event_type}")
        print(f"  desc: {ev.description}")
        if ev.required_artifacts:
            print(f"  required_artifacts: {ev.required_artifacts}")
        if ev.generated_artifacts:
            print(f"  generated_artifacts: {ev.generated_artifacts}")
        if ev.reason:
            print(f"  reason: {ev.reason}")


@dataclass
class DomainConfig:
    name: str
    event_types: List[Dict[str, str]]      # [{'id': 'proposal', 'description': '...'}, ...]
    artifacts: List[str]     # fields that must be seen at least once


# ======================
# Domain configurations
# ======================

DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    "research": DomainConfig(
        name="research",
        event_types=[
            {"id": "proposal", "description": "Define research goals and an initial high-level plan."},
            {"id": "scheme_design", "description": "Design an initial experimental or methodological scheme."},
            {"id": "pilot_experiments", "description": "Run small-scale pilot experiments to sanity-check the idea."},
            {"id": "main_experiments", "description": "Run main experiments, refine the scheme and collect results."},
            {"id": "analysis", "description": "Analyze results and summarize current research status."},
            {"id": "writing", "description": "Draft the paper and refine its structure in preparation for submission."},
        ],
        artifacts=[
            "research_plan",     # for Plan/Design, Revise, Critique
            "research_goals",    # for Plan/Design, Revise, Critique
            "experiment_results",  # for Critique, Tutoring
            "method_scheme",   # for Plan/Design, Revise, Critique, Tutoring
            "paper_paragraph",   # for Plan/Design, Revise, Critique, Tutoring
        ],
    ),
    "tutoring": DomainConfig(
        name="tutoring",
        event_types=[
            {"id": "objective_clarification", "description": "Clarify learning objectives, constraints, and self-reported background; optionally draft an initial study plan."},
            {"id": "plan_milestones", "description": "Define or revise milestones and checkpoints, and produce or update a staged study plan."},
            {"id": "lesson", "description": "Teach a concept/method/pattern relevant to current objectives or recent difficulties, and update reusable teaching notes."},
            {"id": "practice", "description": "Assign or run targeted practice and record attempts or intermediate steps in a structured way."},
            {"id": "review", "description": "Review practice attempts, identify mistakes and misconceptions, and produce actionable feedback and corrections."},
            {"id": "materials_revision", "description": "Revise teaching notes, study plan wording, or reusable templates/examples in response to learner needs and observed issues, improving clarity and reusability."},
            {"id": "end", "description": "Terminate the timeline when objectives are substantially met and progress is stable."},
        ],
        artifacts=[
            "learning_objectives",  # Plan/Design: set targets & constraints; Critique/Evaluate: check alignment to goals
            "study_plan",           # Plan/Design: (re)plan milestones/checkpoints; Revise/Generate: rewrite/clarify plan wording
            "teaching_notes",       # Concept explanation: primary knowledge artifact; Revise/Generate: refine explanations/examples/structure
            "practice_record",      # Critique/Evaluate: judge correctness/applicability using outcomes; Plan/Design: adjust plan based on performance signals
            "practice_attempts",    # Critique/Evaluate: evaluate step-by-step reasoning and locate errors; supports targeted Lesson content selection
            "feedback_summary",     # Critique/Evaluate: consolidated critique + corrections; Plan/Design: next-step guidance; Revise/Generate: rewrite feedback into checklists/templates
        ],
    )
}

# ======================
# Timeline generation
# ======================
def _build_domain_prompt(
    domain: str,
    topic: str,
    existing_events: Optional[List[Dict[str, Any]]] = None,
    next_index: Optional[int] = None,
    end_available: bool = False,
) -> List[Dict[str, str]]:
    """
    Dispatch to the appropriate domain-specific prompt builder.
    Returns messages list for API call.
    """
    if domain == "research":
        # Use new sequential generation prompt if parameters are provided
        if existing_events is not None and next_index is not None:
            return build_research_next_event_prompt(
                topic=topic,
                existing_events=existing_events,
                next_index=next_index,
                end_available=end_available
            )
        else:
            raise ValueError(
                "For 'research' domain, sequential generation requires "
                "existing_events and next_index parameters."
            )
    elif domain == "engineering":
        # Other domains not implemented yet for sequential generation
        raise NotImplementedError(
            f"Domain '{domain}' is not yet supported. "
            "Sequential generation is currently only implemented for 'research' domain."
        )
    elif domain == "writing":
        raise NotImplementedError(
            f"Domain '{domain}' is not yet supported. "
            "Sequential generation is currently only implemented for 'research' domain."
        )
    elif domain == "tutoring":
        if existing_events is not None and next_index is not None:
            return build_tutoring_next_event_prompt(
                topic=topic,
                existing_events=existing_events,
                next_index=next_index,
                end_available=end_available
            )
        else:
            raise ValueError(
                "For 'tutoring' domain, sequential generation requires "
                "existing_events and next_index parameters."
            )
    else:
        raise ValueError(f"Unknown domain: {domain}")


def generate_timeline_for_domain(
    model: str,
    domain: str,
    topic: str,
    subject: str,
    min_events: int = 20,
    max_events: int = 30,
    response_format_json: bool = True,
    save_output: bool = True,
    output_dir: str = "output",
    run_id: int = 0,
    verbose: bool = False,
    request_type: str = "openai",
    appid: Optional[str] = None,
    appkey: Optional[str] = None,
) -> tuple[List["Event"], Optional[Dict[str, str]]]:
    """
    Generate a timeline for a given domain.

    For the 'research' domain, we generate ONE event per model call:
    in each step, we provide all previous events as context and ask
    the model to output exactly one new event JSON object.
    
    Includes retry mechanism: if any error occurs during generation,
    the entire timeline generation will be retried once.

    Returns:
        tuple: (events, saved_files_dict) where saved_files_dict is None if save_output=False
    """
    if domain not in ["research", "tutoring"]:
        raise NotImplementedError("Sequential per-event generation is currently implemented only for 'research' and 'tutoring' domains.")

    events: List[Event] = []
    interactions: List[Dict[str, Any]] = []  # 保存每一轮的完整交互过程

    for idx in range(1, max_events + 1):
        # 为每个 event 的生成添加重试机制
        max_retries = 3  # 最多重试3次（总共尝试4次）
        last_exception = None
        event_generated = False

        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (总共4次尝试)
            try:
                # 将已有 events 转成简化 dict 传给 prompt 构造器
                existing_dicts: List[Dict[str, Any]] = [asdict(ev) for ev in events]

                messages = _build_domain_prompt(
                    domain=domain,
                    topic=topic,
                    existing_events=existing_dicts,
                    next_index=idx,
                    end_available=idx > min_events
                )

                if verbose:
                    if attempt > 0:
                        print(f"[Run {run_id}] Event {idx} - Retry attempt {attempt + 1}")
                    else:
                        print(f"[Run {run_id}] Event {idx} - Topic: {topic}")

                # Use unified API request function
                content, prompt_tokens, completion_tokens, total_tokens = make_api_request(
                    messages=messages,
                    model=model,
                    request_type=request_type,
                    response_format_json=response_format_json,
                    temperature=0.7,
                    appid=appid,
                    appkey=appkey,
                )

                # 暂停 1s
                time.sleep(1)

                # 记录这一轮的交互过程
                interaction = {
                    "step": idx,
                    "input_messages": messages,
                    "raw_output": content,
                    "tokens": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }
                interactions.append(interaction)

                try:
                    parsed = extract_json_from_text(content)
                except ValueError as e:
                    raise ValueError(f"Model output at step {idx} is not valid JSON: {e}\nRaw content: {content}")

                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"Model JSON output at step {idx} must be a single JSON object (event), "
                        f"but got type {type(parsed)}."
                    )

                ev = Event.from_dict(parsed)
                
                # 确保 domain 和 time_index 正确
                ev.domain = domain
                ev.time_index = idx
                ev.topic = topic
                ev.subject = subject
                # 如果没有 event_id，生成一个
                if not ev.event_id:
                    ev.event_id = f"e_{idx:02d}"
                events.append(ev)
                
                event_generated = True
                break  # 成功生成 event，跳出重试循环

            except Exception as e:
                last_exception = e
                # 如果这是最后一次尝试，不再重试
                if attempt == max_retries:
                    break
                # 否则等待一下再重试
                if verbose:
                    print(f"[Run {run_id}] Event {idx} generation failed, retrying in 2 seconds...")
                time.sleep(2)  # 重试前等待2秒

        # 如果所有尝试都失败，抛出异常
        if not event_generated:
            raise RuntimeError(
                f"Failed to generate event {idx} after {max_retries + 1} attempts for topic '{topic}' (run_id={run_id}). "
                f"Last error: {last_exception}"
            ) from last_exception

        # 如果生成了 end event，提前结束
        if ev.event_type == "end":
            break

    # 计算总的 token 使用情况
    total_prompt_tokens = sum(interaction.get("tokens", {}).get("prompt_tokens", 0) for interaction in interactions)
    total_completion_tokens = sum(interaction.get("tokens", {}).get("completion_tokens", 0) for interaction in interactions)
    total_tokens = sum(interaction.get("tokens", {}).get("total_tokens", 0) for interaction in interactions)

    # 保存结果
    saved_files = None
    if save_output:
        saved_files = save_generation_results(
            output_dir=output_dir,
            domain=domain,
            events=events,
            interactions=interactions,
            run_id=run_id,
            token_stats={
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "num_steps": len(interactions),
            },
        )

    return events, saved_files

# ======================
# Validation utilities
# ======================

def check_artifacts_coverage(
    events: List[Event],
    domain: str,
) -> Dict[str, bool]:
    """
    Check whether required artifacts for a domain appear at least once (non-null).

    Returns a dict {artifact_name: covered_bool}.
    """
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain: {domain}")

    cfg = DOMAIN_CONFIGS[domain]
    coverage: Dict[str, bool] = {a: False for a in cfg.artifacts}

    for ev in events:
        # Check both required_artifacts and generated_artifacts lists
        all_artifacts = set(ev.required_artifacts + ev.generated_artifacts)
        for a in cfg.artifacts:
            if a in all_artifacts:
                coverage[a] = True

    return coverage


def print_timeline_summary(events: List[Event]) -> None:
    """Quick human-readable summary for debugging."""
    for ev in events:
        print(f"[{ev.time_index:02d}] {ev.event_type}")
        print(f"  desc: {ev.description}")
        if ev.required_artifacts:
            print(f"  required_artifacts: {ev.required_artifacts}")
        if ev.generated_artifacts:
            print(f"  generated_artifacts: {ev.generated_artifacts}")
        if ev.reason:
            print(f"  reason: {ev.reason}")
        print()


# ======================
# Example usage (manual test)
# ======================

def generate_single_timeline(
    topic_info: Dict[str, str],
    run_id: int,
    domain: str,
    model: str,
    min_events: int,
    max_events: int,
    output_dir: str,
    lock: threading.Lock,
    request_type: str = "openai",
    appid: Optional[str] = None,
    appkey: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a single timeline for a topic. This function is called by each thread.
    """
    topic = topic_info["topic"]
    subject = topic_info.get("subject", "")
    
    try:
        events, saved_files = generate_timeline_for_domain(
            model=model,
            domain=domain,
            topic=topic,
            subject=subject,
            min_events=min_events,
            max_events=max_events,
            save_output=True,
            output_dir=output_dir,
            run_id=run_id,
            verbose=False,
            request_type=request_type,
            appid=appid,
            appkey=appkey,
        )
        
        result = {
            "run_id": run_id,
            "topic": topic,
            "subject": subject,
            "success": True,
            "num_events": len(events),
            "saved_files": saved_files,
        }
        
        with lock:
            print(f"[✓] Run {run_id}: {topic} - Generated {len(events)} events, flush=True")
        
        return result
    except Exception as e:
        with lock:
            print(f"[✗] Run {run_id}: {topic} - Error: {str(e)}, flush=True")
        return {
            "run_id": run_id,
            "topic": topic,
            "subject": subject,
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    """
    Batch generation with multi-threading.
    
    Example:
        $ python timeline_generator.py research --max_workers 4
    """
    import argparse

    parser = argparse.ArgumentParser(description="Batch generate project timelines for different domains.")
    parser.add_argument(
        "domain",
        type=str,
        choices=list(DOMAIN_CONFIGS.keys()),
        help="Domain: research | engineering | tutoring | writing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model name (e.g., gpt-4o-mini).",
    )
    parser.add_argument(
        "--min_events",
        type=int,
        default=20,
        help="Minimum number of events to generate.",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=30,
        help="Maximum number of events to generate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save all outputs and intermediate processes.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of concurrent threads.",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start from this index in the topics list (for resuming).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of topics to process (for testing).",
    )
    parser.add_argument(
        "--request_type",
        type=str,
        choices=["openai", "tencent"],
        default="openai",
        help="API request type: 'openai' or 'tencent'.",
    )
    parser.add_argument(
        "--appid",
        type=str,
        default=None,
        help="Tencent API appid (required for tencent request type).",
    )
    parser.add_argument(
        "--appkey",
        type=str,
        default=None,
        help="Tencent API appkey (required for tencent request type).",
    )
    args = parser.parse_args()
    
    # Validate tencent request type parameters
    if args.request_type == "tencent":
        if not args.appid or not args.appkey:
            raise ValueError("--appid and --appkey are required when --request_type is 'tencent'")

    # Load topics from all_topics.py
    if args.domain == "research":
        topics = RESEARCH_TOPIC
    elif args.domain == "tutoring":
        topics = TUTORING_TOPIC
    else:
        raise ValueError(f"Domain '{args.domain}' not yet supported. Only 'research' and 'tutoring' are supported.")

    # Apply limits
    if args.limit:
        topics = topics[args.start_from:args.start_from + args.limit]
    else:
        topics = topics[args.start_from:]

    print(f"=== Batch Generation ===")
    print(f"Domain: {args.domain}")
    print(f"Total topics: {len(topics)}")
    print(f"Max workers: {args.max_workers}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Thread-safe lock for printing
    print_lock = threading.Lock()

    # Run batch generation with thread pool
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        futures = []
        for idx, topic_info in enumerate(topics, start=args.start_from):
            future = executor.submit(
                generate_single_timeline,
                topic_info=topic_info,
                run_id=idx,
                domain=args.domain,
                model=args.model,
                min_events=args.min_events,
                max_events=args.max_events,
                output_dir=args.output_dir,
                lock=print_lock,
                request_type=args.request_type,
                appid=args.appid,
                appkey=args.appkey,
            )
            futures.append(future)

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating timelines"):
            result = future.result()
            results.append(result)

    # Print summary
    print("\n=== Generation Summary ===")
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"Total: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed topics:")
        for r in failed:
            print(f"  - Run {r['run_id']}: {r['topic']} - {r.get('error', 'Unknown error')}")