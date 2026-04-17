#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speed test for your prompt + DeepSeek model.

Examples:
1) Single job, draft only, 5 runs sequential:
python speed_test_prompt.py --jobs /path/jobs.jsonl --job_id tutoring__95__t12 --n 5 --concurrency 1

2) Draft+refine, 10 runs with concurrency 2:
python speed_test_prompt.py --jobs /path/jobs.jsonl --job_id tutoring__95__t12 --n 10 --concurrency 2 --refine

Notes:
- This script measures end-to-end latency of each request (HTTP round-trip).
- It also prints p50/p95/avg and slowest samples.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import dotenv
dotenv.load_dotenv()

DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"


def _read_jobs(path: Path) -> List[Dict[str, Any]]:
    jobs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
    return jobs

def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _events_prefix(events: List[Dict[str, Any]], t: int) -> List[Dict[str, Any]]:
    return [e for e in events if int(e.get("time_index", 0)) <= t]

def _allowed_mem_types(domain: str) -> List[str]:
    if domain == "research":
        return [
            "project_state","research_goal","method_scheme","past_experiment","failure_case",
            "review_feedback","writing_draft","resource_constraint","decision_rationale",
        ]
    return [
        "learner_profile","learning_objective","learning_misconception","past_mistake","study_plan",
        "practice_result","feedback_preference","resource_material","progress_summary",
    ]

def _build_draft_prompt(domain: str, events: List[Dict[str, Any]], use_json_mode: bool) -> List[Dict[str, str]]:
    allowed = _allowed_mem_types(domain)
    top_level_spec = (
        'Output MUST be a single JSON object with exactly one key "items". The value of "items" MUST be a JSON array.'
        if use_json_mode else
        "Output MUST be a single JSON array."
    )
    system = f"""
You are generating atomic memory items from a prefix of a long-term timeline.

Output format (STRICT):
- {top_level_spec} Output NO extra text. No markdown fences.
- Use EXACTLY key name "items".
- Allowed mem_type values: {allowed}

Content:
- Each item content MUST be 3-4 sentences.
- Include at least two of: Outcome / Constraint / Rationale / Next-action / Artifact reference.
- You MAY add bounded operational detail (Playbook/Next-action) implied by events, but do NOT invent event-specific facts.

Return JSON only.
""".strip()

    user = {
        "role": "user",
        "content": "Events prefix:\n\n" + json.dumps(events, ensure_ascii=False, indent=2)
    }
    return [{"role": "system", "content": system}, user]

def _post_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    use_json_mode: bool,
    session: requests.Session,
) -> Tuple[float, int]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    t0 = time.perf_counter()
    r = session.post(DEEPSEEK_CHAT_URL, headers=headers, json=payload, timeout=timeout_sec)
    latency = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()

    # best-effort token usage
    usage = data.get("usage", {}) or {}
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    return latency, total_tokens

def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=str, default="evaluation/jobs.example.jsonl")
    ap.add_argument("--job_id", type=str, default="", help="pick a specific job_id; otherwise use the first job")
    ap.add_argument("--model", type=str, default="deepseek-reasoner")
    ap.add_argument("--api_key_env", type=str, default=os.getenv("DEEPSEEK_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--json_mode", action="store_true", default=True)
    ap.add_argument("--no_json_mode", action="store_true")

    ap.add_argument("--n", type=int, default=5, help="number of runs")
    ap.add_argument("--concurrency", type=int, default=5, help="parallel requests")
    args = ap.parse_args()

    if args.no_json_mode:
        args.json_mode = False

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Please export {args.api_key_env}=...")

    jobs = _read_jobs(Path(args.jobs))
    if not jobs:
        raise RuntimeError("jobs.jsonl is empty")

    job = None
    if args.job_id:
        for j in jobs:
            if j.get("job_id") == args.job_id:
                job = j
                break
        if job is None:
            raise RuntimeError(f"job_id not found: {args.job_id}")
    else:
        job = jobs[0]

    domain = job["domain"]
    t = int(job["time_index"])
    events = _read_json(Path(job["events_path"]))
    prefix = _events_prefix(events, t)

    messages = _build_draft_prompt(domain, prefix, use_json_mode=args.json_mode)

    # run
    latencies: List[float] = []
    tokens: List[int] = []

    def _one_run() -> Tuple[float, int]:
        sess = requests.Session()
        return _post_chat(
            api_key=api_key,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout,
            use_json_mode=args.json_mode,
            session=sess,
        )

    t_all0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as ex:
        futs = [ex.submit(_one_run) for _ in range(int(args.n))]
        for fut in as_completed(futs):
            lat, tok = fut.result()
            latencies.append(lat)
            tokens.append(tok)
    t_all = time.perf_counter() - t_all0

    lat_sorted = sorted(latencies)
    p50 = _percentile(lat_sorted, 0.50)
    p95 = _percentile(lat_sorted, 0.95)
    avg = statistics.mean(latencies) if latencies else float("nan")
    mx = max(latencies) if latencies else float("nan")
    mn = min(latencies) if latencies else float("nan")
    tok_avg = statistics.mean(tokens) if tokens else 0

    print("=== Speed Test Summary ===")
    print(f"job_id: {job.get('job_id')}")
    print(f"domain: {domain}, time_index: {t}, prefix_events: {len(prefix)}")
    print(f"model: {args.model}, json_mode: {args.json_mode}, max_tokens: {args.max_tokens}, timeout: {args.timeout}")
    print(f"runs: {args.n}, concurrency: {args.concurrency}")
    print(f"total wall time: {t_all:.3f}s")
    print(f"latency (sec): min={mn:.3f}  p50={p50:.3f}  p95={p95:.3f}  avg={avg:.3f}  max={mx:.3f}")
    print(f"avg total_tokens (best-effort, 0 if unavailable): {tok_avg:.1f}")
    print("slowest samples:", [f"{x:.3f}" for x in sorted(latencies, reverse=True)[:min(5, len(latencies))]])

if __name__ == "__main__":
    main()