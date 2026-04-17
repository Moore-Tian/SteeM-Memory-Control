#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 (Parallel + Detailed + Richer):
For each job (domain, case, time_index), read events.json, take prefix (time_index <= t),
call DeepSeek to generate atomic memory items, write to memory_bank.jsonl.

Enhancements:
- tqdm progress for parallel jobs
- detailed QA log per job (request/response/parse/validate + latency)
- robust JSON extraction and validation
- default JSON mode: response_format={"type":"json_object"} and enforce {"items":[...]}
- parallel multi-requests with ThreadPoolExecutor
- richer memory content:
  - allow "reasonable elaboration" as operationalized checklists/templates/rules derived from events
  - optional 2-stage generation (draft -> refine)

Recommended usage:
python evaluation/make_test_data/generate_memory_bank.py \
    --jobs evaluation/jobs.example.jsonl \
  --model deepseek-reasoner \
  --workers 4 \
  --min_items 16 --max_items 26 \
  --refine \
  --overwrite \
  --min_content_chars 140
"""

from __future__ import annotations

import argparse
import json
import os
import time
import random
import traceback
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv
dotenv.load_dotenv()

import requests

try:
    from tqdm import tqdm
except Exception as e:
    raise RuntimeError("Missing dependency tqdm. Please install with: pip install tqdm") from e


DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"

# -----------------------------
# IO helpers
# -----------------------------
def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _read_jobs(path: Path) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
    return jobs

def _events_prefix(events: List[Dict[str, Any]], t: int) -> List[Dict[str, Any]]:
    return [e for e in events if int(e.get("time_index", 0)) <= t]

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------
# Global rate limiter (avoid 429 -> retries -> very slow)
# -----------------------------
class _RateLimiter:
    """
    Simple global QPS limiter shared across threads.
    If qps <= 0: disabled.
    """
    def __init__(self, qps: float):
        self.qps = float(qps)
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        if self.qps <= 0:
            return
        min_interval = 1.0 / self.qps
        with self._lock:
            now = time.perf_counter()
            if self._next_time <= now:
                self._next_time = now + min_interval
                return
            sleep_s = self._next_time - now
            self._next_time += min_interval
        if sleep_s > 0:
            time.sleep(sleep_s)


# -----------------------------
# DeepSeek client
# -----------------------------
def _post_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    use_json_mode: bool,
    session: requests.Session,
    limiter: Optional[_RateLimiter],
) -> Tuple[Dict[str, Any], float]:
    """
    Returns (response_json, latency_seconds)
    """
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

    if limiter is not None:
        limiter.acquire()

    t0 = time.perf_counter()
    r = session.post(DEEPSEEK_CHAT_URL, headers=headers, json=payload, timeout=timeout_sec)
    latency = time.perf_counter() - t0

    r.raise_for_status()
    data = r.json()
    data["_request_payload"] = payload
    return data, latency


# -----------------------------
# Robust JSON extraction
# -----------------------------
def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        last = s.rfind("```")
        if first_nl != -1 and last != -1 and last > first_nl:
            return s[first_nl:last].strip()
    return s

def _find_balanced_json_substrings(text: str, max_candidates: int = 12) -> List[str]:
    candidates: List[str] = []
    n = len(text)

    starts: List[int] = []
    for i, ch in enumerate(text):
        if ch in "{[":
            starts.append(i)
        if len(starts) >= 60:
            break

    def match_from(start: int) -> Optional[int]:
        open_ch = text[start]
        close_ch = "}" if open_ch == "{" else "]"
        stack = [open_ch]
        in_str = False
        esc = False
        for j in range(start + 1, n):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            else:
                if c == '"':
                    in_str = True
                    continue
                if c == open_ch:
                    stack.append(open_ch)
                elif c == close_ch:
                    stack.pop()
                    if not stack:
                        return j
        return None

    for st in starts:
        ed = match_from(st)
        if ed is None:
            continue
        frag = text[st:ed + 1].strip()
        candidates.append(frag)
        if len(candidates) >= max_candidates:
            break

    return candidates

def _looks_like_items_object(val: Any) -> bool:
    return isinstance(val, dict) and ("items" in val) and isinstance(val.get("items"), (list, dict))

def _robust_json_loads(raw: str) -> Any:
    if raw is None:
        raise ValueError("raw content is None")
    s0 = raw.strip()
    if not s0:
        raise ValueError("empty content")

    # 1) direct
    try:
        return json.loads(s0)
    except Exception:
        pass

    # 2) strip fence
    s1 = _strip_code_fence(s0)
    if s1 != s0:
        try:
            return json.loads(s1)
        except Exception:
            pass

    # 3) candidates
    parsed_candidates: List[Any] = []
    for frag in _find_balanced_json_substrings(s1, max_candidates=14):
        try:
            parsed_candidates.append(json.loads(frag))
        except Exception:
            continue

    for v in parsed_candidates:
        if _looks_like_items_object(v):
            return v
    for v in parsed_candidates:
        if isinstance(v, list):
            return v
    for v in parsed_candidates:
        if isinstance(v, dict):
            return v

    # 4) last resort slice
    for (lch, rch) in [("{", "}"), ("[", "]")]:
        i = s1.find(lch)
        j = s1.rfind(rch)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(s1[i:j + 1])
            except Exception:
                pass

    raise ValueError("failed to parse JSON from model output")


# -----------------------------
# Schema validation / normalization
# -----------------------------
_REQUIRED_KEYS = ["mem_id", "event_id", "time_index", "timestamp", "mem_type", "level", "content", "salience", "related_events"]
_REQUIRED_KEYSET = set(_REQUIRED_KEYS)

def _coerce_items(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "items" in obj:
            v = obj["items"]
            if isinstance(v, list):
                return v
            if isinstance(v, dict) and _REQUIRED_KEYSET.issubset(set(v.keys())):
                return [v]
            return v  # type: ignore
        alt_keys = ["memory_bank", "memories", "memory_items", "data", "results", "output"]
        for k in alt_keys:
            if k in obj:
                return obj[k]  # type: ignore
        if _REQUIRED_KEYSET.issubset(set(obj.keys())):
            return [obj]  # type: ignore
    raise ValueError("Top-level JSON must contain 'items' (or known alternative keys) or be a list.")

def _validate_and_normalize_items(
    obj: Any,
    allowed_mem_types: List[str],
    expect_min: int,
    min_content_chars: int = 0,
) -> List[Dict[str, Any]]:
    items = _coerce_items(obj)
    if not isinstance(items, list):
        raise ValueError("'items' must be a list")
    if len(items) < expect_min:
        raise ValueError(f"Too few memory items: {len(items)} < {expect_min}")

    norm: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"Item #{idx} is not an object")
        for k in _REQUIRED_KEYS:
            if k not in it:
                raise ValueError(f"Item #{idx} missing key: {k}")

        it2 = dict(it)
        it2["time_index"] = int(it2["time_index"]) if it2["time_index"] is not None else it2["time_index"]
        if it2["timestamp"] is not None and not isinstance(it2["timestamp"], str):
            it2["timestamp"] = str(it2["timestamp"])

        if it2["mem_type"] not in allowed_mem_types:
            raise ValueError(f"Item #{idx} invalid mem_type: {it2['mem_type']}")
        if it2["level"] not in ["core", "summary", "episodic"]:
            raise ValueError(f"Item #{idx} invalid level: {it2['level']}")

        if not isinstance(it2["content"], str) or not it2["content"].strip():
            raise ValueError(f"Item #{idx} empty content")

        # soft richness signal (do not fail)
        _ = (min_content_chars > 0 and len(it2["content"].strip()) < min_content_chars)

        try:
            sal = float(it2["salience"])
        except Exception:
            raise ValueError(f"Item #{idx} salience not float: {it2['salience']}")
        if not (0.0 <= sal <= 1.0):
            sal = max(0.0, min(1.0, sal))
        it2["salience"] = sal

        if not isinstance(it2["related_events"], list):
            raise ValueError(f"Item #{idx} related_events must be list")
        it2["related_events"] = [str(x) for x in it2["related_events"]]

        it2["mem_id"] = str(it2["mem_id"])
        it2["event_id"] = str(it2["event_id"])

        norm.append(it2)

    return norm


# -----------------------------
# Prompt builder
# -----------------------------
def _allowed_mem_types(domain: str) -> List[str]:
    if domain == "research":
        return [
            "project_state",
            "research_goal",
            "method_scheme",
            "past_experiment",
            "failure_case",
            "review_feedback",
            "writing_draft",
            "resource_constraint",
            "decision_rationale",
        ]
    return [
        "learner_profile",
        "learning_objective",
        "learning_misconception",
        "past_mistake",
        "study_plan",
        "practice_result",
        "feedback_preference",
        "resource_material",
        "progress_summary",
    ]

def _build_output_example(domain: str) -> str:
    if domain == "research":
        example_mem_type = "past_experiment"
        example_content = (
            "Pilot run improved accuracy qualitatively but increased latency beyond the target threshold stated in planning artifacts. "
            "Constraint: any next iteration must keep latency under the agreed ceiling or adopt a lighter variant. "
            "Next-action: run an ablation to locate which component causes the latency regression and document the decision rationale."
        )
    else:
        example_mem_type = "practice_result"
        example_content = (
            "Learner used the revised notes to quantify BATNA costs using the buckets explicitly introduced (time/resources/reputation) and produced a more realistic alternative than before. "
            "Rationale: the previous review flagged missing quantification as the failure cause. "
            "Playbook: for the next role-play, write one concrete estimate per bucket and prepare two tradable concessions before starting the conversation."
        )

    return f"""
Output example (MUST follow):
{{
  "items": [
    {{
      "mem_id": "m_01",
      "event_id": "e_03",
      "time_index": 3,
      "timestamp": null,
      "mem_type": "{example_mem_type}",
      "level": "summary",
      "content": "{example_content}",
      "salience": 0.7,
      "related_events": ["e_02"]
    }}
  ]
}}
""".strip()

def _build_draft_prompt(
    domain: str,
    events: List[Dict[str, Any]],
    target_min_items: int,
    target_max_items: int,
    use_json_mode: bool,
    allow_elaboration: bool,
    content_sent_min: int,
    content_sent_max: int,
) -> Tuple[List[Dict[str, str]], List[str]]:
    allowed = _allowed_mem_types(domain)
    top_level_spec = (
        'Output MUST be a single JSON object with exactly one key "items". '
        'The value of "items" MUST be a JSON array. Output NO extra text.'
        if use_json_mode else
        "Output MUST be a single JSON array, with NO extra text."
    )

    schema_block = f"""
Schema for each item (follow exactly):
{{
  "mem_id": "m_XX",
  "event_id": "e_YY",
  "time_index": <int>,
  "timestamp": <string or null>,
  "mem_type": <one of ALLOWED_MEM_TYPES>,
  "level": <one of ["core","summary","episodic"]>,
  "content": <{content_sent_min}-{content_sent_max} sentences, specific and decision-relevant>,
  "salience": <float in [0,1]>,
  "related_events": <array of event_id strings, can be []>
}}
""".strip()

    output_example = _build_output_example(domain) if use_json_mode else ""

    # 关键变化：允许“合理扩充”，但限制为 operationalization，不捏造事实
    elaboration_block = ""
    if allow_elaboration:
        elaboration_block = """
Reasonable elaboration (ALLOWED, but bounded):
- You MAY add operational detail that is generally valid and directly implied by the event description,
  such as a short checklist/template/rule-of-thumb that would help future decisions.
- Do NOT present elaborations as things that happened. Do NOT add specific names/numbers unless present in events.
- Prefer phrasing like "Playbook: ..." or "Next-action: ..." for operational detail, but keep it inside content.
""".strip()

    system = f"""
You are generating atomic memory items from a prefix of a long-term timeline.

Output format (STRICT):
- {top_level_spec}
- Use standard JSON escaping rules. Do NOT use markdown fences. Do NOT add any commentary.
- Use EXACTLY the key name "items".
- Allowed mem_type values: {allowed}

{schema_block}

Atomicity:
- One item = one fact/decision/outcome/constraint/rationale/lesson learned; NOT a broad essay summary.

Content rule (CRITICAL):
- content MUST be {content_sent_min}-{content_sent_max} sentences.
- It MUST include at least TWO of the following slots (keep them factual where applicable):
  (1) Outcome/result
  (2) Constraint/boundary/dependency
  (3) Rationale (why; failure cause if applicable)
  (4) Next-action implication (what to do/avoid next)
  (5) Artifact reference (explicitly mention a required/generated artifact name if available)

{elaboration_block}

Prohibitions (STRICT):
- Do NOT mention memory/retrieval/context/system design.
- Do NOT invent event-specific facts. If unknown, say "not recorded" rather than guessing.
- feedback_preference ONLY if an event explicitly states a preference about feedback/style.

Coverage:
- Generate {target_min_items} to {target_max_items} items.
- Mix levels: >=3 core, >=4 summary, >=4 episodic.
- Cover as many different event types as possible.
- Each item must cite its source event_id and time_index.

{output_example}
""".strip()

    user = {
        "role": "user",
        "content": "Events (ordered, prefix). Convert them into atomic memory items.\n\n"
                   + json.dumps(events, ensure_ascii=False, indent=2),
    }
    return [{"role": "system", "content": system}, user], allowed

def _build_refine_prompt(
    domain: str,
    events: List[Dict[str, Any]],
    draft_items: List[Dict[str, Any]],
    use_json_mode: bool,
    allow_elaboration: bool,
    content_sent_min: int,
    content_sent_max: int,
) -> List[Dict[str, str]]:
    allowed = _allowed_mem_types(domain)
    top_level_spec = (
        'Output MUST be a single JSON object with exactly one key "items". '
        'The value of "items" MUST be a JSON array. Output NO extra text.'
        if use_json_mode else
        "Output MUST be a single JSON array, with NO extra text."
    )
    output_example = _build_output_example(domain) if use_json_mode else ""

    elaboration_block = ""
    if allow_elaboration:
        elaboration_block = f"""
Richer rewrite (ALLOWED, bounded):
- Rewrite content to {content_sent_min}-{content_sent_max} sentences and make it more decision-relevant.
- You may add "Playbook/Next-action" operational detail that is generally valid and directly implied by events,
  but do NOT claim it happened; do NOT add event-specific facts not present.
""".strip()

    system = f"""
You are refining a draft memory bank into higher-quality atomic memory items.

Output format (STRICT):
- {top_level_spec}
- Use EXACTLY key "items". No commentary, no markdown fences.
- Allowed mem_type values: {allowed}

What to do:
1) Rewrite each item's "content" to be MORE decision-relevant:
   - {content_sent_min}-{content_sent_max} sentences per item.
   - Include at least TWO slots: Outcome / Constraint / Rationale / Next-action / Artifact reference.
2) Remove or fix inference-based items:
   - feedback_preference ONLY if explicitly stated in events.
   - learner_profile ONLY if explicitly supported by events.
3) De-duplicate near-duplicates; keep the sharper one.
4) Faithfulness:
   - Do NOT add event-specific facts not supported by events. Use "not recorded" if needed.
5) Keep schema fields stable:
   - Keep mem_id/event_id/time_index/mem_type/level/related_events where possible.

{elaboration_block}

Aim:
- Keep total items roughly similar to draft (16-26).
{output_example}
""".strip()

    user_obj = {"events": events, "draft_items": draft_items}
    user = {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False, indent=2)}
    return [{"role": "system", "content": system}, user]


# -----------------------------
# Per-job runner (for parallel)
# -----------------------------
def _process_one_job(job: Dict[str, Any], args: argparse.Namespace, api_key: str, limiter: Optional[_RateLimiter]) -> Dict[str, Any]:
    job_id = job["job_id"]
    domain = job["domain"]
    t = int(job["time_index"])
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bank_path = out_dir / "memory_bank.jsonl"
    meta_path = out_dir / "memory_bank.meta.json"
    qa_log_path = out_dir / args.log_name
    raw_dir = out_dir / "raw_attempts"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if bank_path.exists() and (not args.overwrite):
        _append_jsonl(qa_log_path, {
            "ts": time.time(),
            "job_id": job_id,
            "status": "skipped_exists",
            "bank_path": str(bank_path),
        })
        return {"job_id": job_id, "status": "skipped_exists"}

    rnd = random.Random(args.seed + (abs(hash(job_id)) % 10_000_000))
    events = _read_json(Path(job["events_path"]))
    prefix = _events_prefix(events, t)

    session = requests.Session()
    use_json_mode = bool(args.use_json_mode and (not args.no_json_mode))

    # ---------- Stage A: Draft ----------
    draft_messages, allowed = _build_draft_prompt(
        domain=domain,
        events=prefix,
        target_min_items=args.min_items,
        target_max_items=args.max_items,
        use_json_mode=use_json_mode,
        allow_elaboration=bool(args.allow_elaboration),
        content_sent_min=int(args.content_sent_min),
        content_sent_max=int(args.content_sent_max),
    )

    last_err: Optional[Exception] = None
    draft_items: Optional[List[Dict[str, Any]]] = None
    draft_latency = None
    draft_usage = None

    for attempt in range(1, args.retries + 1):
        resp_data: Optional[Dict[str, Any]] = None
        raw_text: Optional[str] = None
        try:
            resp_data, latency = _post_chat(
                api_key=api_key,
                model=args.model,
                messages=draft_messages,
                temperature=args.temperature,
                max_tokens=args.draft_max_tokens,
                timeout_sec=args.timeout,
                use_json_mode=use_json_mode,
                session=session,
                limiter=limiter,
            )
            draft_latency = latency
            draft_usage = resp_data.get("usage", {}) if isinstance(resp_data, dict) else None

            raw_text = resp_data["choices"][0]["message"]["content"]
            raw_path = raw_dir / f"{job_id}.draft.attempt{attempt}.txt"
            _write_text(raw_path, raw_text or "")

            parsed = _robust_json_loads(raw_text or "")
            draft_items = _validate_and_normalize_items(
                parsed,
                allowed_mem_types=allowed,
                expect_min=max(10, args.min_items // 2),
                min_content_chars=args.min_content_chars,
            )

            _append_jsonl(qa_log_path, {
                "ts": time.time(),
                "job_id": job_id,
                "stage": "draft",
                "attempt": attempt,
                "status": "ok",
                "domain": domain,
                "time_index": t,
                "num_events_prefix": len(prefix),
                "num_items": len(draft_items),
                "latency_sec": latency,
                "request_payload": resp_data.get("_request_payload"),
                "usage": resp_data.get("usage", {}),
                "raw_path": str(raw_path),
            })
            last_err = None
            break

        except Exception as e:
            last_err = e
            tb = traceback.format_exc()
            raw_path = raw_dir / f"{job_id}.draft.attempt{attempt}.txt"
            if not raw_path.exists():
                _write_text(raw_path, raw_text or "")

            _append_jsonl(qa_log_path, {
                "ts": time.time(),
                "job_id": job_id,
                "stage": "draft",
                "attempt": attempt,
                "status": "fail",
                "error": repr(e),
                "traceback": tb,
                "domain": domain,
                "time_index": t,
                "num_events_prefix": len(prefix),
                "request_payload": (resp_data.get("_request_payload") if resp_data else None),
                "raw_path": str(raw_path),
            })

            backoff = (2 ** (attempt - 1)) * 0.8 + rnd.random() * 0.2
            time.sleep(backoff)

    if last_err is not None or draft_items is None:
        _write_text(out_dir / "memory_bank.FAIL.txt", f"{job_id}\nDRAFT\n{repr(last_err)}\n")
        return {"job_id": job_id, "status": "fail_draft", "error": repr(last_err)}

    # ---------- Stage B: Refine (optional) ----------
    final_items = draft_items
    refine_latency = None
    refine_usage = None
    refine_failed = False

    if args.refine:
        refine_messages = _build_refine_prompt(
            domain=domain,
            events=prefix,
            draft_items=draft_items,
            use_json_mode=use_json_mode,
            allow_elaboration=bool(args.allow_elaboration),
            content_sent_min=int(args.content_sent_min),
            content_sent_max=int(args.content_sent_max),
        )

        last_err = None
        for attempt in range(1, args.retries + 1):
            resp_data = None
            raw_text = None
            try:
                resp_data, latency = _post_chat(
                    api_key=api_key,
                    model=args.model,
                    messages=refine_messages,
                    temperature=max(0.0, args.temperature * 0.6),
                    max_tokens=args.refine_max_tokens,
                    timeout_sec=args.timeout,
                    use_json_mode=use_json_mode,
                    session=session,
                    limiter=limiter,
                )
                refine_latency = latency
                refine_usage = resp_data.get("usage", {}) if isinstance(resp_data, dict) else None

                raw_text = resp_data["choices"][0]["message"]["content"]
                raw_path = raw_dir / f"{job_id}.refine.attempt{attempt}.txt"
                _write_text(raw_path, raw_text or "")

                parsed = _robust_json_loads(raw_text or "")
                refined = _validate_and_normalize_items(
                    parsed,
                    allowed_mem_types=_allowed_mem_types(domain),
                    expect_min=max(10, args.min_items // 2),
                    min_content_chars=args.min_content_chars,
                )

                _append_jsonl(qa_log_path, {
                    "ts": time.time(),
                    "job_id": job_id,
                    "stage": "refine",
                    "attempt": attempt,
                    "status": "ok",
                    "domain": domain,
                    "time_index": t,
                    "num_events_prefix": len(prefix),
                    "num_items": len(refined),
                    "latency_sec": latency,
                    "usage": resp_data.get("usage", {}),
                    "raw_path": str(raw_path),
                })

                final_items = refined
                last_err = None
                break

            except Exception as e:
                last_err = e
                tb = traceback.format_exc()
                raw_path = raw_dir / f"{job_id}.refine.attempt{attempt}.txt"
                if not raw_path.exists():
                    _write_text(raw_path, raw_text or "")

                _append_jsonl(qa_log_path, {
                    "ts": time.time(),
                    "job_id": job_id,
                    "stage": "refine",
                    "attempt": attempt,
                    "status": "fail",
                    "error": repr(e),
                    "traceback": tb,
                    "domain": domain,
                    "time_index": t,
                    "num_events_prefix": len(prefix),
                    "raw_path": str(raw_path),
                })

                backoff = (2 ** (attempt - 1)) * 0.8 + rnd.random() * 0.2
                time.sleep(backoff)

        if last_err is not None:
            refine_failed = True
            _write_text(out_dir / "memory_bank.REFINE_FAIL.txt", f"{job_id}\n{repr(last_err)}\n")

    # ---------- Write outputs ----------
    with bank_path.open("w", encoding="utf-8") as f:
        for m in final_items:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    meta = {
        "job_id": job_id,
        "domain": domain,
        "time_index": t,
        "num_events_prefix": len(prefix),
        "num_memory_items": len(final_items),
        "model": args.model,
        "temperature": args.temperature,
        "draft_max_tokens": args.draft_max_tokens,
        "refine_max_tokens": args.refine_max_tokens,
        "timeout": args.timeout,
        "use_json_mode": use_json_mode,
        "refine": bool(args.refine),
        "allow_elaboration": bool(args.allow_elaboration),
        "content_sent_min": int(args.content_sent_min),
        "content_sent_max": int(args.content_sent_max),
        "min_content_chars": int(args.min_content_chars),
        "draft_latency_sec": draft_latency,
        "refine_latency_sec": refine_latency,
        "draft_usage": draft_usage,
        "refine_usage": refine_usage,
        "refine_failed": refine_failed,
        "note": "If refine failed, output may fallback to draft (see memory_bank.REFINE_FAIL.txt).",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # small jitter after success (optional)
    if args.sleep > 0:
        time.sleep(args.sleep + rnd.random() * 0.05)

    return {"job_id": job_id, "status": "ok", "num_items": len(final_items)}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=str, default="evaluation/jobs.example.jsonl")
    ap.add_argument("--api_key_env", type=str, default=os.getenv("DEEPSEEK_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    ap.add_argument("--model", type=str, default="deepseek-reasoner")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)

    ap.add_argument("--min_items", type=int, default=16)
    ap.add_argument("--max_items", type=int, default=26)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--use_json_mode", action="store_true", default=True)
    ap.add_argument("--no_json_mode", action="store_true")

    ap.add_argument("--log_name", type=str, default="memory_bank.qa.jsonl")

    # Parallelism
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shuffle_jobs", action="store_true")

    # Richness controls
    ap.add_argument("--refine", action="store_true", default=True)
    ap.add_argument("--no_refine", action="store_true")
    ap.add_argument("--allow_elaboration", action="store_true", default=True)
    ap.add_argument("--no_elaboration", action="store_true")
    ap.add_argument("--content_sent_min", type=int, default=3)
    ap.add_argument("--content_sent_max", type=int, default=4)
    ap.add_argument("--min_content_chars", type=int, default=140)

    # Token controls (draft/refine separated)
    ap.add_argument("--draft_max_tokens", type=int, default=15000)
    ap.add_argument("--refine_max_tokens", type=int, default=15000)

    # Speed controls
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qps", type=float, default=0.0, help="global QPS limit across all workers; 0 disables")

    args = ap.parse_args()
    if args.no_refine:
        args.refine = False
    if args.no_elaboration:
        args.allow_elaboration = False

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Please export {args.api_key_env}=...")

    jobs = _read_jobs(Path(args.jobs))
    if args.shuffle_jobs:
        random.Random(args.seed).shuffle(jobs)

    # Pre-filter skips (optional)
    if not args.overwrite:
        kept = []
        for job in jobs:
            out_dir = Path(job["out_dir"])
            bank_path = out_dir / "memory_bank.jsonl"
            if bank_path.exists():
                continue
            kept.append(job)
        jobs_to_run = kept
    else:
        jobs_to_run = jobs

    if not jobs_to_run:
        print("[OK] No jobs to run (all exist, or empty jobs file).")
        return

    limiter = _RateLimiter(args.qps) if args.qps and args.qps > 0 else None

    ok_count = 0
    fail_count = 0
    skip_count = 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        future_map = {ex.submit(_process_one_job, job, args, api_key, limiter): job for job in jobs_to_run}
        pbar = tqdm(as_completed(future_map), total=len(future_map), desc="Generate memory_bank (parallel)", unit="job")

        for fut in pbar:
            job = future_map[fut]
            job_id = job.get("job_id")
            try:
                r = fut.result()
                st = r.get("status", "")
                if st == "ok":
                    ok_count += 1
                elif st.startswith("skipped"):
                    skip_count += 1
                else:
                    fail_count += 1
                pbar.set_postfix({"ok": ok_count, "fail": fail_count, "skip": skip_count})
            except Exception as e:
                fail_count += 1
                tqdm.write(f"[CRASH] {job_id}: {e}")
                pbar.set_postfix({"ok": ok_count, "fail": fail_count, "skip": skip_count})

        pbar.close()

    summary_path = Path(args.jobs).with_suffix(".step2.summary.json")
    summary = {
        "ts": time.time(),
        "total_jobs_in_file": len(jobs),
        "submitted": len(jobs_to_run),
        "ok": ok_count,
        "fail": fail_count,
        "skip": skip_count,
        "workers": int(args.workers),
        "model": args.model,
        "refine": bool(args.refine),
        "allow_elaboration": bool(args.allow_elaboration),
        "qps": float(args.qps),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Done. Summary -> {summary_path}")


if __name__ == "__main__":
    main()