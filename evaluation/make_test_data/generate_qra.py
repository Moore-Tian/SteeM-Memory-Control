#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3 (Parallel, Citation-Verified): Generate query, rubric, and multi-variant answers for each job.

Key fixes vs your current version:
1) Path-axis redefinition to avoid “low_path” being inherently worse under general rubrics:
   - low_path: can use ONLY query-explicit context; NO memory IDs, NO hidden prior details.
   - mid/high: progressively use memory items, but task correctness requirements stay identical.

2) Citation verifiability (hard constraints + repair loop):
   - For mid/high: citations must be subset of known mem_id from memory_bank.jsonl
   - All cited IDs must appear in answer as [m_XX]
   - Answer must not contain any [m_XX] not listed in citations
   - Optional relevance sanity: each cited item must share at least one non-trivial token with the answer
   - If violated: automatically run a “repair” prompt (edit answer + citations) for a few attempts.

3) Prompt input to answer stage uses a structured memory list (mem_id/type/level/content),
   reducing “hallucinated citations”.

Usage:
python evaluation/make_test_data/generate_qra.py \
    --jobs evaluation/jobs.example.jsonl \
  --model deepseek-reasoner \
  --workers 8 \
  --overwrite

Recommended initial token budgets (more stable than 4090 everywhere):
  --max_tokens_query 700
  --max_tokens_rubric 1200
  --max_tokens_answer_low 900
  --max_tokens_answer_mid 1200
  --max_tokens_answer_high 1600
"""

from __future__ import annotations

import argparse
import json
import os
import time
import random
import traceback
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
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
            s = line.strip()
            if not s:
                continue
            jobs.append(json.loads(s))
    return jobs

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items

def _events_prefix(events: List[Dict[str, Any]], t: int) -> List[Dict[str, Any]]:
    return [e for e in events if int(e.get("time_index", 0)) <= t]

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
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

    sess = session or requests
    r = sess.post(DEEPSEEK_CHAT_URL, headers=headers, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    data = r.json()
    data["_request_payload"] = payload
    return data


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
        if len(starts) >= 80:
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

    # 3) candidates (prefer dict)
    parsed: List[Any] = []
    for frag in _find_balanced_json_substrings(s1, max_candidates=18):
        try:
            parsed.append(json.loads(frag))
        except Exception:
            continue

    for v in parsed:
        if isinstance(v, dict):
            return v

    # last resort slice
    i = s1.find("{")
    j = s1.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(s1[i:j + 1])
        except Exception:
            pass

    raise ValueError("failed to parse JSON from model output")


# -----------------------------
# Memory packing for prompts
# -----------------------------
def _normalize_memory_items(memory_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in memory_items:
        mid = str(m.get("mem_id", "")).strip()
        if not mid:
            continue
        out.append({
            "mem_id": mid,
            "mem_type": str(m.get("mem_type", "")).strip(),
            "level": str(m.get("level", "")).strip(),
            "content": str(m.get("content", "")).strip(),
        })
    # stable sort by mem_id numeric if present; else lexical
    def keyfn(x: Dict[str, Any]) -> Tuple[int, str]:
        mm = x["mem_id"]
        m = re.search(r"m_(\d+)", mm)
        return (int(m.group(1)) if m else 10**9, mm)
    return sorted(out, key=keyfn)

def _select_memory_for_answers(memory_items: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    mem = _normalize_memory_items(memory_items)
    # keep up to max_items, prioritizing core/summary over episodic if needed
    if len(mem) <= max_items:
        return mem
    core = [x for x in mem if x.get("level") == "core"]
    summ = [x for x in mem if x.get("level") == "summary"]
    epi  = [x for x in mem if x.get("level") == "episodic"]
    keep: List[Dict[str, Any]] = []
    for bucket in (core, summ, epi):
        for x in bucket:
            if len(keep) >= max_items:
                break
            keep.append(x)
        if len(keep) >= max_items:
            break
    return keep[:max_items]

def _memory_id_set(memory_items: List[Dict[str, Any]]) -> Set[str]:
    return {str(m.get("mem_id", "")).strip() for m in memory_items if str(m.get("mem_id", "")).strip()}

def _memory_map(memory_items: List[Dict[str, Any]]) -> Dict[str, str]:
    return {str(m.get("mem_id", "")).strip(): str(m.get("content", "")).strip()
            for m in memory_items if str(m.get("mem_id", "")).strip()}


# -----------------------------
# Prompt builders
# -----------------------------
def _build_query_prompt(domain: str, t: int, prefix_events: List[Dict[str, Any]], memory_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    mem_small = _select_memory_for_answers(memory_items, max_items=18)
    system = f"""
You are generating ONE realistic next user query for a long-running assistant interaction.

Context:
- domain={domain}, current time_index={t}
- You are given a prefix of events and a compact list of prior context.

STRICT output format:
- Output MUST be a single JSON object (no markdown, no commentary).
- Use EXACT keys: query_id, domain, time_index, query, task_type, constraints, expected_output.
- constraints MUST be a JSON array (can be empty).
- query MUST be a single user message in natural language (1-3 sentences).

Faithfulness:
- Do NOT invent new facts about what happened.
- The query may request planning/decision/next step/rewrite/practice, etc.
- Prefer a query that can be answered well both with and without using prior context,
  i.e., it admits legitimate variation in “how much prior experience is reused”.

Return a single JSON object only.
""".strip()
    user_payload = {
        "prefix_events": prefix_events,
        "prior_context_compact": mem_small,
    }
    user = {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)}
    return [{"role": "system", "content": system}, user]

def _build_rubric_prompt(domain: str, t: int, query_obj: Dict[str, Any], mem_ids: List[str]) -> List[Dict[str, str]]:
    """
    Important: general_rubric must NOT bake in “use prior context” as a quality requirement,
    otherwise low_path becomes structurally worse.
    """
    system = f"""
You are creating a query-level evaluation rubric.

STRICT output format:
- Output MUST be a single JSON object (no markdown, no commentary).
- Use EXACT keys: rubric_id, domain, time_index, general_rubric, path_orientation_axis, auto_fail_checks.
- general_rubric MUST be an array of 4-6 criteria.
  Each criterion must have: name, description, score_scale="1-5", anchors with keys "1","3","5".
- path_orientation_axis MUST define differences in reliance on prior context WITHOUT changing correctness requirements:
  keys: low_path, mid_path, high_path. Each value 1-2 sentences.
- auto_fail_checks MUST be an array (can be empty).

Rubric design rules (CRITICAL):
1) general_rubric:
   - Focus on correctness/relevance, actionability, clarity/structure, constraint-handling, faithfulness to the user query.
   - Do NOT require “using prior context” to get high scores.
   - Avoid system words like “memory/retrieval/context window”.

2) path_orientation_axis:
   - low_path: uses only information explicit in the query; no additional prior details.
   - mid_path: uses 1-2 highly relevant prior details (if available) but stays concise.
   - high_path: integrates several specific prior outcomes/constraints/lessons to tailor advice.

3) auto_fail_checks (include at least 3):
   - Citation validity: any cited mem_id MUST be from the allowed set.
   - Citation formatting: cited mem_id must appear as [m_XX] in the answer.
   - No hallucinated prior facts: do not present non-recorded details as “already happened”.

Allowed citation ids (mem_id set) are provided as reference: {mem_ids}

Return a single JSON object only.
""".strip()
    user = {"role": "user", "content": json.dumps({"query": query_obj}, ensure_ascii=False, indent=2)}
    return [{"role": "system", "content": system}, user]

def _build_answer_prompt(
    domain: str,
    query_obj: Dict[str, Any],
    memory_items: List[Dict[str, Any]],
    variant: str,
    max_citations: int,
    word_target: Tuple[int, int],
) -> List[Dict[str, str]]:
    """
    Output schema:
    {
      "variant": "low_path|mid_path|high_path",
      "answer": "<final answer text>",
      "citations": ["m_01", "m_05"]
    }
    """
    mem_for_answer = _select_memory_for_answers(memory_items, max_items=26)
    allowed_ids = [m["mem_id"] for m in mem_for_answer]

    lo, hi = word_target
    if variant == "low_path":
        variant_rules = f"""
Variant = low_path:
- You may use ONLY what is explicitly in the user's query.
- Do NOT use any additional prior details.
- Do NOT cite any mem_id; citations MUST be [] and answer MUST NOT contain patterns like [m_XX].
- Target length: {lo}-{hi} words.
""".strip()
    elif variant == "mid_path":
        variant_rules = f"""
Variant = mid_path:
- Use a SMALL number of highly relevant prior details to tailor the answer.
- Cite at most {max_citations} mem_id inline exactly like [m_XX] when used.
- citations must list exactly the mem_id you cited (unique).
- Target length: {lo}-{hi} words.
""".strip()
    else:
        variant_rules = f"""
Variant = high_path:
- Integrate several specific prior outcomes/constraints/lessons to tailor the answer.
- Cite at most {max_citations} mem_id inline exactly like [m_XX] when used.
- citations must list exactly the mem_id you cited (unique).
- Target length: {lo}-{hi} words.
""".strip()

    system = f"""
You are answering the user's query.

STRICT output format:
- Output MUST be a single JSON object (no markdown, no commentary).
- Use EXACT keys: variant, answer, citations.
- citations MUST be a JSON array of mem_id strings.

Citation integrity rules (HARD):
- You may ONLY cite mem_id from this allowed set: {allowed_ids}
- Any cited mem_id must appear in the answer exactly like [m_XX].
- The answer must NOT contain any [m_XX] that is not listed in citations.
- If unsure about a prior detail, do not cite it. Keep it as general guidance.

Faithfulness:
- You may add operational guidance (templates/checklists/rules-of-thumb) that is generally valid.
- Do NOT present elaborations as events that already happened.
- Do NOT invent specific names/numbers unless present in the provided memory items.

{variant_rules}

Return a single JSON object only.
""".strip()

    user_payload = {
        "query": query_obj,
        "memory_items": mem_for_answer,   # structured, to prevent hallucinated citations
    }
    user = {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)}
    return [{"role": "system", "content": system}, user]

def _build_answer_repair_prompt(
    variant: str,
    query_obj: Dict[str, Any],
    memory_items: List[Dict[str, Any]],
    bad_answer_obj: Dict[str, Any],
    reason: str,
    max_citations: int,
    word_target: Tuple[int, int],
) -> List[Dict[str, str]]:
    """
    Repair: fix citations + remove hallucinated prior details while preserving intent.
    """
    mem_for_answer = _select_memory_for_answers(memory_items, max_items=26)
    allowed_ids = [m["mem_id"] for m in mem_for_answer]
    lo, hi = word_target

    system = f"""
You are repairing an answer JSON object to satisfy strict citation and faithfulness constraints.

You MUST output a single JSON object with keys: variant, answer, citations.

Repair objectives:
1) Fix citations:
   - citations must be a subset of allowed mem_id: {allowed_ids}
   - any cited mem_id must appear in the answer as [m_XX]
   - the answer must not contain any [m_XX] not listed in citations
2) Remove any prior details that are not supported by the provided memory items.
   - Keep operational advice, but do not present it as “already happened”.
3) Keep the answer focused and useful, matching the user query.
4) Length target: {lo}-{hi} words.

Variant rule reminder:
- low_path: citations MUST be [] and answer must contain no [m_XX]
- mid/high: cite at most {max_citations} items, only when actually used.

Reason for repair (do not repeat verbatim): {reason}
Return only the repaired JSON object.
""".strip()

    user_payload = {
        "variant": variant,
        "query": query_obj,
        "memory_items": mem_for_answer,
        "bad_answer": bad_answer_obj,
    }
    user = {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)}
    return [{"role": "system", "content": system}, user]


# -----------------------------
# Validators
# -----------------------------
_MEM_REF_RE = re.compile(r"\[(m_\d+)\]")

def _require_keys(obj: Any, keys: List[str]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON must be an object")
    for k in keys:
        if k not in obj:
            raise ValueError(f"missing key: {k}")
    return obj

def _validate_query(obj: Any, domain: str, t: int) -> Dict[str, Any]:
    o = _require_keys(obj, ["query_id", "domain", "time_index", "query", "task_type", "constraints", "expected_output"])
    o["domain"] = domain
    o["time_index"] = int(o.get("time_index", t))
    if not isinstance(o["query"], str) or not o["query"].strip():
        raise ValueError("query empty")
    if not isinstance(o["constraints"], list):
        raise ValueError("constraints must be list")
    return o

def _validate_rubric(obj: Any, domain: str, t: int) -> Dict[str, Any]:
    o = _require_keys(obj, ["rubric_id", "domain", "time_index", "general_rubric", "path_orientation_axis", "auto_fail_checks"])
    o["domain"] = domain
    o["time_index"] = int(o.get("time_index", t))

    gr = o["general_rubric"]
    if not isinstance(gr, list) or not (4 <= len(gr) <= 6):
        raise ValueError("general_rubric must be list of 4-6")

    # basic criterion schema check
    for i, c in enumerate(gr):
        if not isinstance(c, dict):
            raise ValueError(f"general_rubric[{i}] must be object")
        for k in ["name", "description", "score_scale", "anchors"]:
            if k not in c:
                raise ValueError(f"general_rubric[{i}] missing {k}")
        if str(c.get("score_scale")) != "1-5":
            c["score_scale"] = "1-5"
        anchors = c.get("anchors")
        if not isinstance(anchors, dict) or any(x not in anchors for x in ["1", "3", "5"]):
            raise ValueError(f"general_rubric[{i}].anchors must include 1/3/5")

    ax = o["path_orientation_axis"]
    if not isinstance(ax, dict) or any(k not in ax for k in ["low_path", "mid_path", "high_path"]):
        raise ValueError("path_orientation_axis missing keys")

    if not isinstance(o["auto_fail_checks"], list):
        o["auto_fail_checks"] = []

    return o

def _word_count(s: str) -> int:
    # language-agnostic-ish: count whitespace-separated tokens; good enough as a soft bound
    return len([x for x in re.split(r"\s+", s.strip()) if x])

def _tokens_for_relevance(text: str) -> Set[str]:
    toks = set()
    for w in re.findall(r"[A-Za-z0-9_\-]{5,}", text.lower()):
        toks.add(w)
        if len(toks) >= 80:
            break
    return toks

def _validate_answer_strict(
    obj: Any,
    variant: str,
    allowed_mem_ids: Set[str],
    mem_content_map: Dict[str, str],
    max_citations: int,
    word_target: Tuple[int, int],
    enable_relevance_check: bool,
) -> Dict[str, Any]:
    o = _require_keys(obj, ["variant", "answer", "citations"])
    o["variant"] = variant
    if not isinstance(o["answer"], str) or not o["answer"].strip():
        raise ValueError("answer empty")
    if not isinstance(o["citations"], list):
        raise ValueError("citations must be list")

    ans = o["answer"].strip()
    cited_list = [str(x).strip() for x in o["citations"] if str(x).strip()]
    # unique preserve order
    seen = set()
    cited = []
    for x in cited_list:
        if x not in seen:
            cited.append(x)
            seen.add(x)
    o["citations"] = cited

    # word length soft bounds (do not fail hard; but keep as a sanity signal if needed)
    lo, hi = word_target
    wc = _word_count(ans)
    # We keep it as a warning-only; enforcement tends to reduce success rate.
    o["_word_count"] = wc
    o["_word_target"] = [lo, hi]

    in_text_refs = _MEM_REF_RE.findall(ans)
    in_text_refs_set = set(in_text_refs)

    if variant == "low_path":
        if len(o["citations"]) != 0:
            raise ValueError("low_path citations must be []")
        if in_text_refs_set:
            raise ValueError("low_path answer must not contain [m_XX] citations")
        return o

    # mid/high:
    if len(o["citations"]) > max_citations:
        raise ValueError(f"too many citations: {len(o['citations'])} > {max_citations}")

    # citations must be valid ids
    bad = [x for x in o["citations"] if x not in allowed_mem_ids]
    if bad:
        raise ValueError(f"invalid citations not in memory_bank: {bad}")

    # every cited id must appear in text
    missing_in_text = [x for x in o["citations"] if x not in in_text_refs_set]
    if missing_in_text:
        raise ValueError(f"citations listed but not present in answer text as [m_XX]: {missing_in_text}")

    # answer must not contain uncited ids
    extra_in_text = sorted(list(in_text_refs_set - set(o["citations"])))
    if extra_in_text:
        raise ValueError(f"answer contains [m_XX] not listed in citations: {extra_in_text}")

    # optional relevance sanity: require at least one shared token per citation
    if enable_relevance_check:
        ans_toks = _tokens_for_relevance(ans)
        for mid in o["citations"]:
            mem_txt = mem_content_map.get(mid, "")
            mem_toks = _tokens_for_relevance(mem_txt)
            if mem_toks and ans_toks and (ans_toks.isdisjoint(mem_toks)):
                raise ValueError(f"citation seems unrelated to answer content: {mid}")

    return o


# -----------------------------
# Call with retries + logging
# -----------------------------
def _call_stage_with_retries(
    *,
    stage: str,
    job_id: str,
    session: requests.Session,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    use_json_mode: bool,
    qa_log_path: Path,
    raw_dir: Path,
    retries: int,
    rnd: random.Random,
) -> Tuple[Dict[str, Any], str, str, Dict[str, Any]]:
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        resp_data: Optional[Dict[str, Any]] = None
        raw_text: Optional[str] = None
        finish_reason = None
        usage: Dict[str, Any] = {}
        raw_path = raw_dir / f"{job_id}.{stage}.attempt{attempt}.txt"

        try:
            resp_data = _post_chat(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout,
                use_json_mode=use_json_mode,
                session=session,
            )

            choice0 = (resp_data.get("choices") or [{}])[0]
            msg0 = choice0.get("message") or {}
            raw_text = msg0.get("content")
            finish_reason = choice0.get("finish_reason")
            usage = resp_data.get("usage", {}) or {}

            _write_text(raw_path, raw_text or "")

            if resp_data.get("error"):
                raise RuntimeError(f"api_error: {resp_data.get('error')}")
            if raw_text is None or not str(raw_text).strip():
                raise ValueError("empty content")

            _append_jsonl(qa_log_path, {
                "ts": time.time(),
                "job_id": job_id,
                "stage": stage,
                "attempt": attempt,
                "status": "ok",
                "finish_reason": finish_reason,
                "usage": usage,
                "request_payload": resp_data.get("_request_payload"),
                "raw_path": str(raw_path),
            })
            return resp_data, str(raw_text), str(finish_reason), usage

        except Exception as e:
            last_err = e
            tb = traceback.format_exc()

            if not raw_path.exists():
                _write_text(raw_path, raw_text or "")

            _append_jsonl(qa_log_path, {
                "ts": time.time(),
                "job_id": job_id,
                "stage": stage,
                "attempt": attempt,
                "status": "fail",
                "error": repr(e),
                "traceback": tb,
                "finish_reason": finish_reason,
                "usage": usage,
                "request_payload": (resp_data.get("_request_payload") if resp_data else None),
                "raw_path": str(raw_path),
            })

            backoff = (2 ** (attempt - 1)) * 0.9 + rnd.random() * 0.3
            time.sleep(backoff)

    raise RuntimeError(f"{stage} failed after {retries} attempts: {repr(last_err)}")


# -----------------------------
# Per-job runner
# -----------------------------
def _process_one_job(job: Dict[str, Any], args: argparse.Namespace, api_key: str) -> Dict[str, Any]:
    job_id = job["job_id"]
    domain = job["domain"]
    t = int(job["time_index"])
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_log_path = out_dir / args.log_name
    raw_dir = out_dir / "raw_attempts"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Required input from Step2
    mem_path = out_dir / "memory_bank.jsonl"
    if not mem_path.exists():
        _append_jsonl(qa_log_path, {"ts": time.time(), "job_id": job_id, "status": "skipped_no_memory_bank"})
        return {"job_id": job_id, "status": "skipped_no_memory_bank"}

    query_path = out_dir / "query.json"
    rubric_path = out_dir / "rubric.json"
    answers_path = out_dir / "answers.json"
    if (not args.overwrite) and query_path.exists() and rubric_path.exists() and answers_path.exists():
        _append_jsonl(qa_log_path, {"ts": time.time(), "job_id": job_id, "status": "skipped_exists"})
        return {"job_id": job_id, "status": "skipped_exists"}

    rnd = random.Random(args.seed + (abs(hash(job_id)) % 10_000_000))
    session = requests.Session()
    use_json_mode = bool(args.use_json_mode and (not args.no_json_mode))

    events = _read_json(Path(job["events_path"]))
    prefix = _events_prefix(events, t)
    memory_items_raw = _read_jsonl(mem_path)

    mem_for_answer = _select_memory_for_answers(memory_items_raw, max_items=args.max_memory_items)
    allowed_ids = _memory_id_set(mem_for_answer)
    mem_map = _memory_map(mem_for_answer)
    allowed_id_list_sorted = sorted(list(allowed_ids))

    # ---- Stage Q: query ----
    q_messages = _build_query_prompt(domain=domain, t=t, prefix_events=prefix, memory_items=mem_for_answer)
    _, q_raw, q_finish, q_usage = _call_stage_with_retries(
        stage="query",
        job_id=job_id,
        session=session,
        api_key=api_key,
        model=args.model,
        messages=q_messages,
        temperature=args.temperature_query,
        max_tokens=args.max_tokens_query,
        timeout=args.timeout,
        use_json_mode=use_json_mode,
        qa_log_path=qa_log_path,
        raw_dir=raw_dir,
        retries=args.retries,
        rnd=rnd,
    )
    q_obj = _validate_query(_robust_json_loads(q_raw), domain=domain, t=t)
    _write_json(query_path, q_obj)

    # ---- Stage R: rubric ----
    r_messages = _build_rubric_prompt(domain=domain, t=t, query_obj=q_obj, mem_ids=allowed_id_list_sorted)
    _, r_raw, r_finish, r_usage = _call_stage_with_retries(
        stage="rubric",
        job_id=job_id,
        session=session,
        api_key=api_key,
        model=args.model,
        messages=r_messages,
        temperature=args.temperature_rubric,
        max_tokens=args.max_tokens_rubric,
        timeout=args.timeout,
        use_json_mode=use_json_mode,
        qa_log_path=qa_log_path,
        raw_dir=raw_dir,
        retries=args.retries,
        rnd=rnd,
    )
    r_obj = _validate_rubric(_robust_json_loads(r_raw), domain=domain, t=t)
    _write_json(rubric_path, r_obj)

    # ---- Stage A: answers (3 variants, strict citation validation + repair) ----
    answers: List[Dict[str, Any]] = []
    variants = ["low_path", "mid_path", "high_path"]

    variant_max_tokens = {
        "low_path": args.max_tokens_answer_low,
        "mid_path": args.max_tokens_answer_mid,
        "high_path": args.max_tokens_answer_high,
    }
    variant_word_targets = {
        "low_path": (args.low_words_min, args.low_words_max),
        "mid_path": (args.mid_words_min, args.mid_words_max),
        "high_path": (args.high_words_min, args.high_words_max),
    }

    for v in variants:
        max_cit = (0 if v == "low_path" else args.max_citations)

        # primary generation
        a_messages = _build_answer_prompt(
            domain=domain,
            query_obj=q_obj,
            memory_items=mem_for_answer,
            variant=v,
            max_citations=max_cit,
            word_target=variant_word_targets[v],
        )

        try:
            _, a_raw, a_finish, a_usage = _call_stage_with_retries(
                stage=f"answer.{v}",
                job_id=job_id,
                session=session,
                api_key=api_key,
                model=args.model,
                messages=a_messages,
                temperature=args.temperature_answer,
                max_tokens=variant_max_tokens[v],
                timeout=args.timeout,
                use_json_mode=use_json_mode,
                qa_log_path=qa_log_path,
                raw_dir=raw_dir,
                retries=args.retries,
                rnd=rnd,
            )
            a_obj = _robust_json_loads(a_raw)
            a_obj = _validate_answer_strict(
                a_obj,
                variant=v,
                allowed_mem_ids=allowed_ids,
                mem_content_map=mem_map,
                max_citations=max_cit,
                word_target=variant_word_targets[v],
                enable_relevance_check=bool(args.enable_citation_relevance_check),
            )
            a_obj["_finish_reason"] = a_finish
            a_obj["_usage"] = a_usage

        except Exception as e:
            # attempt repair loop (only if enabled)
            if args.enable_answer_repair:
                repair_err = e
                bad_obj: Dict[str, Any]
                try:
                    bad_obj = _robust_json_loads(a_raw) if "a_raw" in locals() else {"variant": v, "answer": "", "citations": []}
                except Exception:
                    bad_obj = {"variant": v, "answer": "", "citations": []}

                repaired: Optional[Dict[str, Any]] = None
                for rep_i in range(1, args.max_answer_repairs + 1):
                    rep_messages = _build_answer_repair_prompt(
                        variant=v,
                        query_obj=q_obj,
                        memory_items=mem_for_answer,
                        bad_answer_obj=bad_obj,
                        reason=repr(repair_err),
                        max_citations=max_cit,
                        word_target=variant_word_targets[v],
                    )
                    try:
                        _, rep_raw, rep_finish, rep_usage = _call_stage_with_retries(
                            stage=f"answer.{v}.repair{rep_i}",
                            job_id=job_id,
                            session=session,
                            api_key=api_key,
                            model=args.model,
                            messages=rep_messages,
                            temperature=max(0.0, args.temperature_answer * 0.5),
                            max_tokens=max(600, variant_max_tokens[v]),
                            timeout=args.timeout,
                            use_json_mode=use_json_mode,
                            qa_log_path=qa_log_path,
                            raw_dir=raw_dir,
                            retries=max(2, args.repair_stage_retries),
                            rnd=rnd,
                        )
                        rep_obj = _validate_answer_strict(
                            _robust_json_loads(rep_raw),
                            variant=v,
                            allowed_mem_ids=allowed_ids,
                            mem_content_map=mem_map,
                            max_citations=max_cit,
                            word_target=variant_word_targets[v],
                            enable_relevance_check=bool(args.enable_citation_relevance_check),
                        )
                        rep_obj["_finish_reason"] = rep_finish
                        rep_obj["_usage"] = rep_usage
                        rep_obj["_repaired_from_error"] = repr(e)
                        repaired = rep_obj
                        break
                    except Exception as ee:
                        repair_err = ee
                        continue

                if repaired is not None:
                    a_obj = repaired
                else:
                    _append_jsonl(qa_log_path, {
                        "ts": time.time(),
                        "job_id": job_id,
                        "stage": f"answer.{v}",
                        "status": "variant_fail_after_repairs",
                        "error": repr(e),
                    })
                    a_obj = {"variant": v, "answer": "", "citations": [], "_error": repr(e)}
            else:
                _append_jsonl(qa_log_path, {
                    "ts": time.time(),
                    "job_id": job_id,
                    "stage": f"answer.{v}",
                    "status": "variant_fail",
                    "error": repr(e),
                })
                a_obj = {"variant": v, "answer": "", "citations": [], "_error": repr(e)}

        answers.append(a_obj)

    out_answers = {
        "job_id": job_id,
        "domain": domain,
        "time_index": t,
        "query_id": q_obj.get("query_id"),
        "rubric_id": r_obj.get("rubric_id"),
        "answers": answers,
    }
    _write_json(answers_path, out_answers)

    meta = {
        "job_id": job_id,
        "domain": domain,
        "time_index": t,
        "num_prefix_events": len(prefix),
        "num_memory_items_used_for_answer": len(mem_for_answer),
        "model": args.model,
        "json_mode": use_json_mode,
        "query_finish_reason": q_finish,
        "rubric_finish_reason": r_finish,
        "note": "answers.json includes low/mid/high variants; citations are strictly validated against memory_bank.jsonl.",
    }
    _write_json(out_dir / "qra.meta.json", meta)

    ok_variants = sum(1 for a in answers if isinstance(a, dict) and a.get("answer"))
    status = "ok" if ok_variants == 3 else ("partial_ok" if ok_variants > 0 else "fail_all_answers")
    return {"job_id": job_id, "status": status, "ok_variants": ok_variants}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--jobs", type=str, default="evaluation/jobs.example.jsonl")
    ap.add_argument("--api_key_env", type=str, default=os.getenv("DEEPSEEK_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    ap.add_argument("--model", type=str, default="deepseek-reasoner")

    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shuffle_jobs", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--use_json_mode", action="store_true", default=True)
    ap.add_argument("--no_json_mode", action="store_true")

    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--retries", type=int, default=4)

    # Token budgets (recommend starting modest for stability)
    ap.add_argument("--max_tokens_query", type=int, default=4090)
    ap.add_argument("--max_tokens_rubric", type=int, default=4090)
    ap.add_argument("--max_tokens_answer_low", type=int, default=4090)
    ap.add_argument("--max_tokens_answer_mid", type=int, default=4090)
    ap.add_argument("--max_tokens_answer_high", type=int, default=4090)

    # Temperatures
    ap.add_argument("--temperature_query", type=float, default=0.2)
    ap.add_argument("--temperature_rubric", type=float, default=0.0)
    ap.add_argument("--temperature_answer", type=float, default=0.0)

    # Citations
    ap.add_argument("--max_citations", type=int, default=6)
    ap.add_argument("--max_memory_items", type=int, default=26)

    # Answer length targets (soft bounds, used as guidance + metadata)
    ap.add_argument("--low_words_min", type=int, default=140)
    ap.add_argument("--low_words_max", type=int, default=260)
    ap.add_argument("--mid_words_min", type=int, default=180)
    ap.add_argument("--mid_words_max", type=int, default=340)
    ap.add_argument("--high_words_min", type=int, default=240)
    ap.add_argument("--high_words_max", type=int, default=450)

    # Repair + relevance checks
    ap.add_argument("--enable_answer_repair", action="store_true", default=True)
    ap.add_argument("--max_answer_repairs", type=int, default=3)
    ap.add_argument("--repair_stage_retries", type=int, default=2)
    ap.add_argument("--enable_citation_relevance_check", action="store_true", default=True)

    ap.add_argument("--log_name", type=str, default="qra.qa.jsonl")

    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Please export {args.api_key_env}=...")

    jobs = _read_jobs(Path(args.jobs))
    if args.shuffle_jobs:
        random.Random(args.seed).shuffle(jobs)

    # prefilter skips
    if not args.overwrite:
        kept = []
        for job in jobs:
            out_dir = Path(job["out_dir"])
            if (out_dir / "query.json").exists() and (out_dir / "rubric.json").exists() and (out_dir / "answers.json").exists():
                continue
            kept.append(job)
        jobs_to_run = kept
    else:
        jobs_to_run = jobs

    if not jobs_to_run:
        print("[OK] No jobs to run.")
        return

    ok = 0
    partial = 0
    fail = 0
    skip = 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        fut_map = {ex.submit(_process_one_job, job, args, api_key): job for job in jobs_to_run}
        pbar = tqdm(as_completed(fut_map), total=len(fut_map), desc="Generate Q/R/A (parallel)", unit="job")

        for fut in pbar:
            job = fut_map[fut]
            job_id = job.get("job_id")
            try:
                r = fut.result()
                st = r.get("status")
                if st == "ok":
                    ok += 1
                elif st == "partial_ok":
                    partial += 1
                elif st and st.startswith("skipped"):
                    skip += 1
                else:
                    fail += 1
                pbar.set_postfix({"ok": ok, "partial": partial, "fail": fail, "skip": skip})
            except Exception as e:
                fail += 1
                tqdm.write(f"[CRASH] {job_id}: {e}")
                pbar.set_postfix({"ok": ok, "partial": partial, "fail": fail, "skip": skip})

        pbar.close()

    summary = {
        "ts": time.time(),
        "total_jobs_in_file": len(jobs),
        "submitted": len(jobs_to_run),
        "ok": ok,
        "partial_ok": partial,
        "fail": fail,
        "skip": skip,
        "workers": args.workers,
        "model": args.model,
        "json_mode": bool(args.use_json_mode and (not args.no_json_mode)),
        "citation_relevance_check": bool(args.enable_citation_relevance_check),
        "answer_repair": bool(args.enable_answer_repair),
    }
    summary_path = Path(args.jobs).with_suffix(".step3.qra.summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Done. Summary -> {summary_path}")


if __name__ == "__main__":
    main()