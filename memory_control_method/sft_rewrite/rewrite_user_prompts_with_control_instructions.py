#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Only modify the USER prompt in a simulated SFT dataset:

Replace conversations[1].value with:
  <random control instruct for dependence_score> + "\n\n" + <base query>

Do NOT modify:
  - system message (user simulator)
  - assistant message (existing response)
  - other fields

Inputs:
  1) simulated continued SFT jsonl:
     - fields: conversations (system/user/assistant...), uid, dependence_score
  2) base merged answers jsonl:
     - fields: query_id, query (or user_prompt)
  3) control instruct json:
     - fields: levels[{level:int, instructions:[str]}]

Output:
  jsonl with updated user prompts.

Example:
    python rewrite_user_prompts_with_control_instructions.py \
        --simulated_in path/to/sft_val_simulated_continued.jsonl \
        --base_in path/to/all_answers_with_query_id.jsonl \
        --instruct_json sft_rewrite/control_instruct.json \
        --out path/to/sft_val_continued_control.jsonl \
        --seed 42 \
        --prefer_query_field query
"""

import argparse
import json
import random
import sys
from typing import Dict, Any, List, Optional, Tuple


def load_instruct_pool(path: str) -> Dict[int, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pool: Dict[int, List[str]] = {}
    levels = data.get("levels", [])
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"Invalid instruct json: missing 'levels' list in {path}")

    for lvl in levels:
        level = lvl.get("level", None)
        instrs = lvl.get("instructions", None)
        if not isinstance(level, int) or not isinstance(instrs, list) or not instrs:
            raise ValueError(f"Invalid level entry in instruct json: {lvl}")
        pool[level] = [str(x) for x in instrs]

    for k in range(1, 6):
        if k not in pool:
            raise ValueError(f"Instruct json missing level {k}")
    return pool


def build_base_index(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict: query_id -> {query, user_prompt}
    """
    idx: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON on line {ln} of {path}: {e}")

            qid = obj.get("query_id")
            if not isinstance(qid, str) or not qid:
                continue

            idx[qid] = {
                "query": obj.get("query"),
                "user_prompt": obj.get("user_prompt"),
            }

    if not idx:
        raise ValueError(f"No query_id indexed from base file: {path}")
    return idx


def extract_query_id_from_uid(uid: str) -> Optional[str]:
    if not isinstance(uid, str) or not uid:
        return None
    parts = uid.split("::")
    return parts[0] if parts else None


def pick_base_query(base: Dict[str, Any], prefer_query_field: str) -> str:
    q = base.get(prefer_query_field)
    if q is None or (isinstance(q, str) and not q.strip()):
        q = base.get("query") or base.get("user_prompt") or ""
    return str(q)


def replace_user_prompt_only(
    sample: Dict[str, Any],
    base_query: str,
    instruct: str,
    joiner: str
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Only update conversations[1].value (user).
    Return (updated_sample, warning_message)
    """
    conv = sample.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2:
        return sample, "missing_or_short_conversations"

    # Find the first user turn (prefer index 1, fallback search)
    user_idx = None
    if isinstance(conv[1], dict) and conv[1].get("from") == "user":
        user_idx = 1
    else:
        for i, turn in enumerate(conv):
            if isinstance(turn, dict) and turn.get("from") == "user":
                user_idx = i
                break

    if user_idx is None:
        return sample, "no_user_turn_found"

    conv[user_idx]["value"] = f"{instruct}{joiner}{base_query}"
    sample["conversations"] = conv
    return sample, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulated_in", required=True)
    ap.add_argument("--base_in", required=True)
    ap.add_argument("--instruct_json", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--prefer_query_field",
        type=str,
        default="query",
        choices=["query", "user_prompt"],
        help="Which field from base file to use as the base query; fallback to the other if missing."
    )
    ap.add_argument(
        "--joiner",
        type=str,
        default="\n\n",
        help="String inserted between instruct and base query in the user prompt."
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)

    instruct_pool = load_instruct_pool(args.instruct_json)
    base_idx = build_base_index(args.base_in)

    total = 0
    replaced = 0
    miss_base = 0
    bad_score = 0
    bad_uid = 0
    warn_counts: Dict[str, int] = {}

    with open(args.simulated_in, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for ln, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                sample = json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON on line {ln} of {args.simulated_in}: {e}")

            uid = sample.get("uid")
            qid = extract_query_id_from_uid(uid)
            if not qid:
                bad_uid += 1
                warn_counts["bad_uid"] = warn_counts.get("bad_uid", 0) + 1
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            base = base_idx.get(qid)
            if base is None:
                miss_base += 1
                warn_counts["missing_base_query_id"] = warn_counts.get("missing_base_query_id", 0) + 1
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            score = sample.get("dependence_score")
            if not isinstance(score, int) or score < 1 or score > 5:
                bad_score += 1
                warn_counts["bad_dependence_score"] = warn_counts.get("bad_dependence_score", 0) + 1
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            instruct = rng.choice(instruct_pool[score])
            base_query = pick_base_query(base, args.prefer_query_field)

            updated, warn = replace_user_prompt_only(
                sample=sample,
                base_query=base_query,
                instruct=instruct,
                joiner=args.joiner
            )

            if warn:
                warn_counts[warn] = warn_counts.get(warn, 0) + 1

            replaced += 1
            fout.write(json.dumps(updated, ensure_ascii=False) + "\n")

    summary = {
        "total_lines": total,
        "replaced_user_prompts": replaced,
        "missing_base_query_id": miss_base,
        "bad_uid": bad_uid,
        "bad_dependence_score": bad_score,
        "warnings": warn_counts
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()