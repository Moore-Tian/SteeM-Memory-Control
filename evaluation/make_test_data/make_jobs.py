#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: 从已生成好的 timeline（events/interactions/stats）中抽取若干 (domain, case_id, time_index) 作为测试 job。

输入目录结构（示例）：
/data/.../output_gemini/
  research/
    0/
      events.json
      interactions.json
      stats.json
    1/...
  tutoring/
    0/...

输出：
- jobs.jsonl：每行一个 job（后续 Step2/3 读取）
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Job:
    job_id: str
    domain: str
    case_id: str
    time_index: int
    root_dir: str
    case_dir: str
    events_path: str
    interactions_path: str
    stats_path: str
    out_dir: str


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_cases(root: Path, domain: str) -> List[Path]:
    dom_dir = root / domain
    if not dom_dir.exists():
        return []
    cases = []
    for p in dom_dir.iterdir():
        if p.is_dir():
            if (p / "events.json").exists() and (p / "stats.json").exists():
                cases.append(p)
    cases.sort(key=lambda x: x.name)
    return cases


def _candidate_timepoints(events: List[Dict[str, Any]], min_events_prefix: int, min_t: int) -> List[int]:
    # 可选 time_index：保证 prefix(<=t) 至少有 min_events_prefix 条事件
    by_t = {}
    for e in events:
        t = int(e.get("time_index", 0))
        by_t.setdefault(t, 0)
        by_t[t] += 1
    max_t = max(by_t.keys()) if by_t else 0

    prefix = 0
    candidates = []
    for t in range(1, max_t + 1):
        prefix += by_t.get(t, 0)
        if t >= min_t and prefix >= min_events_prefix:
            candidates.append(t)
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="examples/event_output", help="timeline 根目录，例如 examples/event_output")
    ap.add_argument("--domains", type=str, default="research,tutoring", help="逗号分隔，例如 research,tutoring")
    ap.add_argument("--num_cases_per_domain", type=int, default=1)
    ap.add_argument("--timepoints_per_case", type=int, default=2)
    ap.add_argument("--min_events_prefix", type=int, default=8, help="要求 t 时刻之前至少多少事件，避免 memory 太贫")
    ap.add_argument("--min_t", type=int, default=3, help="最小 time_index")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="jobs.jsonl")
    ap.add_argument("--out_base", type=str, default="evaluation/test_set", help="每个 job 的输出根目录")
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(args.root).resolve()
    out_base = Path(args.out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    all_jobs: List[Job] = []

    for domain in domains:
        cases = _discover_cases(root, domain)
        if not cases:
            print(f"[WARN] domain not found or empty: {domain} under {root}")
            continue

        picked_cases = cases if len(cases) <= args.num_cases_per_domain else random.sample(cases, args.num_cases_per_domain)

        for case_dir in picked_cases:
            case_id = case_dir.name
            events_path = case_dir / "events.json"
            interactions_path = case_dir / "interactions.json"
            stats_path = case_dir / "stats.json"

            events = _read_json(events_path)
            if not isinstance(events, list) or not events:
                print(f"[WARN] empty events: {events_path}")
                continue

            cands = _candidate_timepoints(events, min_events_prefix=args.min_events_prefix, min_t=args.min_t)
            if not cands:
                # 退化：直接用最大 time_index
                max_t = max(int(e.get("time_index", 0)) for e in events)
                cands = [max_t]

            picked_ts = cands if len(cands) <= args.timepoints_per_case else random.sample(cands, args.timepoints_per_case)

            for t in sorted(picked_ts):
                job_id = f"{domain}__{case_id}__t{t}"
                out_dir = out_base / domain / case_id / f"t{t}"
                out_dir.mkdir(parents=True, exist_ok=True)

                all_jobs.append(
                    Job(
                        job_id=job_id,
                        domain=domain,
                        case_id=case_id,
                        time_index=int(t),
                        root_dir=str(root),
                        case_dir=str(case_dir),
                        events_path=str(events_path),
                        interactions_path=str(interactions_path),
                        stats_path=str(stats_path),
                        out_dir=str(out_dir),
                    )
                )

    out_path = Path(args.out).resolve()
    with out_path.open("w", encoding="utf-8") as f:
        for j in all_jobs:
            f.write(json.dumps(asdict(j), ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(all_jobs)} jobs -> {out_path}")
    print(f"[OK] outputs will be placed under -> {out_base}")


if __name__ == "__main__":
    main()