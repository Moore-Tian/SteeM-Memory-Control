#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_dependence_metrics.py

Compute delta metrics from judge results.
Calculates |target_dependence_score - overall_memory_dependence_score|,
grouped by (domain, task).

Enhancements:
- Explicitly skip samples that:
  - generation failed (api_error or empty generated_text)
  - judge failed (judge_parse_ok=False or missing overall score)
- Record skip counts in output JSON
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict, Counter


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def extract_score(score_value: Any) -> Optional[int]:
    if score_value is None:
        return None

    if isinstance(score_value, int):
        return score_value

    if isinstance(score_value, float):
        return int(score_value)

    if isinstance(score_value, str):
        try:
            return int(float(score_value))
        except (ValueError, TypeError):
            return None

    if isinstance(score_value, dict):
        for key in ["overall_memory_dependence_score", "score", "value", "rating"]:
            if key in score_value:
                val = score_value[key]
                if isinstance(val, (int, float)):
                    return int(val)
                if isinstance(val, str):
                    try:
                        return int(float(val))
                    except (ValueError, TypeError):
                        continue

    return None


def _nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def compute_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(samples)

    skipped_generation_failed = 0
    skipped_judge_failed = 0
    skipped_missing_target = 0

    skip_reason_counter = Counter()

    valid_items = []

    for s in samples:
        # 1) generation failed (most important)
        if s.get("api_error"):
            skipped_generation_failed += 1
            skip_reason_counter["generation_failed_api_error"] += 1
            continue

        # Some pipelines may not put api_error but still have empty output
        if not _nonempty_str(s.get("generated_text")):
            skipped_generation_failed += 1
            skip_reason_counter["generation_failed_empty_output"] += 1
            continue

        # 2) judge failed
        if s.get("judge_parse_ok") is False:
            skipped_judge_failed += 1
            skip_reason_counter["judge_failed_parse_or_skip"] += 1
            continue

        target_score = extract_score(s.get("target_dependence_score"))
        if target_score is None:
            skipped_missing_target += 1
            skip_reason_counter["missing_target_dependence_score"] += 1
            continue

        actual_score = extract_score(s.get("overall_memory_dependence_score"))
        if actual_score is None:
            skipped_judge_failed += 1
            skip_reason_counter["missing_overall_score"] += 1
            continue

        domain = s.get("domain", "UNKNOWN")
        task = s.get("task", "UNKNOWN")
        delta = abs(target_score - actual_score)

        valid_items.append(
            {
                "domain": domain,
                "task": task,
                "target_score": target_score,
                "actual_score": actual_score,
                "delta": delta,
            }
        )

    print(f"[INFO] Total input samples: {total}")
    print(f"[INFO] Valid samples: {len(valid_items)}")
    print(f"[INFO] Skipped generation_failed: {skipped_generation_failed}")
    print(f"[INFO] Skipped judge_failed: {skipped_judge_failed}")
    print(f"[INFO] Skipped missing_target: {skipped_missing_target}")

    # Group by (domain, task)
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for it in valid_items:
        grouped[(it["domain"], it["task"])].append(it)

    # Per-group stats
    by_domain_task: Dict[str, Any] = {}
    for (domain, task), items in grouped.items():
        deltas = [x["delta"] for x in items]
        targets = [x["target_score"] for x in items]
        actuals = [x["actual_score"] for x in items]

        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        mean_target = sum(targets) / len(targets) if targets else 0.0
        mean_actual = sum(actuals) / len(actuals) if actuals else 0.0

        # breakdown by target score
        by_target: Dict[int, List[float]] = defaultdict(list)
        for x in items:
            by_target[x["target_score"]].append(x["delta"])

        target_stats = {}
        for t in sorted(by_target.keys()):
            ds = by_target[t]
            target_stats[str(t)] = {
                "mean_delta": (sum(ds) / len(ds)) if ds else 0.0,
                "count": len(ds),
            }

        by_domain_task[f"{domain}::{task}"] = {
            "domain": domain,
            "task": task,
            "count": len(items),
            "mean_delta": mean_delta,
            "mean_target_score": mean_target,
            "mean_actual_score": mean_actual,
            "by_target_score": target_stats,
        }

    # Overall stats
    all_deltas = [x["delta"] for x in valid_items]
    all_targets = [x["target_score"] for x in valid_items]
    all_actuals = [x["actual_score"] for x in valid_items]

    overall_by_target: Dict[int, List[float]] = defaultdict(list)
    for x in valid_items:
        overall_by_target[x["target_score"]].append(x["delta"])

    overall_by_target_stats = {}
    for t in sorted(overall_by_target.keys()):
        ds = overall_by_target[t]
        overall_by_target_stats[str(t)] = {
            "mean_delta": (sum(ds) / len(ds)) if ds else 0.0,
            "count": len(ds),
        }

    overall_stats = {
        "total_input_samples": total,
        "valid_samples": len(valid_items),
        "skipped_generation_failed": skipped_generation_failed,
        "skipped_judge_failed": skipped_judge_failed,
        "skipped_missing_target": skipped_missing_target,
        "skip_reason_breakdown": dict(skip_reason_counter),
        "mean_delta": (sum(all_deltas) / len(all_deltas)) if all_deltas else 0.0,
        "mean_target_score": (sum(all_targets) / len(all_targets)) if all_targets else 0.0,
        "mean_actual_score": (sum(all_actuals) / len(all_actuals)) if all_actuals else 0.0,
        "by_target_score": overall_by_target_stats,
    }

    return {
        "overall_stats": overall_stats,
        "by_domain_task": by_domain_task,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute delta metrics from judge results")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL file with judge results")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file for metrics")
    args = parser.parse_args()

    print(f"[INFO] Loading judge results from {args.input_jsonl}...", flush=True)
    samples = load_jsonl(Path(args.input_jsonl))
    print(f"[INFO] Loaded {len(samples)} samples", flush=True)

    print(f"[INFO] Computing metrics...", flush=True)
    metrics = compute_metrics(samples)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Metrics saved to: {output_path}", flush=True)

    print("\n[INFO] Summary:")
    print(f"  Total input samples: {metrics['overall_stats']['total_input_samples']}")
    print(f"  Valid samples: {metrics['overall_stats']['valid_samples']}")
    print(f"  Skipped generation_failed: {metrics['overall_stats']['skipped_generation_failed']}")
    print(f"  Skipped judge_failed: {metrics['overall_stats']['skipped_judge_failed']}")
    print(f"  Skipped missing_target: {metrics['overall_stats']['skipped_missing_target']}")
    print(f"  Overall mean delta: {metrics['overall_stats']['mean_delta']:.4f}")
    print(f"  Number of (domain, task) groups: {len(metrics['by_domain_task'])}")

    print("\n[INFO] Overall statistics by target score:")
    for target_score, stats in sorted(metrics["overall_stats"]["by_target_score"].items()):
        print(f"  Target={target_score}: mean_delta={stats['mean_delta']:.4f}, count={stats['count']}")

    print("\n[INFO] Statistics by (domain, task):")
    for key, stats in sorted(metrics["by_domain_task"].items()):
        print(f"  {key}: mean_delta={stats['mean_delta']:.4f}, count={stats['count']}")


if __name__ == "__main__":
    main()
