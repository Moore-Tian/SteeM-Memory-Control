# memory_builder.py
# Build multi-level simulated memory (core / episodic / retrieved)
# from timeline events + artifacts.
#
# New assumed layout:
#
# input_dir/
#   0/
#     events.json
#     artifacts/
#       research_plan.json
#       research_goals.json
#       method_scheme.json
#       experiment_results.json
#       paper_paragraph.json
#       teaching_notes.json
#       practice_record.json
#       feedback_summary.json
#       (可能还有其他统计类 json，会被过滤掉)
#   1/
#     events.json
#     artifacts/
#       ...
#
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# 只把这些文件视为「版本化 artifact」
VALID_ARTIFACT_LABELS = {
    "research_plan",
    "research_goals",
    "experiment_results",
    "method_scheme",
    "paper_paragraph",
    "learning_objectives",
    "study_plan",
    "teaching_notes",
    "practice_record",
    "feedback_summary",
}

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
class ArtifactVersion:
    """
    A single version of one artifact at one event.
    """
    event_id: str
    time_index: int
    domain: str
    topic: str
    subject: str
    artifact_label: str
    text: str
    reason: str


@dataclass
class ArtifactStore:
    """
    Holds latest artifact texts + version history per label for ONE scenario + ONE domain.
    """
    current: Dict[str, Optional[str]]
    history: Dict[str, List[ArtifactVersion]]

    @staticmethod
    def empty(labels: List[str]) -> "ArtifactStore":
        return ArtifactStore(
            current={lbl: None for lbl in labels},
            history={lbl: [] for lbl in labels},
        )

    def apply_version(self, v: ArtifactVersion) -> None:
        # 动态新增 label 也可以
        self.current[v.artifact_label] = v.text
        self.history.setdefault(v.artifact_label, []).append(v)

    def get_recent_versions(self, label: str, n: int = 2) -> List[ArtifactVersion]:
        if label not in self.history:
            return []
        versions = self.history[label]
        return versions[-n:] if len(versions) > n else versions


# ======================
# Helpers: load events / artifacts
# ======================

def load_events(events_path: str) -> List[Event]:
    raw = json.load(open(events_path, "r", encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"events.json must be a list: {events_path}")
    events = [Event.from_dict(x) for x in raw]
    events.sort(key=lambda e: e.time_index)
    return events


def load_artifact_versions_from_dir(dir_path: str) -> List[ArtifactVersion]:
    """
    从某个 scenario 的 artifacts 目录中加载所有版本化的 artifacts。
    只读取文件名在 VALID_ARTIFACT_LABELS 中的 *.json。
    """
    versions: List[ArtifactVersion] = []

    for fp in glob.glob(os.path.join(dir_path, "*.json")):
        base = os.path.splitext(os.path.basename(fp))[0]
        if base not in VALID_ARTIFACT_LABELS:
            # 跳过 artifact_store / artifact_stats 等统计文件
            continue

        raw = json.load(open(fp, "r", encoding="utf-8"))
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            continue

        file_inferred_label = base

        for d in raw:
            if not isinstance(d, dict):
                continue
            artifact_label = d.get("artifact_label") or file_inferred_label
            versions.append(
                ArtifactVersion(
                    event_id=str(d.get("event_id", "")),
                    time_index=int(d.get("time_index", 0)),
                    domain=str(d.get("domain", "")),
                    topic=str(d.get("topic", "")),
                    subject=str(d.get("subject", "")),
                    artifact_label=str(artifact_label),
                    text=str(d.get("text", "")),
                    reason=str(d.get("reason", "")),
                )
            )

    versions.sort(key=lambda v: v.time_index)
    return versions


def build_artifact_store(versions: List[ArtifactVersion]) -> ArtifactStore:
    labels = sorted({v.artifact_label for v in versions})
    store = ArtifactStore.empty(labels)
    for v in versions:
        store.apply_version(v)
    return store


# ======================
# Helpers: time-truncated retrieval
# ======================

def _latest_before(versions: List[ArtifactVersion], t: int) -> Optional[ArtifactVersion]:
    """
    返回 time_index <= t 的最后一个版本（假设已按时间升序排序）。
    """
    best: Optional[ArtifactVersion] = None
    for v in versions:
        if v.time_index <= t:
            best = v
        else:
            break
    return best


def get_latest_artifact_before(store: ArtifactStore, label: str, t: int) -> Optional[ArtifactVersion]:
    if label not in store.history:
        return None
    return _latest_before(store.history[label], t)


def get_recent_artifacts_before(store: ArtifactStore, label: str, t: int, n: int = 2) -> List[ArtifactVersion]:
    if label not in store.history:
        return []
    eligible = [v for v in store.history[label] if v.time_index <= t]
    return eligible[-n:] if len(eligible) > n else eligible


# ======================
# Memory builder
# ======================

def build_memory_for_anchor(
    events: List[Event],
    store_by_domain: Dict[str, ArtifactStore],
    anchor_idx: int,
    *,
    episodic_k: int = 5,
    max_retrieved: int = 8,
    core_memory: Optional[Dict[str, Any]] = None,
    include_anchor_in_episodic: bool = False,
    topic_strict_match: bool = True,
) -> Dict[str, Any]:
    """
    在给定 anchor event 处构造 memory snapshot。

    - episodic_memory: 过去 k 个 event 的 description
    - retrieved_memory: anchor 发生时刻之前的、按 domain/topic 过滤后的 artifact 版本
    - core_memory: 由你注入的用户画像等静态信息
    """
    if anchor_idx < 0 or anchor_idx >= len(events):
        raise IndexError(f"anchor_idx out of range: {anchor_idx}")

    anchor = events[anchor_idx]
    t = anchor.time_index

    # Episodic window
    start = max(0, anchor_idx - episodic_k)
    end = anchor_idx + 1 if include_anchor_in_episodic else anchor_idx
    episodic_slice = events[start:end]

    episodic_memory = [
        {
            "mem_type": "episodic_event",
            "event_id": e.event_id,
            "time_index": e.time_index,
            "domain": e.domain,
            "event_type": e.event_type,
            "text": e.description,
        }
        for e in episodic_slice
    ]

    # Retrieved artifacts (per domain)
    store = store_by_domain.get(anchor.domain)
    retrieved_memory: List[Dict[str, Any]] = []
    seen_labels: set[str] = set()

    def push(v: ArtifactVersion) -> None:
        retrieved_memory.append(
            {
                "mem_type": "retrieved_artifact",
                "artifact_label": v.artifact_label,
                "event_id": v.event_id,
                "time_index": v.time_index,
                "domain": v.domain,
                "topic": v.topic,
                "subject": v.subject,
                "text": v.text,
            }
        )

    if store:
        # 1) 优先放当前 event 要求的 artifacts
        for lbl in (anchor.required_artifacts or []):
            v = get_latest_artifact_before(store, lbl, t)
            if v:
                if topic_strict_match and anchor.topic and v.topic and v.topic != anchor.topic:
                    continue
                push(v)
                seen_labels.add(lbl)
            if len(retrieved_memory) >= max_retrieved:
                break

        # 2) 再从同 domain 的其他 label 中回填
        if len(retrieved_memory) < max_retrieved:
            cands: List[ArtifactVersion] = []
            for lbl, versions in store.history.items():
                if lbl in seen_labels:
                    continue
                v = _latest_before(versions, t)
                if not v:
                    continue
                if topic_strict_match and anchor.topic and v.topic and v.topic != anchor.topic:
                    continue
                cands.append(v)

            cands.sort(key=lambda x: x.time_index, reverse=True)
            for v in cands:
                push(v)
                if len(retrieved_memory) >= max_retrieved:
                    break

    return {
        "anchor": {
            "event_id": anchor.event_id,
            "time_index": anchor.time_index,
            "domain": anchor.domain,
            "event_type": anchor.event_type,
            "topic": anchor.topic,
            "subject": anchor.subject,
        },
        "core_memory": core_memory or {},
        "episodic_memory": episodic_memory,
        "retrieved_memory": retrieved_memory,
    }


# ======================
# Scenario-level helpers
# ======================

DIGIT_DIR_RE = re.compile(r"^\d+$")


def list_scenario_ids(input_dir: str) -> List[str]:
    ids: List[str] = []
    for name in os.listdir(input_dir):
        p = os.path.join(input_dir, name)
        if os.path.isdir(p) and DIGIT_DIR_RE.match(name):
            ids.append(name)
    return sorted(ids, key=lambda x: int(x))


def build_store_by_domain_for_scenario(
    scenario_dir: str,
    domains: Tuple[str, ...] = ("research", "tutoring"),
) -> Dict[str, ArtifactStore]:
    """
    从 scenario_dir/artifacts/ 读取所有 artifacts，并按 domain 分组。
    """
    artifacts_dir = os.path.join(scenario_dir, "artifacts")
    if not os.path.isdir(artifacts_dir):
        # 没有 artifacts 也可以继续，只是检索为空
        return {}

    versions = load_artifact_versions_from_dir(artifacts_dir)
    stores: Dict[str, ArtifactStore] = {}

    for v in versions:
        if domains and v.domain not in domains:
            continue
        store = stores.setdefault(v.domain, ArtifactStore.empty([]))
        store.apply_version(v)

    return stores


def build_all_memories(
    input_dir: str,
    output_dir: str,
    *,
    episodic_k: int = 5,
    max_retrieved: int = 8,
    include_anchor_in_episodic: bool = False,
    topic_strict_match: bool = True,
    domains: Tuple[str, ...] = ("research", "tutoring"),
) -> None:
    """
    对每个 scenario：
    - 读 <scenario>/events.json
    - 读 <scenario>/artifacts/*.json
    - 为每个 event 写一份 memory snapshot
    """
    os.makedirs(output_dir, exist_ok=True)

    for sid in list_scenario_ids(input_dir):
        scenario_dir = os.path.join(input_dir, sid)
        events_path = os.path.join(scenario_dir, "events.json")
        if not os.path.isfile(events_path):
            raise FileNotFoundError(f"Missing events.json: {events_path}")

        events = load_events(events_path)
        store_by_domain = build_store_by_domain_for_scenario(scenario_dir, domains=domains)

        out_sdir = os.path.join(output_dir, sid)
        os.makedirs(out_sdir, exist_ok=True)

        for i in range(len(events)):
            mem = build_memory_for_anchor(
                events,
                store_by_domain,
                i,
                episodic_k=episodic_k,
                max_retrieved=max_retrieved,
                core_memory={"persona": "replace_me_with_your_profile"},
                include_anchor_in_episodic=include_anchor_in_episodic,
                topic_strict_match=topic_strict_match,
            )

            anchor = events[i]
            out_path = os.path.join(
                out_sdir, f"{anchor.time_index:04d}_{anchor.event_id}.memory.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(mem, f, ensure_ascii=False, indent=2)


# ======================
# CLI
# ======================

if __name__ == "__main__":
    """
    Example:

    python memory_builder.py \
        --input_dir /path/to/framework_data/scenarios \
        --output_dir /path/to/framework_data/memories
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root dir containing scenario subdirs (0/,1/,...) each with events.json and artifacts/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to write per-event memory snapshots.",
    )
    parser.add_argument("--episodic_k", type=int, default=5)
    parser.add_argument("--max_retrieved", type=int, default=8)
    parser.add_argument(
        "--include_anchor_in_episodic",
        action="store_true",
        help="If set, include anchor event itself in episodic_memory window.",
    )
    parser.add_argument(
        "--no_topic_strict_match",
        action="store_true",
        help="Disable topic-strict filtering when retrieving artifacts.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="research,tutoring",
        help="Comma-separated domains to load artifacts from.",
    )
    args = parser.parse_args()

    domains = tuple([x.strip() for x in args.domains.split(",") if x.strip()])

    build_all_memories(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        episodic_k=args.episodic_k,
        max_retrieved=args.max_retrieved,
        include_anchor_in_episodic=args.include_anchor_in_episodic,
        topic_strict_match=not args.no_topic_strict_match,
        domains=domains,
    )