# artifact_generator.py

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import make_api_request, extract_json_from_text

from artifact_prompts import (
    RESEARCH_ARTIFACT_LABELS,
    TUTORING_ARTIFACT_LABELS,
    build_research_multiple_artifacts_prompt,
    build_tutoring_multiple_artifacts_prompt,
)


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
    reason: str  # 模型生成的 reason，说明为什么生成这个版本的 artifact


@dataclass
class ArtifactStore:
    """
    Holds latest artifact texts + version history per label for ONE timeline.
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
        self.current[v.artifact_label] = v.text
        self.history[v.artifact_label].append(v)
    
    def get_recent_versions(self, label: str, n: int = 2) -> List[ArtifactVersion]:
        """
        Get the most recent n versions of an artifact label.
        Returns empty list if label doesn't exist or has no history.
        """
        if label not in self.history:
            return []
        versions = self.history[label]
        # Return the last n versions (most recent first)
        return versions[-n:] if len(versions) > n else versions


# ======================
# IO helpers
# ======================

def load_events_from_run(run_dir: Path) -> List[Event]:
    """
    Load events from <run_dir>/events.json.
    """
    events_path = run_dir / "events.json"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events.json in run dir: {events_path}")

    with open(events_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"events.json must be a list, but got: {type(raw)}")

    events: List[Event] = [Event.from_dict(e) for e in raw]
    if not events:
        raise ValueError(f"events.json is empty in run dir: {run_dir}")

    # Basic normalization
    for i, ev in enumerate(events, start=1):
        if not ev.event_id:
            ev.event_id = f"e_{i:02d}"
        if not ev.time_index:
            ev.time_index = i
        if not ev.domain:
            ev.domain = "research"

    return events


def save_artifact_results_for_run(
    run_dir: Path,
    store: ArtifactStore,
    interactions: List[Dict[str, Any]],
    token_stats: Dict[str, int],
) -> Dict[str, str]:
    """
    Save per-label artifact files + store + interactions + stats in the run dir.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    # 1) Per-label versions: <label>.json
    for label, versions in store.history.items():
        out_path = run_dir / f"{label}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(v) for v in versions],
                f,
                ensure_ascii=False,
                indent=2,
            )
        saved[label] = str(out_path)

    # 2) artifact_store.json（current + history）
    store_path = run_dir / "artifact_store.json"
    # history 里是 dataclass，需要 asdict 处理
    history_as_dict: Dict[str, List[Dict[str, Any]]] = {
        label: [asdict(v) for v in versions]
        for label, versions in store.history.items()
    }
    store_serializable = {
        "current": store.current,
        "history": history_as_dict,
    }
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(store_serializable, f, ensure_ascii=False, indent=2)
    saved["artifact_store"] = str(store_path)

    # 3) interactions
    interactions_path = run_dir / "artifact_interactions.json"
    with open(interactions_path, "w", encoding="utf-8") as f:
        json.dump(interactions, f, ensure_ascii=False, indent=2)
    saved["artifact_interactions"] = str(interactions_path)

    # 4) stats
    stats_path = run_dir / "artifact_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(token_stats, f, ensure_ascii=False, indent=2)
    saved["artifact_stats"] = str(stats_path)

    return saved


def artifacts_already_generated(run_dir: Path) -> bool:
    """
    Skip condition: if we already have artifact_store.json,认为该 run 已生成过 artifacts。
    你也可以改成检查某一个 label.json。
    """
    p = run_dir / "artifact_store.json"
    return p.exists()


# ======================
# Validation
# ======================

def _validate_single_artifact_output(
    parsed: Dict[str, Any],
    event: Event,
    artifact_label: str,
) -> ArtifactVersion:
    """
    Ensure model output matches schema and the current event.
    The model output should have "artifact_text" and "reason" fields.
    """
    required_keys = ["artifact_text", "reason"]
    for k in required_keys:
        if k not in parsed:
            raise ValueError(f"Missing key '{k}' in artifact output: keys={list(parsed.keys())}")

    text = parsed["artifact_text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Field 'artifact_text' must be a non-empty string.")

    reason = parsed.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("Field 'reason' must be a string.")

    topic = event.topic or ""
    subject = event.subject or ""

    return ArtifactVersion(
        event_id=event.event_id,
        time_index=event.time_index,
        domain=event.domain,
        topic=topic,
        subject=subject,
        artifact_label=artifact_label,
        text=text.strip(),
        reason=reason.strip() if reason else "",
    )


def _validate_multiple_artifacts_output(
    parsed: Dict[str, Any],
    event: Event,
    artifact_labels: List[str],
) -> List[ArtifactVersion]:
    """
    Ensure model output matches schema for multiple artifacts.
    The model output should be a dictionary with artifact labels as keys.
    """
    if not isinstance(parsed, dict):
        raise ValueError(f"Output must be a dictionary, got {type(parsed)}")
    
    versions: List[ArtifactVersion] = []
    topic = event.topic or ""
    subject = event.subject or ""
    
    for label in artifact_labels:
        if label not in parsed:
            raise ValueError(f"Missing artifact '{label}' in output. Available: {list(parsed.keys())}")
        
        artifact_data = parsed[label]
        if not isinstance(artifact_data, dict):
            raise ValueError(f"Artifact '{label}' must be a dictionary, got {type(artifact_data)}")
        
        required_keys = ["artifact_text", "reason"]
        for k in required_keys:
            if k not in artifact_data:
                raise ValueError(f"Missing key '{k}' in artifact '{label}': keys={list(artifact_data.keys())}")
        
        text = artifact_data["artifact_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Field 'artifact_text' for '{label}' must be a non-empty string.")
        
        reason = artifact_data.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError(f"Field 'reason' for '{label}' must be a string.")
        
        versions.append(ArtifactVersion(
            event_id=event.event_id,
            time_index=event.time_index,
            domain=event.domain,
            topic=topic,
            subject=subject,
            artifact_label=label,
            text=text.strip(),
            reason=reason.strip() if reason else "",
        ))
    
    return versions


# ======================
# Core generation for ONE timeline (ONE run dir)
# ======================

def generate_research_artifacts_for_run(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    model: str = "gpt-5-mini",
    response_format_json: bool = True,
    temperature: float = 0.7,
    verbose: bool = False,
    sleep_s: float = 0.5,
) -> Dict[str, Any]:
    """
    For a single run dir:
    - load events.json from run_dir
    - sequentially generate artifacts per label across the timeline
    - save results into output_dir (if specified) or the same run directory
    Supports both 'research' and 'tutoring' domains.
    """
    events = load_events_from_run(run_dir)
    first = events[0]
    topic = first.topic or ""
    subject = first.subject or ""
    domain = first.domain or "research"

    # Select artifact labels and prompt builder based on domain
    if domain == "research":
        artifact_labels = RESEARCH_ARTIFACT_LABELS
        build_prompt_fn = build_research_multiple_artifacts_prompt
    elif domain == "tutoring":
        artifact_labels = TUTORING_ARTIFACT_LABELS
        build_prompt_fn = build_tutoring_multiple_artifacts_prompt
    else:
        raise ValueError(f"Unsupported domain: '{domain}'. Expected 'research' or 'tutoring'.")

    store = ArtifactStore.empty(artifact_labels)
    interactions: List[Dict[str, Any]] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # 历史事件简略信息（只用于 prompt context，保留最近的5个）
    prior_event_dicts: List[Dict[str, Any]] = []

    for ev in events:
        if verbose:
            print(
                f"[{run_dir.name}] Artifacts at event {ev.time_index:02d} ({ev.event_type})",
                flush=True,
            )

        if ev.event_type == "end":
            # 约定：end 事件不再更新任何 artifact，直接退出
            break

        # 1. 根据 required_artifacts，从 artifacts 状态库中检索需要的 artifacts（只保留最近的1个）
        required_artifacts_recent_versions: Dict[str, List[Dict[str, Any]]] = {}
        required_labels: List[str] = ev.required_artifacts or []
        for req_label in required_labels:
            if req_label in artifact_labels:
                recent_versions = store.get_recent_versions(req_label, n=1)
                # 转换为字典格式以便在 prompt 中使用
                required_artifacts_recent_versions[req_label] = [
                    {
                        "event_id": v.event_id,
                        "time_index": v.time_index,
                        "text": v.text,
                    }
                    for v in recent_versions
                ]

        # 2. 选取最近的5个 event 信息也加入上下文（已经在 prior_event_dicts 中维护）

        # 3. 根据 generated_artifacts 字段要求，一次性生成所有需要生成的 generated_artifacts
        gen_labels: List[str] = ev.generated_artifacts or []
        
        # 过滤掉不支持的 labels
        valid_gen_labels = [label for label in gen_labels if label in artifact_labels]
        if not valid_gen_labels:
            # 如果没有有效的 labels，跳过这个 event
            continue
        
        if len(gen_labels) != len(valid_gen_labels):
            # 如果有不支持的 labels，报错
            invalid_labels = [label for label in gen_labels if label not in artifact_labels]
            raise ValueError(f"Unsupported artifact labels: {invalid_labels}")
        
        # print("########################################################"*2)
        # print(f"Generating artifacts for event {ev.time_index:02d}: {valid_gen_labels}, run_dir: {run_dir.name}")
        # print("########################################################"*2)
        
        # 一次性生成所有 artifacts
        messages = build_prompt_fn(
            topic=topic,
            subject=subject,
            artifact_labels=valid_gen_labels,
            event={
                "event_id": ev.event_id,
                "time_index": ev.time_index,
                "domain": ev.domain,
                "event_type": ev.event_type,
                "description": ev.description,
                "reason": ev.reason,
                "required_artifacts": ev.required_artifacts,
                "generated_artifacts": ev.generated_artifacts,
            },
            prior_events=prior_event_dicts,
            required_artifacts_recent_versions=required_artifacts_recent_versions,
            current_artifact_state=store.current,
        )

        # 重试机制：最大重试5次，防止模型输出不符合JSON格式要求
        max_retries = 5
        retry_count = 0
        success = False
        content = None
        prompt_tokens = 0
        completion_tokens = 0
        _total = 0
        versions: List[ArtifactVersion] = []
        last_error = None

        while retry_count < max_retries and not success:
            try:
                # API 调用
                content, prompt_tokens, completion_tokens, _total = make_api_request(
                    messages=messages,
                    model=model,
                    response_format_json=response_format_json,
                    temperature=temperature,
                )

                # JSON 解析和验证
                parsed = extract_json_from_text(content)
                versions = _validate_multiple_artifacts_output(parsed, ev, valid_gen_labels)
                
                # 如果到这里没有异常，说明成功了
                success = True
                
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                # 捕获 JSON 解析错误、验证错误等
                retry_count += 1
                last_error = e
                if verbose:
                    print(
                        f"[{run_dir.name}] Event {ev.time_index:02d}, artifacts {valid_gen_labels}: "
                        f"Retry {retry_count}/{max_retries} due to error: {str(e)}",
                        flush=True,
                    )
                if retry_count < max_retries:
                    # 重试前等待一小段时间
                    time.sleep(sleep_s * retry_count)  # 指数退避
                else:
                    # 达到最大重试次数，抛出异常
                    raise ValueError(
                        f"Failed to generate valid artifacts {valid_gen_labels} for event {ev.event_id} "
                        f"after {max_retries} retries. Last error: {str(last_error)}"
                    ) from last_error
            except Exception as e:
                # 其他类型的错误（如 API 错误），不重试，直接抛出
                raise

        if not success or not versions:
            # 理论上不应该到这里，但为了安全起见
            raise ValueError(
                f"Failed to generate artifacts {valid_gen_labels} for event {ev.event_id} "
                f"after {max_retries} retries."
            )

        time.sleep(sleep_s)

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += _total

        # # 打印生成的 artifacts
        # print("--------------------------------------------------------"*2)
        # print("--------------------------------------------------------"*2)
        # print(f"Artifacts: {valid_gen_labels}")
        # print("--------------------------------------------------------"*2)
        # print(f"Context:\n{messages[-1]['content']}")
        # print("--------------------------------------------------------"*2)
        # for version in versions:
        #     print("--------------------------------------------------------"*2)
        #     print(f"Artifact: {version.artifact_label}")
        #     print(f"Reason: {version.reason}")
        #     print(f"Text: {version.text}")
        #     print("--------------------------------------------------------"*2)

        # 记录交互信息（为每个 artifact 分别记录）
        for version in versions:
            interactions.append(
                {
                    "run_dir": str(run_dir),
                    "time_index": ev.time_index,
                    "event_id": ev.event_id,
                    "event_type": ev.event_type,
                    "artifact_label": version.artifact_label,
                    "input_messages": messages,
                    "raw_output": content,
                    "tokens": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": _total,
                    },
                    "retry_count": retry_count,
                }
            )

        # 4. 继续更新 artifacts 状态，进行下一步
        for version in versions:
            store.apply_version(version)

        # 更新历史事件简略信息（保留最近的5个）
        prior_event_dicts.append(
            {
                "event_id": ev.event_id,
                "time_index": ev.time_index,
                "event_type": ev.event_type,
                "description": ev.description,
            }
        )
        # 只保留最近的5个事件
        if len(prior_event_dicts) > 5:
            prior_event_dicts = prior_event_dicts[-5:]

    token_stats = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "num_interactions": len(interactions),
    }

    # 确定保存路径：如果指定了 output_dir，则保存到 output_dir/{domain}/{run_id}/
    if output_dir is not None:
        # 从 run_dir 中提取 run_id（目录名）
        run_id = run_dir.name
        save_dir = output_dir / domain / run_id
    else:
        # 如果没有指定 output_dir，保存到原来的 run_dir
        save_dir = run_dir

    saved_files = save_artifact_results_for_run(
        run_dir=save_dir,
        store=store,
        interactions=interactions,
        token_stats=token_stats,
    )

    return {
        "run_dir": str(run_dir),
        "topic": topic,
        "subject": subject,
        "domain": domain,
        "success": True,
        "saved_files": saved_files,
        "token_stats": token_stats,
    }


# ======================
# Batch runner
# ======================

def _run_single(
    run_dir: Path,
    output_dir: Optional[Path],
    model: str,
    temperature: float,
    sleep_s: float,
    skip_existing: bool,
    verbose: bool,
    lock: threading.Lock,
) -> Dict[str, Any]:
    try:
        # 确定检查路径：如果指定了 output_dir，检查 output_dir/{domain}/{run_id}/
        if output_dir is not None:
            # 从 run_dir 中提取 run_id（目录名）
            run_id = run_dir.name
            # 需要先加载 events 来确定 domain
            try:
                events = load_events_from_run(run_dir)
                domain = events[0].domain or "research"
            except Exception:
                domain = "research"
            check_dir = output_dir / domain / run_id
        else:
            check_dir = run_dir

        if skip_existing and artifacts_already_generated(check_dir):
            with lock:
                print(f"[SKIP] {run_dir.name}: artifact_store.json already exists.", flush=True)
            return {
                "run_dir": str(run_dir),
                "success": True,
                "skipped": True,
            }

        result = generate_research_artifacts_for_run(
            run_dir=run_dir,
            output_dir=output_dir,
            model=model,
            temperature=temperature,
            sleep_s=sleep_s,
            verbose=verbose,
        )

        with lock:
            print(
                f"[OK] {run_dir.name}: artifacts generated. Total interactions={result['token_stats']['num_interactions']}",
                flush=True,
            )

        result["skipped"] = False
        return result

    except Exception as e:
        with lock:
            print(f"[ERR] {run_dir.name}: {e}", flush=True)
        return {
            "run_dir": str(run_dir),
            "success": False,
            "error": str(e),
        }


if __name__ == "__main__":
    """
    Example usage:

      python artifact_generator.py \\
        --input_dir output/research \\
        --model gpt-5-mini \\
        --max_workers 4 \\
        --skip_existing

    约定：
    - input_dir 下的每个子目录（目录名是数字字符串，比如 "0", "1", ...）是一个 run。
    - 每个 run 目录中必须有 events.json。
    - 本脚本会在每个 run 目录内生成各类 artifact 的 json 文件。
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate per-artifact timelines for research domain from existing events.json files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing subdirectories for each run (e.g., output/research).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for artifacts. If not specified, artifacts will be saved in the same directory as events.json. Structure: output_dir/research/{run_id}/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model name to use.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of concurrent threads.",
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
        help="If set, skip runs that already have artifact_store.json.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-event progress (may be noisy).",
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

    args = parser.parse_args()

    base = Path(args.input_dir)
    if not base.exists():
        raise FileNotFoundError(f"input_dir does not exist: {base}")

    # 只选择子目录名为数字的目录，按数值升序排序
    all_run_dirs: List[Path] = []
    for p in base.iterdir():
        if p.is_dir() and p.name.isdigit():
            all_run_dirs.append(p)
    all_run_dirs.sort(key=lambda d: int(d.name))

    if not all_run_dirs:
        raise RuntimeError(f"No numeric subdirectories found under input_dir={base}")

    if args.limit is not None:
        run_dirs = all_run_dirs[args.start_from: args.start_from + args.limit]
    else:
        run_dirs = all_run_dirs[args.start_from:]

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Artifact Generation (research) ===")
    print(f"input_dir: {base}")
    if output_dir is not None:
        print(f"output_dir: {output_dir}")
    else:
        print("output_dir: (same as input_dir)")
    print(f"Total run dirs found: {len(all_run_dirs)}")
    print(f"Run dirs to process: {len(run_dirs)}")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    print(f"Skip existing: {args.skip_existing}")
    print()

    lock = threading.Lock()
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                _run_single,
                run_dir=rd,
                output_dir=output_dir,
                model=args.model,
                temperature=args.temperature,
                sleep_s=args.sleep_s,
                skip_existing=args.skip_existing,
                verbose=args.verbose,
                lock=lock,
            )
            for rd in run_dirs
        ]

        for fut in as_completed(futures):
            results.append(fut.result())

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    skipped = [r for r in successful if r.get("skipped")]

    print("\n=== Summary ===")
    print(f"Total runs processed: {len(results)}")
    print(f"Successful: {len(successful)} (skipped: {len(skipped)})")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed runs:")
        for r in failed:
            print(f"  - {r['run_dir']}: {r.get('error', 'Unknown error')}")