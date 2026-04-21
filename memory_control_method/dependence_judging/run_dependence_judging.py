import argparse
import json
import os
import random
import re
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUBRICS_PATH = METHOD_ROOT / "rubrics" / "dependence_rubrics_text.py"
spec = importlib.util.spec_from_file_location("_dependence_rubrics_text", RUBRICS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load rubrics module from {RUBRICS_PATH}")
rubrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rubrics_module)
RUBRICS_TEXT = getattr(rubrics_module, "RUBRICS_TEXT", "")
if not isinstance(RUBRICS_TEXT, str) or not RUBRICS_TEXT.strip():
    raise ImportError(f"Could not extract RUBRICS_TEXT from {RUBRICS_PATH}")


# =========================================================
# 通用工具
# =========================================================
def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        assert isinstance(data, list)
        return data
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def robust_json_from_text(text: str) -> Optional[dict]:
    """
    从模型输出中尽量截取第一个 JSON 对象。
    """
    if text is None:
        return None

    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    snippet = text[first : last + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


# =========================================================
# phase=prepare : 仅基于 answers 生成 judge_inputs.jsonl
# 支持在已有采样基础上追加（--base-judge-inputs）
# =========================================================
def phase_prepare(args):
    answers_path = Path(args.answers)
    out_path = Path(args.out)
    n_per_task = args.n_per_task
    random.seed(args.seed)

    print(f"[INFO] Loading answers from: {answers_path}")
    answers = load_json_or_jsonl(answers_path)
    print(f"[INFO] Answer entries: {len(answers)}")

    # 若指定 base-judge-inputs，则加载已有 (task, query_id)，避免重复采样
    base_used_pairs = set()
    if args.base_judge_inputs:
        base_path = Path(args.base_judge_inputs)
        if base_path.exists():
            print(f"[INFO] Loading base judge inputs from: {base_path}")
            base_items = load_json_or_jsonl(base_path)
            for item in base_items:
                t = item.get("task")
                qid = item.get("query_id")
                if t is not None and qid is not None:
                    base_used_pairs.add((t, qid))
            print(f"[INFO] Existing (task, query_id) pairs in base: {len(base_used_pairs)}")
        else:
            print(f"[WARN] base_judge_inputs not found: {base_path}, ignored.")

    # 按 query_key 分组：优先用 query_id；没有则退化为 uid 前缀
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for ans in answers:
        qid = ans.get("query_id")
        if qid:
            query_key = qid
        else:
            uid = ans["uid"]
            query_key = uid.split("::", 1)[0]
        grouped[query_key].append(ans)

    print(f"[INFO] Unique query keys in answers: {len(grouped)}")

    # 按 task 分组 query
    task_groups: Dict[str, List[Tuple[str, List[dict]]]] = defaultdict(list)
    incomplete_queries = 0

    for qkey, ans_list in grouped.items():
        meta0 = ans_list[0]
        task = meta0.get("task")
        if task is None:
            task = "UNKNOWN"

        # 检查模式/样本情况（不丢弃，仅统计）
        by_mode = defaultdict(list)
        for a in ans_list:
            by_mode[a.get("mode", "none")].append(a)

        missing_modes = []
        for m in ["none", "low", "medium", "high"]:
            sids = sorted([int(x.get("sample_id", -1)) for x in by_mode.get(m, [])])
            if sids != [0, 1, 2, 3]:
                missing_modes.append(m)

        if missing_modes:
            incomplete_queries += 1
            print(f"[WARN] Query {qkey} has incomplete samples for modes: {missing_modes}")

        task_groups[task].append((qkey, ans_list))

    total_valid_queries = sum(len(v) for v in task_groups.values())
    print(f"[INFO] Total query groups (all kept, may be incomplete): {total_valid_queries}")
    print(f"[INFO] Query groups with some missing modes/samples: {incomplete_queries}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w", encoding="utf-8")

    # 每个 task 抽取 n_per_task 个 query（如 n_per_task 大于剩余数量则全取）
    for task, qlist in task_groups.items():
        if not qlist:
            continue

        # 剔除 base 中已使用过的 (task, query_id)
        filtered = []
        for qkey, ans_list in qlist:
            pair = (task, qkey)
            if pair in base_used_pairs:
                continue
            filtered.append((qkey, ans_list))

        if not filtered:
            print(f"[INFO] Task={task}: no remaining queries after excluding base.")
            continue

        if len(filtered) <= n_per_task:
            selected = filtered
        else:
            selected = random.sample(filtered, n_per_task)

        print(
            f"[INFO] Task={task}: selected {len(selected)} queries "
            f"(from {len(filtered)} remaining, original {len(qlist)})"
        )

        for qkey, ans_list in selected:
            meta0 = ans_list[0]
            query = meta0.get("query")
            system_prompt = meta0.get("system_prompt") or ""
            event_id = meta0.get("event_id")
            dindex = meta0.get("directory_index")

            for ans in ans_list:
                base_answer = (
                    ans.get("answer")
                    or ans.get("response")
                    or ans.get("raw")
                    or ""
                )
                think_text = ans.get("think")
                if think_text:
                    combined_answer = f"<think>{think_text}</think>\n\n{base_answer}"
                else:
                    combined_answer = base_answer

                rec = {
                    "uid": ans["uid"],
                    "query_id": ans.get("query_id", qkey),
                    "event_id": event_id,
                    "directory_index": dindex,
                    "domain": meta0.get("domain"),
                    "task": task,
                    "mode": ans.get("mode"),
                    "sample_id": ans.get("sample_id"),
                    "query": query,
                    "context": system_prompt,
                    "answer": combined_answer,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fout.close()
    print(f"[INFO] Judge input written to: {out_path}")


# =========================================================
# phase=judge : 多卡并行 judge dependence score
# =========================================================
def phase_judge(args):
    from vllm import LLM, SamplingParams

    judge_data_path = Path(args.judge_data)
    out_path = Path(args.out)
    shard_id = args.shard_id
    num_shards = args.num_shards

    max_retries = max(0, args.max_retries)
    max_attempts = 1 + max_retries  # 总尝试次数上限 = 首次 + 重试次数

    print(f"[INFO] Loading judge data from: {judge_data_path}")
    all_items = load_json_or_jsonl(judge_data_path)
    print(f"[INFO] Total judge items: {len(all_items)}")

    shard_items = [x for i, x in enumerate(all_items) if i % num_shards == shard_id]
    print(f"[INFO] Shard {shard_id}/{num_shards}: items = {len(shard_items)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 断点续跑 + invalid 重试 ----------
    done_uids = set()
    uid_attempt_counts: Dict[str, int] = defaultdict(int)
    hard_failed_records: List[dict] = []

    if out_path.exists():
        existing_by_uid: Dict[str, List[dict]] = defaultdict(list)
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                uid = obj.get("uid")
                if uid is None:
                    continue
                existing_by_uid[uid].append(obj)

        # 统计已有尝试次数
        for uid, recs in existing_by_uid.items():
            uid_attempt_counts[uid] = len(recs)

        print(
            f"[INFO] Shard {shard_id}: existing uids = {len(existing_by_uid)}, "
            f"with attempts in [min={min(uid_attempt_counts.values() or [0])}, "
            f"max={max(uid_attempt_counts.values() or [0])}]"
        )

        if args.retry_invalid:
            kept_records: List[dict] = []
            invalid_to_retry = 0
            invalid_maxed = 0

            for uid, recs in existing_by_uid.items():
                attempts = len(recs)
                valid_recs = [
                    r for r in recs
                    if r.get("judge_parse_ok")
                    and r.get("overall_memory_dependence_score") is not None
                ]

                if valid_recs:
                    # 保留最后一次有效结果
                    final_rec = valid_recs[-1]
                    kept_records.append(final_rec)
                    done_uids.add(uid)
                else:
                    # 没有任何有效结果
                    if attempts >= max_attempts:
                        # 超过最大尝试次数：视为 hard failed，保留最后一次，并额外写入 failed_out
                        final_rec = recs[-1]
                        final_rec["max_retry_exceeded"] = True
                        kept_records.append(final_rec)
                        hard_failed_records.append(final_rec)
                        done_uids.add(uid)
                        invalid_maxed += 1
                    else:
                        # 还可以 retry：不保留旧记录，交给后续重新 judge
                        invalid_to_retry += 1

            print(
                f"[INFO] Shard {shard_id}: valid kept = {len(done_uids)}, "
                f"invalid to retry = {invalid_to_retry}, "
                f"invalid exceeded max_attempts = {invalid_maxed}"
            )

            # 覆盖写回仅包含“有效 + 超过最大尝试的无效”记录
            with out_path.open("w", encoding="utf-8") as fout:
                for obj in kept_records:
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # 如果指定了 failed_out，则把 hard_failed_records 额外写入（按 uid 去重）
            if args.failed_out and hard_failed_records:
                failed_path = Path(args.failed_out)
                failed_path.parent.mkdir(parents=True, exist_ok=True)
                existing_failed_uids = set()
                if failed_path.exists():
                    with failed_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            uid = obj.get("uid")
                            if uid is not None:
                                existing_failed_uids.add(uid)
                with failed_path.open("a", encoding="utf-8") as f:
                    for rec in hard_failed_records:
                        uid = rec.get("uid")
                        if uid is None or uid in existing_failed_uids:
                            continue
                        existing_failed_uids.add(uid)
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    f"[INFO] Shard {shard_id}: hard-failed records appended to {failed_path}, "
                    f"total failed_uids now = {len(existing_failed_uids)}"
                )

        else:
            # 不重跑 invalid：所有 uid 视为 done
            for uid in existing_by_uid.keys():
                done_uids.add(uid)
            print(
                f"[INFO] Shard {shard_id}: retry_invalid=False, "
                f"all existing {len(done_uids)} uids treated as done."
            )
    else:
        print(f"[INFO] Shard {shard_id}: no existing output, start fresh")

    # rubrics 采用自然语言文本
    rubrics_text = RUBRICS_TEXT

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1,
    )

    # 这里必须用 "a" 追加，因为上面可能已经重写了有效记录
    fout = out_path.open("a", encoding="utf-8", buffering=1)
    pbar = tqdm(total=len(shard_items), desc=f"Judge shard {shard_id}", dynamic_ncols=True)

    buf_prompts: List[str] = []
    buf_meta: List[dict] = []

    def build_judge_prompt(item: dict) -> str:
        """
        使用 chat_template 构造 judge prompt。
        """
        system_prompt = (
            "You are an expert evaluator of how strongly a response depends on "
            "the given memory / project history / user profile.\n\n"
            "You are given a rubric written in natural language describing how "
            "to score the memory dependence of an answer relative to its "
            "context/history.\n\n"
            "RUBRIC (natural language description):\n"
            f"{rubrics_text}\n\n"
            "You MUST output a single JSON object that strictly follows the "
            "schema described in the rubric as 'global_instructions.output_schema'. "
            "Do NOT output any text before or after the JSON. Do NOT use code fences."
        )

        user_prompt = (
            "Please evaluate how strongly the following ANSWER depends on the "
            "provided MEMORY / CONTEXT, according to the rubric.\n\n"
            f"TASK TYPE:\n{item.get('task')}\n\n"
            "MEMORY / CONTEXT (includes user profile, cross-session summaries, "
            "recent events, and any relevant artifacts if present):\n"
            f"{item.get('context')}\n\n"
            "USER QUERY:\n"
            f"{item.get('query')}\n\n"
            "ANSWER TO EVALUATE (may contain internal thinking segments like "
            "<think>...</think>):\n"
            f"{item.get('answer')}\n\n"
            "Now follow the rubric and produce your evaluation as a JSON object."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def flush():
        nonlocal buf_prompts, buf_meta
        if not buf_prompts:
            return
        outputs = llm.generate(buf_prompts, sampling)
        for out, meta in zip(outputs, buf_meta):
            raw_text = out.outputs[0].text
            judge_json = robust_json_from_text(raw_text)

            record = dict(meta)
            record["judge_raw"] = raw_text

            if judge_json is None:
                record["judge_parse_ok"] = False
                record["dimension_scores"] = None
                record["overall_memory_dependence_score"] = None
                record["rationale"] = None
            else:
                record["judge_parse_ok"] = True
                record["dimension_scores"] = judge_json.get("dimension_scores")
                record["overall_memory_dependence_score"] = judge_json.get(
                    "overall_memory_dependence_score"
                )
                record["rationale"] = judge_json.get("rationale")

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.update(1)

        buf_prompts.clear()
        buf_meta.clear()

    for item in shard_items:
        uid = item["uid"]
        if uid in done_uids:
            # 已有“完成状态”的样本直接略过，但进度条仍前进
            pbar.update(1)
            continue

        # 根据历史尝试次数，更新 attempt_index
        attempts_so_far = uid_attempt_counts.get(uid, 0)
        attempt_index = attempts_so_far + 1

        item_with_attempt = dict(item)
        item_with_attempt["attempt_index"] = attempt_index

        buf_prompts.append(build_judge_prompt(item))
        buf_meta.append(item_with_attempt)

        if len(buf_prompts) >= args.batch_size:
            flush()

    flush()
    fout.close()
    pbar.close()

    print(f"[INFO] Shard {shard_id} judge finished. Output: {out_path}")

# =========================================================
# phase=analyze : 统计 & 可视化
#   支持 --blacklist-answers，将该 json/jsonl 中出现的 uid 作为黑名单剔除
#   新增：
#     - 记录 domain / query_id
#     - 输出 domain × task × score 层面的分布统计
# =========================================================
def phase_analyze(args):
    import pandas as pd
    import matplotlib.pyplot as plt

    judged_path = Path(args.judged)
    outdir = Path(args.analysis_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading judged results from: {judged_path}")
    data = load_json_or_jsonl(judged_path)
    print(f"[INFO] Total judged entries: {len(data)}")

    # -------- 黑名单：来自 answers 文件（如 all_answers_with_query_id.not_trunc.jsonl） --------
    blacklist_uids: Set[str] = set()
    if getattr(args, "blacklist_answers", None):
        bl_path = Path(args.blacklist_answers)
        if bl_path.exists():
            print(f"[INFO] Loading blacklist answers from: {bl_path}")
            bl_data = load_json_or_jsonl(bl_path)
            for obj in bl_data:
                uid = obj.get("uid")
                if uid:
                    blacklist_uids.add(uid)
            print(f"[INFO] Blacklist size (unique uids): {len(blacklist_uids)}")
        else:
            print(f"[WARN] Blacklist answers file not found: {bl_path}, ignored.")

    rows = []
    skipped_no_score = 0
    skipped_bad_type = 0
    skipped_blacklist = 0

    def extract_int_score(raw_score):
        """
        尽量从 raw_score 中提取一个整数分数。
        支持：
          - 直接是 int/float/str
          - 是 dict 时，从常见 key 中找出一个标量再转 int
        若失败则返回 None。
        """
        # 1) 直接是标量
        if isinstance(raw_score, (int, float, str)):
            try:
                return int(raw_score)
            except Exception:
                return None

        # 2) 是 dict：尝试几个常见 key
        if isinstance(raw_score, dict):
            for k in [
                "overall_memory_dependence_score",
                "score",
                "value",
                "rating",
            ]:
                if k in raw_score:
                    v = raw_score[k]
                    if isinstance(v, (int, float, str)):
                        try:
                            return int(v)
                        except Exception:
                            continue
            return None

        # 3) 其它类型直接视为失败
        return None

    for x in data:
        if not x.get("judge_parse_ok"):
            skipped_no_score += 1
            continue

        raw_score = x.get("overall_memory_dependence_score")
        if raw_score is None:
            skipped_no_score += 1
            continue

        score_int = extract_int_score(raw_score)
        if score_int is None:
            skipped_bad_type += 1
            continue

        uid = x.get("uid")
        if blacklist_uids and uid in blacklist_uids:
            skipped_blacklist += 1
            continue

        # 这里记录 domain / query_id，方便后面做 domain × task × score 统计
        domain = x.get("domain") or "UNKNOWN"
        task = x.get("task") or "UNKNOWN"

        qid = x.get("query_id")
        if not qid and uid:
            # 退化：从 uid 前缀恢复 query 粒度
            qid = uid.split("::", 1)[0]

        rows.append({
            "uid": uid,
            "query_id": qid,
            "domain": domain,
            "task": task,
            "mode": x.get("mode"),
            "score": score_int,
        })

    print(f"[INFO] Valid scored rows (after blacklist): {len(rows)}")
    print(f"[INFO] Skipped rows (no score / parse_not_ok): {skipped_no_score}")
    print(f"[INFO] Skipped rows (bad score type / unconvertible): {skipped_bad_type}")
    print(f"[INFO] Skipped rows (in blacklist): {skipped_blacklist}")

    if not rows:
        print("[WARN] No valid rows to analyze. Exiting analyze phase.")
        return

    df = pd.DataFrame(rows)

    # ----------------- 全局分布统计（描述性统计） -----------------
    overall_stats = df["score"].describe()
    print("[INFO] Overall score stats:")
    print(overall_stats)

    overall_stats.to_csv(outdir / "overall_score_stats.csv")

    # ----------------- 每个分值的样本数 + 百分比 -----------------
    total = len(df)
    vc = df["score"].value_counts().sort_index()

    print("\n[INFO] Per-score counts and percentages (overall):")
    per_score_rows = []
    for s in range(1, 6):
        count = int(vc.get(s, 0))
        pct = (count / total * 100.0) if total > 0 else 0.0
        print(f"  Score = {s}: count = {count:6d}, "
              f"percentage = {pct:8.4f}%")
        per_score_rows.append({
            "score": s,
            "count": count,
            "percentage": pct,
        })

    per_score_df = pd.DataFrame(per_score_rows)
    per_score_df.to_csv(outdir / "overall_score_per_level.csv", index=False)

    # ----------------- 按 score 统计：记录数 & 去重 query 数 -----------------
    score_query_rows = []
    for s, sub in df.groupby("score"):
        num_records = len(sub)
        num_queries = sub["query_id"].nunique()
        avg_rec_per_query = num_records / num_queries if num_queries > 0 else 0.0
        score_query_rows.append({
            "score": s,
            "num_records": num_records,
            "num_queries": num_queries,
            "avg_records_per_query": avg_rec_per_query,
        })
    score_query_df = pd.DataFrame(score_query_rows).sort_values("score")
    score_query_df.to_csv(outdir / "score_query_summary.csv", index=False)

    # ----------------- 按 (domain, task, score) 统计：主要给后续抽样策略用 -----------------
    dts_rows = []
    grouped_dts = df.groupby(["domain", "task", "score"])
    for (dom, task, s), sub in grouped_dts:
        num_records = len(sub)
        num_queries = sub["query_id"].nunique()
        avg_rec_per_query = num_records / num_queries if num_queries > 0 else 0.0
        dts_rows.append({
            "domain": dom,
            "task": task,
            "score": s,
            "num_records": num_records,
            "num_queries": num_queries,
            "avg_records_per_query": avg_rec_per_query,
        })

    dts_df = pd.DataFrame(dts_rows).sort_values(
        ["domain", "task", "score"]
    )
    dts_df.to_csv(outdir / "domain_task_score_distribution.csv", index=False)

    # ----------------- 按 (domain, task) 总量统计（跨 score 聚合） -----------------
    dt_rows = []
    grouped_dt = df.groupby(["domain", "task"])
    for (dom, task), sub in grouped_dt:
        dt_rows.append({
            "domain": dom,
            "task": task,
            "total_records": len(sub),
            "total_queries": sub["query_id"].nunique(),
        })
    dt_df = pd.DataFrame(dt_rows).sort_values(["domain", "task"])
    dt_df.to_csv(outdir / "domain_task_overall_summary.csv", index=False)

    # ----------------- 画图函数 -----------------
    def plot_distribution(sub_df: pd.DataFrame, title: str, fname_prefix: str):
        if sub_df.empty:
            return
        plt.figure()
        sub_df["score"].hist(bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        plt.xlabel("overall_memory_dependence_score")
        plt.ylabel("count")
        plt.title(title)
        plt.xticks([1, 2, 3, 4, 5])
        plt.tight_layout()
        plt.savefig(outdir / f"{fname_prefix}_hist.png")
        plt.close()

        plt.figure()
        sub_df["score"].plot(kind="box")
        plt.ylabel("overall_memory_dependence_score")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outdir / f"{fname_prefix}_box.png")
        plt.close()

    # overall
    plot_distribution(df, "Overall Dependence Score Distribution", "overall")

    # 按 mode
    for mode, sub in df.groupby("mode"):
        plot_distribution(sub, f"Score Distribution - mode={mode}", f"mode_{mode}")

    # 按 task（保留原有逻辑）
    for task, sub in df.groupby("task"):
        safe_task = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(task))
        plot_distribution(sub, f"Score Distribution - task={task}", f"task_{safe_task}")

    # task × mode（保留原有逻辑）
    summary_rows = []
    for (task, mode), sub in df.groupby(["task", "mode"]):
        safe_task = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(task))
        title = f"Score Distribution - task={task}, mode={mode}"
        fname = f"task_{safe_task}__mode_{mode}"
        plot_distribution(sub, title, fname)

        summary_rows.append({
            "task": task,
            "mode": mode,
            "count": len(sub),
            "mean": sub["score"].mean(),
            "std": sub["score"].std(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "task_mode_summary.csv", index=False)
    print(f"[INFO] Analysis done. Figures & summary saved to: {outdir}")

# =========================================================
# CLI
# =========================================================
def parse_main_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["prepare", "judge", "analyze"],
        help="pipeline 阶段：prepare / judge / analyze",
    )

    # phase=prepare
    ap.add_argument("--answers", type=str, help="带 query_id / task / system_prompt 的 answers jsonl（修复后的文件）")
    ap.add_argument(
        "--base-judge-inputs",
        type=str,
        default=None,
        help="若提供，则在该文件中已有的 (task, query_id) 基础上为每个 task 额外采样 n-per-task 个新 query。",
    )
    ap.add_argument("--n-per-task", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    # phase=judge
    ap.add_argument("--judge-data", type=str, help="prepare 阶段生成的 judge_inputs.jsonl")
    ap.add_argument("--model-path", type=str, default=os.environ.get("MODEL_PATH", "Qwen3-8B"))
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument(
        "--retry-invalid",
        action="store_true",
        help=(
            "如输出文件已存在：仅保留其中 judge_parse_ok 且带 overall_memory_dependence_score 的记录，"
            "对其他 uid 按 max-retries 逻辑重新调用 judge。"
        ),
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help=(
            "对 invalid 样本允许的重试次数（不含第一次尝试）。"
            "总尝试次数上限 = 1 + max_retries。超过后仍 invalid 的样本会被标记 "
            "max_retry_exceeded=True，并可写入 --failed-out。"
        ),
    )
    ap.add_argument(
        "--failed-out",
        type=str,
        default=None,
        help="可选：将超过最大尝试次数仍 invalid 的样本额外写入该 jsonl 文件（per shard）。",
    )
    ap.add_argument("--out", type=str, help="judge 阶段输出的 jsonl（per shard）")

    # phase=analyze
    ap.add_argument("--judged", type=str, help="合并后的 judge 输出 jsonl")
    ap.add_argument("--analysis-outdir", type=str, help="统计与图表输出目录")
    ap.add_argument(
        "--blacklist-answers",
        type=str,
        default=None,
        help=(
            "可选：answers json/jsonl 文件，文件中出现的 uid 将在分析和可视化时作为黑名单剔除，"
            "例如 all_answers_with_query_id.not_trunc.jsonl。"
        ),
    )

    return ap.parse_args()


def main():
    args = parse_main_args()
    if args.phase == "prepare":
        assert args.answers and args.out
        phase_prepare(args)
    elif args.phase == "judge":
        assert args.judge_data and args.out
        phase_judge(args)
    elif args.phase == "analyze":
        assert args.judged and args.analysis_outdir
        phase_analyze(args)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
