import json
import os
import argparse
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

from tqdm import tqdm
from vllm import LLM, SamplingParams


# =========================================================
# Args
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--instructions", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--shard_id", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)

    ap.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", "Qwen3-8B"))
    ap.add_argument("--batch_size", type=int, default=16)

    # sampling
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=8000)

    # think control
    ap.add_argument(
        "--force_think_tags",
        action="store_true",
        help="在 system 中显式要求：先在 <think>...</think> 中思考，再给最终答案。",
    )
    ap.add_argument(
        "--think_tag",
        type=str,
        default="think",
        help="思考块使用的 tag 名，如 <think>...</think>。",
    )

    # 新增：选择启用哪些 mode
    ap.add_argument(
        "--modes",
        type=str,
        default="none,low,medium,high",
        help="逗号分隔的 mode 列表，子集于 {none,low,medium,high}。"
             "例如: 'none,low' 只生成 non-instruct 和 low instruct。",
    )

    # 新增：为每个 mode 单独设置采样次数
    ap.add_argument(
        "--samples-per-mode",
        type=str,
        default="4",
        help=(
            "可以是单个整数（所有 mode 共用），例如 '4'；"
            "也可以是逗号分隔的 'mode:k' 列表，例如 'none:5,low:10'。"
            "未显式指定的 mode 使用默认值（若是单个整数，就用该整数；否则为 4）。"
        ),
    )

    return ap.parse_args()


# =========================================================
# IO utils
# =========================================================
def load_data(path: str) -> List[Dict]:
    """支持 JSON array 和 JSONL 两种格式。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        assert isinstance(data, list)
        return data
    # JSONL
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_existing_uids(path: Path) -> Set[str]:
    """用于断点续跑：读取已有输出中的 uid 集合。"""
    if not path.exists():
        return set()
    uids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                uids.add(json.loads(line)["uid"])
            except Exception:
                continue
    return uids


def build_uid(sample_id: str, mode: str, k: int) -> str:
    return f"{sample_id}::{mode}::{k}"


def parse_mode_from_uid(uid: str) -> Optional[str]:
    """
    从 uid 里解析出 mode。uid 形式为: <query_id>::<mode>::<k>
    """
    parts = uid.split("::")
    if len(parts) != 3:
        return None
    return parts[1]


# =========================================================
# Query ID 构造（query 粒度）
# =========================================================
def make_query_id(sample: Dict) -> str:
    """
    为每一条 query 构造一个稳定的 query_id，用于：
      - 唯一标识 query（跨 task / target / query 文本）
      - 作为 uid 的前缀

    形式：event_id__directory_index__task__target__<短 hash>
    """
    event_id = sample.get("event_id", "NA")
    dindex = str(sample.get("directory_index", "0"))
    task = sample.get("task", "NA")
    target = sample.get("target", "NA")
    query = (sample.get("query") or "").strip()

    base = json.dumps(
        {
            "event_id": event_id,
            "directory_index": dindex,
            "task": task,
            "target": target,
            "query": query,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    return f"{event_id}__{dindex}__{task}__{target}__{h}"


# =========================================================
# Think extraction
# =========================================================
def extract_think_and_answer(text: str, tag: str = "think") -> Tuple[Optional[str], str]:
    """
    从文本中提取 <think>...</think>（可多段），返回：
      - think_text: 多个块用空行连接；如无则为 None
      - answer_text: 删除所有 think 块后的剩余文本
    """
    if text is None:
        return None, ""

    pattern = re.compile(
        rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    thinks = [m.group(1).strip() for m in pattern.finditer(text)]
    answer = pattern.sub("", text).strip()

    think_text: Optional[str] = None
    if thinks:
        think_text = "\n\n".join([t for t in thinks if t])

    return think_text, answer


# =========================================================
# Prompt building  (chat + think)
# =========================================================
def build_chat_prompt(
    sample: Dict,
    mode: str,
    instruction_text: Optional[str],
    force_think_tags: bool,
    think_tag: str,
    tokenizer,
) -> Tuple[str, str, str]:
    """
    构造：
    - system_prompt: 基于 full_context + think 约束
    - user_prompt:   query + （可选）memory_dependence instruction
    - chat_prompt:   tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    返回: (system_prompt, user_prompt, chat_prompt)
    """
    full_context = (sample.get("full_context") or "").strip()
    base_query = (sample.get("query") or "").strip()

    # -------- system prompt --------
    sys_parts: List[str] = []

    sys_parts.append("You are a helpful, rigorous AI assistant.")

    if full_context:
        sys_parts.append(full_context)

    system_prompt = "\n\n".join(sys_parts).strip()

    # -------- user prompt --------
    if mode == "none" or not instruction_text:
        user_prompt = base_query
    else:
        user_prompt = instruction_text.strip() + "\n\n---\n\n" + base_query

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return system_prompt, user_prompt, chat_prompt


# =========================================================
# helpers for mode config
# =========================================================
ALL_SUPPORTED_MODES = ["none", "low", "medium", "high"]


def parse_enabled_modes(modes_str: str) -> List[str]:
    modes = [m.strip() for m in modes_str.split(",") if m.strip()]
    if not modes:
        raise ValueError("至少需要启用一个 mode（例如 --modes none,low）。")
    for m in modes:
        if m not in ALL_SUPPORTED_MODES:
            raise ValueError(f"不支持的 mode: {m}，必须是 {ALL_SUPPORTED_MODES} 之一。")
    # 去重并保持顺序
    seen = set()
    result = []
    for m in modes:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def parse_samples_per_mode(spec: str, enabled_modes: List[str], default_global: int = 4) -> Dict[str, int]:
    """
    spec:
      - 若是单个整数字符串，例如 "4"：所有 enabled_modes 都用该值。
      - 若是 'none:5,low:10' 形式：对指定 mode 用对应值，其余 enabled_modes 用 default_global。
    """
    spec = spec.strip()
    # 单个整数
    if spec.isdigit():
        v = int(spec)
        return {m: v for m in enabled_modes}

    # mode:k,mode:k...
    result: Dict[str, int] = {}
    # 先给个默认
    for m in enabled_modes:
        result[m] = default_global

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            raise ValueError(
                f"--samples-per-mode 解析失败：'{p}' 不是 'mode:k' 形式。"
            )
        m, v_str = p.split(":", 1)
        m = m.strip()
        v_str = v_str.strip()
        if m not in enabled_modes:
            # 可以选择宽松处理：忽略未启用的 mode
            # 也可以严格报错，这里选择忽略并给个提示
            print(f"[WARN] 在 --samples-per-mode 中指定了未启用的 mode '{m}'，已忽略。")
            continue
        if not v_str.isdigit():
            raise ValueError(f"--samples-per-mode 中 '{p}' 的采样次数不是整数。")
        result[m] = int(v_str)

    return result


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    # ---------- data ----------
    data = load_data(args.data)
    instr_cfg = json.load(open(args.instructions, "r", encoding="utf-8"))

    # 全部模式到 instruct 文本的映射
    all_modes_to_instr: Dict[str, Optional[str]] = {
        "none": None,
        "low": instr_cfg["instructions"]["low_strict"]["instruction_text"],
        "medium": instr_cfg["instructions"]["medium_balanced"]["instruction_text"],
        "high": instr_cfg["instructions"]["high_continuation"]["instruction_text"],
    }

    # 解析启用的 mode 列表
    enabled_modes: List[str] = parse_enabled_modes(args.modes)

    # 解析每个 mode 的采样次数
    samples_per_mode: Dict[str, int] = parse_samples_per_mode(
        args.samples_per_mode,
        enabled_modes=enabled_modes,
        default_global=4,
    )

    print("[INFO] Enabled modes and samples per mode:")
    for m in enabled_modes:
        print(f"  mode={m}, samples={samples_per_mode[m]}")

    # 将 instruct 文本裁剪到启用的 modes
    modes: Dict[str, Optional[str]] = {m: all_modes_to_instr[m] for m in enabled_modes}

    shard_data = [
        x for i, x in enumerate(data)
        if i % args.num_shards == args.shard_id
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_uids = load_existing_uids(out_path)

    # 只统计当前启用的 mode 中，已经存在的 uid 数量用于进度条
    already = 0
    for uid in done_uids:
        m = parse_mode_from_uid(uid)
        if m in enabled_modes:
            already += 1

    total_needed_per_query = sum(samples_per_mode[m] for m in enabled_modes)
    total_needed = len(shard_data) * total_needed_per_query

    pbar = tqdm(
        total=max(0, total_needed - already),
        desc=f"Shard {args.shard_id}",
        dynamic_ncols=True,
    )

    # ---------- vLLM ----------
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
        frequency_penalty=0.6,
    )

    fout = out_path.open("a", buffering=1, encoding="utf-8")

    buf_prompts: List[str] = []
    buf_meta: List[Tuple] = []

    def flush():
        nonlocal buf_prompts, buf_meta
        if not buf_prompts:
            return

        outputs = llm.generate(buf_prompts, sampling)
        for out, meta in zip(outputs, buf_meta):
            sample_obj, mode_i, k_i, uid_i, qid_i, sys_p, usr_p, full_p = meta
            raw_text = out.outputs[0].text

            think_text, answer_text = extract_think_and_answer(
                raw_text,
                tag=args.think_tag,
            )

            record = {
                "uid": uid_i,
                "query_id": qid_i,
                "event_id": sample_obj.get("event_id"),
                "directory_index": sample_obj.get("directory_index"),
                "domain": sample_obj.get("domain"),
                "subject": sample_obj.get("subject"),
                "topic": sample_obj.get("topic"),
                "task": sample_obj.get("task"),
                "target": sample_obj.get("target"),
                "mode": mode_i,
                "sample_id": k_i,
                "query": sample_obj.get("query"),
                "context": sample_obj.get("full_context"),
                "system_prompt": sys_p,
                "user_prompt": usr_p,
                "full_prompt": full_p,
                "think": think_text,
                "answer": answer_text,
                "raw": raw_text,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.update(1)

        buf_prompts.clear()
        buf_meta.clear()

    # ---------- main loop ----------
    for sample in shard_data:
        query_id = make_query_id(sample)

        for mode in enabled_modes:
            inst = modes[mode]
            samples_this_mode = samples_per_mode[mode]

            for k in range(samples_this_mode):
                uid = build_uid(query_id, mode, k)
                if uid in done_uids:
                    continue

                sys_p, usr_p, chat_p = build_chat_prompt(
                    sample=sample,
                    mode=mode,
                    instruction_text=inst,
                    force_think_tags=args.force_think_tags,
                    think_tag=args.think_tag,
                    tokenizer=tokenizer,
                )

                buf_prompts.append(chat_p)
                buf_meta.append((sample, mode, k, uid, query_id, sys_p, usr_p, chat_p))

                if len(buf_prompts) >= args.batch_size:
                    flush()

    flush()
    fout.close()
    pbar.close()


if __name__ == "__main__":
    main()