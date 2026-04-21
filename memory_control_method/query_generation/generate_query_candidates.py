import json
import os
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

# Use environment variables to avoid embedding local endpoints or paths.
client_kwargs = {"api_key": os.environ.get("OPENAI_API_KEY", "EMPTY")}
base_url = os.environ.get("OPENAI_BASE_URL")
if base_url:
    client_kwargs["base_url"] = base_url
client = OpenAI(**client_kwargs)

model = os.environ.get("OPENAI_MODEL", "qwen3-32b")
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_PATH = Path(
    os.environ.get(
        "PERSONA_TOPIC_PREFS_PATH",
        "path/to/persona_topic_prefs.json",
    )
)
OUTPUT_PATH = Path(
    os.environ.get(
        "QUERY_CANDIDATES_OUTPUT_PATH",
        str(SCRIPT_DIR / "query_candidates.json"),
    )
)

# 对哪些 persona+topic 生成 query：至少多少条 preference
MIN_PREFS_PER_TOPIC = 10

# 每个 persona-topic 生成 3 条 query
NUM_QUERIES = 3


def build_prompt(persona_id: str, topic: str, prefs: List[str]) -> str:
    """
    构造给大模型的 user prompt。
    说明任务 + 提供 persona_id、topic、preferences。
    """
    # 把所有偏好格式化成 bullet list
    prefs_lines = "\n".join(f"- {p}" for p in prefs)

    prompt = f"""
You are a large language model that generates user chat queries based on a given persona's memory.

You are given:
- a persona id
- a topic
- a list of preference-like memory sentences under that topic

Your task:

1. Based ONLY on this topic and these memories, imagine what this persona might ask a chatbot.
2. Generate EXACTLY {NUM_QUERIES} distinct queries.
3. The {NUM_QUERIES} queries must cover these three relevance levels to the given memories relatively:
   - one query with "strong" relevance (directly and specifically grounded in the memories)
   - one query with "medium" relevance (clearly related, but not just restating a single memory)
   - one query with "weak" relevance (more general or tangential, but STILL clearly within the same topic "{topic}" and at least indirectly connected to the memories)
4. Even the "weak" query MUST still be within the same topic and cannot be completely unrelated.

For each query, you must also explain briefly WHY you think this query has that relevance level to the memories.

Output format:

Return ONLY a JSON array (no extra text, no explanations outside JSON). The array must contain exactly {NUM_QUERIES} objects. Each object must have the following fields:

- "persona_id": string, the persona id you received
- "topic": string, the topic you received
- "relevance": string, one of "strong", "medium", "weak"
- "query": string, the generated query text
- "reason": string, a brief explanation of why this query has that relevance level to the given memories

Example of the overall shape (this is just an example, do NOT reuse it literally):

[
  {{
    "persona_id": "193",
    "topic": "Sports",
    "relevance": "strong",
    "query": "...",
    "reason": "..."
  }},
  {{
    "persona_id": "193",
    "topic": "Sports",
    "relevance": "medium",
    "query": "...",
    "reason": "..."
  }},
  {{
    "persona_id": "193",
    "topic": "Sports",
    "relevance": "weak",
    "query": "...",
    "reason": "..."
  }}
]

Followings are the persona id, topic, and memories under this topic:

Persona id: {persona_id}
Topic: {topic}
Memories under this topic:
{prefs_lines}

Again, complete the given task and output ONLY valid JSON. Do not wrap it in markdown or add any other commentary.
"""
    return prompt.strip()



import json
import re

def strip_think_blocks(text: str) -> str:
    """
    删除模型输出中的 <think>...</think> 段落（可能不止一段）。
    """
    while True:
        start = text.find("<think>")
        if start == -1:
            break
        end = text.find("</think>", start)
        if end == -1:
            # 有开没关，就把后面全砍掉
            text = text[:start]
            break
        # 删掉整个 <think>...</think>
        text = text[:start] + text[end + len("</think>"):]
    return text


def call_model_for_persona_topic(persona_id: str, topic: str, prefs: List[str]) -> List[Dict[str, Any]]:
    """
    调用大模型，为单个 persona+topic 生成 queries。
    返回：解析后的 JSON 列表（长度=3）。
    """
    user_prompt = build_prompt(persona_id, topic, prefs)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that strictly follows the user's JSON output instructions."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=4096,
    )

    content = resp.choices[0].message.content

    content = strip_think_blocks(content)

    # 尝试直接解析 JSON；如果模型带了前后空白或换行，一般没问题
    # 如果有 code fence，可以简单清理一下
    text = content.strip()
    if text.startswith("```"):
        # 去掉 ```json ... ``` 包裹
        text = text.strip("`")
        # 保险起见卸掉可能的语言标记
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print("JSON decode error for persona", persona_id, "topic", topic)
        print("Raw content:\n", content)
        raise e

    if not isinstance(data, list):
        raise ValueError(f"Model output is not a JSON list for persona {persona_id}, topic {topic}")

    return data


from tqdm import tqdm


def main():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        persona_topic_prefs = json.load(f)

    all_results: List[Dict[str, Any]] = []

    total_pairs = 0
    used_pairs = 0

    for persona_id, topic2prefs in tqdm(persona_topic_prefs.items(), desc="Processing personas"):
        if not isinstance(topic2prefs, dict):
            continue

        for topic, prefs in topic2prefs.items():
            total_pairs += 1

            # 确保 prefs 是 list
            if not isinstance(prefs, list):
                continue

            if len(prefs) < MIN_PREFS_PER_TOPIC:
                continue  # 偏好数量不够就跳过

            print(f"[INFO] Generating queries for persona {persona_id}, topic '{topic}', prefs={len(prefs)}")

            try:
                result = call_model_for_persona_topic(persona_id, topic, prefs)
            except Exception as e:
                print(f"[ERROR] Failed for persona {persona_id}, topic '{topic}': {e}")
                continue

            # 可以顺便检查一下每条的 persona_id/topic 是否正确，不正确就强行覆盖
            for item in result:
                if not isinstance(item, dict):
                    continue
                item["persona_id"] = persona_id
                item["topic"] = topic
            all_results.extend(result)
            used_pairs += 1


    print(f"\nTotal persona-topic pairs in file: {total_pairs}")
    print(f"Pairs used for generation (prefs >= {MIN_PREFS_PER_TOPIC}): {used_pairs}")
    print(f"Total generated query items: {len(all_results)}")

    # 保存所有结果到一个 JSON 文件
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved all query candidates to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()