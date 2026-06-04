#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
judge_memory_dependence.py

Judge memory dependence scores on model outputs with vLLM.
Loads output JSONL, constructs judge prompts, and adds judge results to the original data.

Enhancements:
- Robustly import RUBRICS_TEXT (package import or file fallback)
- Skip samples with API errors / empty answers / missing required fields (no crash)
- Write skipped samples to output with skip_reason
- Keep batch-save behavior
"""

from __future__ import annotations

import os
import json
import re
import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Set environment variables before importing torch/vllm
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_TORCH_COMPILE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


# -----------------------------
# Import rubrics text robustly
# -----------------------------
THIS_FILE = Path(__file__).resolve()
METHOD_ROOT = THIS_FILE.parents[1]
RUBRICS_PATH = METHOD_ROOT / "rubrics" / "dependence_rubrics_text.py"

RUBRICS_TEXT: str

if not RUBRICS_PATH.exists():
    raise ImportError(f"Could not find rubrics file at {RUBRICS_PATH}")

spec = importlib.util.spec_from_file_location("_dependence_rubrics_text", RUBRICS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load rubrics module from {RUBRICS_PATH}")

rubrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rubrics_module)
RUBRICS_TEXT = getattr(rubrics_module, "RUBRICS_TEXT", "")
if not isinstance(RUBRICS_TEXT, str) or not RUBRICS_TEXT.strip():
    raise ImportError(f"Could not extract RUBRICS_TEXT from {RUBRICS_PATH}")


# -----------------------------
# Utilities
# -----------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def robust_json_from_text(text: str) -> Optional[dict]:
    """
    Try to extract a JSON object from model output.
    - strip code fences
    - take substring from first '{' to last '}'
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


def _nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _make_skipped_result(sample: Dict[str, Any], reason: str) -> Dict[str, Any]:
    r = dict(sample)
    r["judge_raw"] = ""
    r["attempt_index"] = 0
    r["judge_parse_ok"] = False
    r["dimension_scores"] = None
    r["overall_memory_dependence_score"] = None
    r["rationale"] = None
    r["max_retry_exceeded"] = True
    r["skip_reason"] = reason
    return r


# -----------------------------
# Judge class
# -----------------------------
class VLLMJudge:
    """Wrapper for vLLM-based memory dependence judging."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        print(f"[INFO] Loading vLLM model from {model_path} ...", flush=True)
        self.model_path = model_path
        self.max_new_tokens = int(max_new_tokens)

        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print(f"[INFO] Model loaded successfully.", flush=True)

    def build_judge_prompt(
        self,
        item: dict,
        task_field: str = "task",
        context_field: str = "full_context",
        query_field: str = "query",
        answer_field: str = "generated_text",
    ) -> str:
        system_prompt = (
            "You are an expert evaluator of how strongly a response depends on "
            "the given memory / project history / user profile.\n\n"
            "You are given a rubric written in natural language describing how "
            "to score the memory dependence of an answer relative to its "
            "context/history.\n\n"
            "RUBRIC (natural language description):\n"
            f"{RUBRICS_TEXT}\n\n"
            "You MUST output a single JSON object that strictly follows the "
            "schema described in the rubric as 'global_instructions.output_schema'. "
            "Do NOT output any text before or after the JSON. Do NOT use code fences."
        )

        task = item.get(task_field, "")
        context = item.get(context_field, "")
        query = item.get(query_field, "")
        answer = item.get(answer_field, "")

        if os.environ.get("DEBUG_PROMPT", "0") == "1":
            uid = item.get("uid")
            api_err = item.get("api_error")
            ans_len = len(answer) if isinstance(answer, str) else 0
            print(f"[DEBUG] uid={uid} api_error={api_err} answer_len={ans_len} keys={list(item.keys())}", flush=True)

        # NOTE: process_samples() already filters invalid items; this is just a safety net.
        if not (_nonempty_str(task) and _nonempty_str(context) and _nonempty_str(query) and _nonempty_str(answer)):
            raise ValueError(
                f"Missing/empty required fields: need {task_field}, {context_field}, {query_field}, {answer_field}"
            )

        user_prompt = (
            "Please evaluate how strongly the following ANSWER depends on the "
            "provided MEMORY / CONTEXT, according to the rubric.\n\n"
            f"TASK TYPE:\n{task}\n\n"
            "MEMORY / CONTEXT (includes user profile, cross-session summaries, "
            "recent events, and any relevant artifacts if present):\n"
            f"{context}\n\n"
            "USER QUERY:\n"
            f"{query}\n\n"
            "ANSWER TO EVALUATE (may contain internal thinking segments like "
            "<think>...</think>):\n"
            f"{answer}\n\n"
            "Now follow the rubric and produce your evaluation as a JSON object."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if max_tokens is None:
            max_tokens = self.max_new_tokens

        sampling_params = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            n=1,
        )

        outputs = self.llm.generate(prompts, sampling_params=sampling_params)

        results: List[Dict[str, Any]] = []
        for out in outputs:
            if not out.outputs:
                gen_text = ""
            else:
                gen = out.outputs[0]
                gen_text = (gen.text or "").strip()
            results.append({"text": gen_text})
        return results

    def process_samples(
        self,
        samples: List[Dict[str, Any]],
        task_field: str = "task",
        context_field: str = "full_context",
        query_field: str = "query",
        answer_field: str = "generated_text",
        batch_size: int = 8,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_retries: int = 3,
        verbose: bool = True,
        output_file: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        - Skips invalid samples (api_error / empty generated_text / missing required fields)
        - Judges valid samples with retry-on-parse-failure
        - Writes results incrementally (skipped written immediately; judged written per batch)
        """
        n = len(samples)
        max_attempts = 1 + max(0, max_retries)

        sample_attempt_counts = [0] * n
        sample_results: List[Optional[Dict[str, Any]]] = [None] * n
        written_indices: Set[int] = set()

        def _write_one(idx: int, result: Dict[str, Any]) -> None:
            if output_file is None:
                return
            if idx in written_indices:
                return
            with output_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            written_indices.add(idx)

        # Build prompts only for valid samples; write skipped immediately.
        prompts: List[str] = []
        valid_orig_indices: List[int] = []

        skipped_count = 0
        for orig_idx, sample in enumerate(samples):
            api_err = sample.get("api_error")
            task = sample.get(task_field, "")
            context = sample.get(context_field, "")
            query = sample.get(query_field, "")
            answer = sample.get(answer_field, "")

            if api_err:
                r = _make_skipped_result(sample, f"skip: generation_failed api_error={api_err}")
                sample_results[orig_idx] = r
                _write_one(orig_idx, r)
                skipped_count += 1
                continue

            if not (_nonempty_str(task) and _nonempty_str(context) and _nonempty_str(query) and _nonempty_str(answer)):
                r = _make_skipped_result(
                    sample,
                    f"skip: missing_or_empty_fields required=({task_field},{context_field},{query_field},{answer_field})",
                )
                sample_results[orig_idx] = r
                _write_one(orig_idx, r)
                skipped_count += 1
                continue

            try:
                prompt = self.build_judge_prompt(
                    sample,
                    task_field=task_field,
                    context_field=context_field,
                    query_field=query_field,
                    answer_field=answer_field,
                )
            except Exception as e:
                r = _make_skipped_result(sample, f"skip: prompt_build_error {repr(e)}")
                sample_results[orig_idx] = r
                _write_one(orig_idx, r)
                skipped_count += 1
                continue

            prompts.append(prompt)
            valid_orig_indices.append(orig_idx)

        if verbose:
            print(f"[INFO] Total samples: {n}", flush=True)
            print(f"[INFO] Valid for judging: {len(prompts)}", flush=True)
            print(f"[INFO] Skipped pre-judge: {skipped_count}", flush=True)

        # If nothing to judge, return early
        if not prompts:
            final_results: List[Dict[str, Any]] = []
            for i in range(n):
                if sample_results[i] is None:
                    # Should not happen; but guard anyway
                    r = _make_skipped_result(samples[i], "skip: unknown_no_prompt")
                    sample_results[i] = r
                    _write_one(i, r)
                final_results.append(sample_results[i])  # type: ignore
            return final_results

        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(prompts))

            batch_prompts = prompts[start:end]
            batch_orig_indices = valid_orig_indices[start:end]
            batch_samples = [samples[i] for i in batch_orig_indices]

            if verbose:
                print(
                    f"\n[BATCH {batch_idx + 1}/{total_batches}] Processing {len(batch_samples)} samples...",
                    flush=True,
                )

            batch_failed_local_indices = list(range(len(batch_samples)))
            retry_count = 0

            while retry_count < max_attempts and batch_failed_local_indices:
                if retry_count == 0:
                    if verbose:
                        print(f"  Initial run: {len(batch_failed_local_indices)} items", flush=True)
                    retry_temperature = temperature
                    retry_top_p = top_p
                    retry_max_tokens = None
                else:
                    if verbose:
                        print(f"  Retry {retry_count}/{max_attempts-1}: {len(batch_failed_local_indices)} items", flush=True)
                    retry_temperature = 0.6
                    retry_top_p = 0.8
                    retry_max_tokens = 10000

                retry_prompts = [batch_prompts[i] for i in batch_failed_local_indices]
                retry_orig_indices = [batch_orig_indices[i] for i in batch_failed_local_indices]

                outputs = self.generate_batch(
                    retry_prompts,
                    temperature=retry_temperature,
                    top_p=retry_top_p,
                    max_tokens=retry_max_tokens,
                )

                new_failed_local_indices: List[int] = []

                for local_i, (orig_i, output) in enumerate(zip(retry_orig_indices, outputs)):
                    raw_text = output["text"]
                    judge_json = robust_json_from_text(raw_text)

                    sample_attempt_counts[orig_i] += 1

                    result = dict(samples[orig_i])
                    result["judge_raw"] = raw_text
                    result["attempt_index"] = sample_attempt_counts[orig_i]

                    if judge_json is None:
                        if sample_attempt_counts[orig_i] < max_attempts:
                            # retry
                            # map back to local index in batch_failed_local_indices
                            new_failed_local_indices.append(batch_failed_local_indices[local_i])
                            if verbose:
                                print(
                                    f"    Sample {orig_i}: parse error -> retry "
                                    f"({sample_attempt_counts[orig_i]}/{max_attempts})",
                                    flush=True,
                                )
                        else:
                            result["judge_parse_ok"] = False
                            result["dimension_scores"] = None
                            result["overall_memory_dependence_score"] = None
                            result["rationale"] = None
                            result["max_retry_exceeded"] = True
                            result["skip_reason"] = "judge_failed_parse"
                            sample_results[orig_i] = result
                            _write_one(orig_i, result)
                    else:
                        result["judge_parse_ok"] = True
                        result["dimension_scores"] = judge_json.get("dimension_scores")
                        result["overall_memory_dependence_score"] = judge_json.get("overall_memory_dependence_score")
                        result["rationale"] = judge_json.get("rationale")
                        result["max_retry_exceeded"] = False
                        sample_results[orig_i] = result
                        _write_one(orig_i, result)

                batch_failed_local_indices = new_failed_local_indices
                retry_count += 1

            # Batch summary
            if verbose:
                batch_ok = 0
                for orig_i in batch_orig_indices:
                    r = sample_results[orig_i]
                    if r is not None and r.get("judge_parse_ok") is True:
                        batch_ok += 1
                print(f"  Batch complete: {batch_ok}/{len(batch_orig_indices)} successful", flush=True)

        # Ensure every sample has a result (should be true)
        final_results: List[Dict[str, Any]] = []
        missing = 0
        for i in range(n):
            if sample_results[i] is None:
                missing += 1
                r = _make_skipped_result(samples[i], "skip: missing_final_result")
                sample_results[i] = r
                _write_one(i, r)
            final_results.append(sample_results[i])  # type: ignore

        if verbose and missing > 0:
            print(f"[WARN] Filled {missing} missing results with skip placeholders.", flush=True)

        return final_results


def main():
    parser = argparse.ArgumentParser(description="Judge memory dependence scores using vLLM on model outputs")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL file with model outputs")
    parser.add_argument("--model_path", type=str, required=True, help="Path to vLLM model for judging")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL file with judge results added")
    parser.add_argument("--task_field", type=str, default="task", help="Field name for task type (default: 'task')")
    parser.add_argument("--context_field", type=str, default="full_context", help="Field name for context/memory (default: 'full_context')")
    parser.add_argument("--query_field", type=str, default="query", help="Field name for user query (default: 'query')")
    parser.add_argument("--answer_field", type=str, default="generated_text", help="Field name for answer (default: 'generated_text')")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens for generation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for generation")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries for failed parsing (default: 3)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process (for testing)")

    args = parser.parse_args()

    print(f"[INFO] Loading input data from {args.input_jsonl}...", flush=True)
    samples = load_jsonl(Path(args.input_jsonl))
    if args.limit:
        samples = samples[: args.limit]
    print(f"[INFO] Loaded {len(samples)} samples", flush=True)

    model = VLLMJudge(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear output file to start fresh
    if output_path.exists():
        output_path.unlink()

    print(f"[INFO] Starting batch processing...", flush=True)
    results = model.process_samples(
        samples=samples,
        task_field=args.task_field,
        context_field=args.context_field,
        query_field=args.query_field,
        answer_field=args.answer_field,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retries=args.max_retries,
        verbose=True,
        output_file=output_path,
    )

    success_count = sum(1 for r in results if r.get("judge_parse_ok", False))
    skipped_count = sum(1 for r in results if r.get("skip_reason"))
    print(f"\n[INFO] Processing complete!")
    print(f"  Total samples: {len(results)}")
    print(f"  Successful judges: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Others (failed parse etc.): {len(results) - success_count - skipped_count}")
    print(f"  Results saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
