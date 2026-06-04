#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_model_inference.py

Run batch inference on a JSONL dataset with vLLM.
Loads JSONL with system and user query fields, constructs messages, and generates responses.
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Set environment variables before importing torch/vllm
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_TORCH_COMPILE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


class VLLMInference:
    """Wrapper for vLLM-based batch inference."""
    
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 2048,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        print(f"[INFO] Loading vLLM model from {model_path} ...", flush=True)
        
        self.model_path = model_path
        self.max_new_tokens = int(max_new_tokens)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Initialize vLLM
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            max_model_len=16384,
        )
        print(f"[INFO] Model loaded successfully.", flush=True)
    
    def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of messages."""
        # Convert messages to chat format
        chat_prompts = []
        for messages in messages_list:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            chat_prompts.append(text)
        
        sampling_params = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(self.max_new_tokens),
        )
        
        outputs = self.llm.generate(chat_prompts, sampling_params=sampling_params)
        
        results: List[Dict[str, Any]] = []
        for out in outputs:
            prompt_len = len(out.prompt_token_ids or [])
            if not out.outputs:
                gen_token_ids = []
                gen_text = ""
            else:
                gen = out.outputs[0]
                gen_token_ids = gen.token_ids or []
                gen_text = (gen.text or "").strip()
            
            results.append({
                "text": gen_text,
                "input_tokens": int(prompt_len),
                "output_tokens": int(len(gen_token_ids)),
            })
        
        return results
    
    def process_samples(
        self,
        samples: List[Dict[str, Any]],
        system_field: str = "system",
        user_query_field: str = "user_query",
        batch_size: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        verbose: bool = True,
        output_file: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process samples in batches.
        If output_file is provided, results are saved after each batch.
        Returns list of results with generation information.
        """
        all_results = []
        
        # Build messages for all samples
        messages_list = []
        for sample in samples:
            system_content = sample.get(system_field, "")
            user_content = sample.get(user_query_field, "")
            if not system_content or not user_content:
                raise ValueError(f"Missing required fields in sample: {sample.keys()}")
            
            messages = []
            if system_content:
                messages.append({"role": "system", "content": system_content})
            if user_content:
                messages.append({"role": "user", "content": user_content})
            
            if not messages:
                # If no messages found, create empty user message
                messages = [{"role": "user", "content": ""}]
            
            messages_list.append(messages)
        
        # Process in batches
        total_batches = (len(samples) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]
            batch_messages = messages_list[start_idx:end_idx]
            
            if verbose:
                print(f"\n[BATCH {batch_idx + 1}/{total_batches}] Processing {len(batch_samples)} samples...", flush=True)
            
            # Generate responses
            outputs = self.generate_batch(batch_messages, temperature=temperature, top_p=top_p)
            
            # Combine original sample data with generation results
            batch_final_results = []
            for i, (sample, output) in enumerate(zip(batch_samples, outputs)):
                result = {
                    **sample,  # Include all original fields
                    "generated_text": output["text"],
                    "input_tokens": output["input_tokens"],
                    "output_tokens": output["output_tokens"],
                }
                all_results.append(result)
                batch_final_results.append(result)
            
            # Save batch results if output file is provided
            if output_file is not None:
                with output_file.open("a", encoding="utf-8") as f:
                    for result in batch_final_results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            if verbose:
                print(f"  Batch complete: {len(batch_samples)} samples processed", flush=True)
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Batch inference using vLLM on JSONL dataset")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Input JSONL file with samples")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to vLLM model")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Output JSONL file for inference results")
    parser.add_argument("--system_field", type=str, default="system",
                        help="Field name for system message (default: 'system')")
    parser.add_argument("--user_query_field", type=str, default="user_query",
                        help="Field name for user query (default: 'user_query')")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum new tokens for generation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p for generation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Load input data
    print(f"[INFO] Loading input data from {args.input_jsonl}...", flush=True)
    samples = load_jsonl(Path(args.input_jsonl))
    if args.limit:
        samples = samples[:args.limit]
    print(f"[INFO] Loaded {len(samples)} samples", flush=True)
    
    # Initialize model
    model = VLLMInference(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # Prepare output file
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear output file if it exists (to start fresh)
    if output_path.exists():
        output_path.unlink()
    
    # Process samples (results are saved after each batch)
    print(f"[INFO] Starting batch processing...", flush=True)
    results = model.process_samples(
        samples=samples,
        system_field=args.system_field,
        user_query_field=args.user_query_field,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=True,
        output_file=output_path,
    )
    
    # Print statistics
    total_tokens = sum(r.get("output_tokens", 0) for r in results)
    print(f"\n[INFO] Processing complete!")
    print(f"  Total samples: {len(results)}")
    print(f"  Total output tokens: {total_tokens}")
    print(f"  Results saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()

