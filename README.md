<p align="center">
	<img src="paper/SteeM%20logo.png" alt="SteeM logo" width="180">
</p>

<h1 align="center">SteeM</h1>

<p align="center"><strong>Controllable Memory Usage: Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction</strong></p>

<p align="center">
	<a href="https://arxiv.org/abs/2601.05107"><img src="https://img.shields.io/badge/arXiv-2601.05107-b31b1b.svg" alt="arXiv"></a>
	<a href="paper/steem_memory_control_paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-4f6db8.svg" alt="Paper PDF"></a>
</p>

This repository is the paper companion release for SteeM. It packages the paper snapshot, the released data-construction pipeline, and a compact code extraction of the controllable memory method used in the paper.

SteeM studies a central tension in long-term human-agent interaction: heavy use of retrieved memory can improve consistency and personalization, but it can also trap the agent in past decisions and styles. The repository accompanies our paper by releasing the data pipeline used to build long-term interaction contexts and the method code used to measure and control memory dependence.

<p align="center">
	<img src="paper/main.png" alt="Main figure for SteeM" width="100%">
</p>

<p align="center"><em>Figure 1: SteeM introduces rubric-based memory-dependence scoring, preference-aligned data generation, training, and controllable generation modes ranging from fresh-start to history-adherent behavior.</em></p>

## Paper Links

- arXiv: [2601.05107](https://arxiv.org/abs/2601.05107)
- PDF in this repository: [paper/steem_memory_control_paper.pdf](paper/steem_memory_control_paper.pdf)

## Repository Overview

- [data_pipeline](data_pipeline): the released pipeline and selected outputs used to construct timelines, artifacts, concepts, cross-session summaries, and merged contexts
- [memory_control_method](memory_control_method): a compact method package containing query generation, controlled answer generation, dependence judging, evaluation, SFT rewrite utilities, and released rubrics

## Released Assets

- [data_pipeline/timeline_generation/output_gemini](data_pipeline/timeline_generation/output_gemini): 194 research cases and 194 tutoring cases, each containing `events.json` and `stats.json`
- [data_pipeline/context_merge/all_contexts.json.gz](data_pipeline/context_merge/all_contexts.json.gz): gzip-compressed release of the 11,541 merged query-context examples
- [data_pipeline/context_merge/sampled_contexts.json](data_pipeline/context_merge/sampled_contexts.json): 400 sampled context examples
- [memory_control_method](memory_control_method): compact public-facing code structure for the controllable memory method

## Quick Start

Run commands from the repository root.

### Data Pipeline

```bash
bash data_pipeline/timeline_generation/run_generate.sh
bash data_pipeline/artifact_generation/run_generate.sh
bash data_pipeline/concept_generation/run_generate.sh
bash data_pipeline/cross_session_generation/run_generate.sh
```

These entry scripts expect credentials and optional service URLs to be provided through the environment rather than embedded in the repository.

### Method Evaluation

```bash
MODEL_PATH=/path/to/model \
TEST_DATA_PATH=/path/to/test_data.jsonl \
RUN_JUDGE=1 \
RUN_METRIC=1 \
bash memory_control_method/evaluation/run_evaluation_pipeline.sh
```

For the data pipeline, set the environment variables you need before running generation scripts:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `OPENAI_UID` (optional)
- `TENCENT_CHAT_COMPLETIONS_URL` only if you use `request_type=tencent`

See [memory_control_method/README.md](memory_control_method/README.md) for the method-side directory layout and evaluation flow.

To inspect the full merged-context release locally, decompress it with:

```bash
gzip -dk data_pipeline/context_merge/all_contexts.json.gz
```

## Citation

If you find this repository useful, please cite the paper:

```bibtex
@article{tian2026controllable,
	title={Controllable Memory Usage: Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction},
	author={Tian, Muzhao and Huang, Zisu and Wang, Xiaohua and Xu, Jingwen and Guo, Zhengkang and Qian, Qi and Shen, Yuanzhe and Song, Kaitao and Yuan, Jiakang and Lv, Changze and Zheng, Xiaoqing},
	journal={arXiv preprint arXiv:2601.05107},
	year={2026}
}
```
