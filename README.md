# Controllable Memory Usage: Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction

This repository contains the public, sanitized code and example resources for our paper on controllable memory usage in long-term human-agent interaction.

We study how an agent's reliance on memory can be treated as an explicit, user-controllable behavior dimension. The code in this release focuses on the data construction and evaluation pipeline around long-horizon interaction traces, artifact evolution, cross-session preference summaries, and memory-dependence evaluation.

## What Is Included

- Core pipeline code under `framework_data/`:
  - `event/`: long-horizon event timeline generation
  - `concept/`: concept extraction from event timelines
  - `artifacts/`: evolving artifact generation over time
  - `cross-session/`: task-specific cross-session summaries
  - `composed_memory/`: episodic and retrieved memory composition
  - `all_context_merge/`: query rules and persona templates
- Evaluation scripts under `evaluation/`:
  - job creation
  - memory-bank generation
  - query/rubric/answer generation
  - rubric resources
- Small sanitized examples under `examples/`
- Paper PDF under `docs/`

## What Is Not Included

This public package intentionally excludes the following:

- Raw private or downloaded datasets
- Large internal generation dumps and logs
- Notebooks used for internal analysis
- Hard-coded API keys, app IDs, and machine-specific absolute paths
- Third-party repositories copied into the original workspace

## Repository Structure

```text
framework_data/
  all_context_merge/
  artifacts/
  composed_memory/
  concept/
  cross-session/
  event/
evaluation/
  general_rubrics/
  make_test_data/
  preference_rubrics/
examples/
  event_output/
  mix/
docs/
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set API credentials through environment variables before running any generation script.

## Quick Start

Generate sample timelines:

```bash
bash framework_data/event/run_generate.sh
```

Generate concepts from example mixed cases:

```bash
bash framework_data/concept/run_generate.sh
```

Generate artifacts from example event outputs:

```bash
bash framework_data/artifacts/run_generate.sh
```

Generate cross-session summaries from example mixed cases:

```bash
bash framework_data/cross-session/run_generate.sh
```

Compose episodic and retrieved memories:

```bash
bash framework_data/composed_memory/run_compose.sh
```

Create evaluation jobs from example event outputs:

```bash
python evaluation/make_test_data/make_jobs.py \
  --root examples/event_output \
  --out evaluation/jobs.example.jsonl \
  --out_base evaluation/test_set
```

## Notes On The Examples

The `examples/` directory contains only a very small subset of generated outputs to demonstrate file formats and expected directory layout. It is not the full experiment release.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{tian2026controllable,
  title={Controllable Memory Usage: Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction},
  author={Tian, Muzhao and Huang, Zisu and Wang, Xiaohua and Xu, Jingwen and Guo, Zhengkang and Qian, Qi and Song, Kaitao and Yuan, Jiakang and Lv, Changze and Zheng, Xiaoqing},
  year={2026}
}
```