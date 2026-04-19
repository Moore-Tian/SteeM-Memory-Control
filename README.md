<p align="center">
  <img src="assets/steem_bot.jpg" alt="SteeM Bot" width="200">
</p>

<h1 align="center">SteeM: Controllable Memory Usage</h1>

<h4 align="center">Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction</h4>

<p align="center">
  <a href="https://arxiv.org/abs/2601.05107"><img src="https://img.shields.io/badge/arXiv-2601.05107-b31b1b.svg" alt="arXiv"></a>
</p>

<p align="center">
  <b>Muzhao Tian</b> · <b>Zisu Huang</b> · <b>Xiaohua Wang</b> · <b>Jingwen Xu</b> · <b>Zhengkang Guo</b> · <b>Qi Qian</b> · <b>Kaitao Song</b> · <b>Jiakang Yuan</b> · <b>Changze Lv</b> · <b>Xiaoqing Zheng</b>
</p>

---

This repository provides code and data for our paper on **controllable memory usage** in long-term human-agent interaction. Instead of treating memory usage as an all-or-nothing hidden policy, we model it as a user-controllable preference that an agent can learn to follow.

**Key contributions:**

- A **rubric-based formulation** for measuring how strongly a response depends on retrieved memory
- A **long-horizon synthetic data pipeline** built from timelines, evolving artifacts, and cross-session summaries
- **SteeM**, a training recipe combining preference-aligned data generation, SFT, and GRPO to improve memory-dependence controllability

<p align="center">
  <a href="assets/main.pdf">
    <img src="assets/main.png" alt="Main figure of the SteeM paper" width="920">
  </a>
</p>

## What Is Included

| Component | Path | Description |
|---|---|---|
| **Timeline generation** | `framework_data/event/` | Long-horizon event timeline generation |
| **Concept extraction** | `framework_data/concept/` | Concept extraction from event timelines |
| **Artifact evolution** | `framework_data/artifacts/` | Evolving artifact generation over time |
| **Cross-session summaries** | `framework_data/cross-session/` | Task-specific cross-session preference summaries |
| **Memory composition** | `framework_data/composed_memory/` | Compose intra-session, inter-session, and user-profile memory into unified snapshots |
| **Evaluation** | `evaluation/` | Job creation, memory-bank generation, query/rubric/answer generation, rubric resources |
| **Timeline data** | `data/timeline/` | 388 released timelines (194 research + 194 tutoring) |

## Repository Structure

```text
data/timeline/           # Released timeline data (research/ + tutoring/)
framework_data/
  event/                 # Timeline generation
  concept/               # Concept extraction
  artifacts/             # Artifact evolution
  cross-session/         # Cross-session summaries
  composed_memory/       # Intra-/inter-session + user-profile memory composition
evaluation/
  general_rubrics/       # Task-quality rubrics
  preference_rubrics/    # Memory-dependence rubrics
  make_test_data/        # Evaluation data generation scripts
assets/                  # Paper figure & logo
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then fill in your OPENAI_API_KEY
```

## Quick Start

```bash
# Generate sample timelines
bash framework_data/event/run_generate.sh

# Extract concepts from timelines
bash framework_data/concept/run_generate.sh

# Generate evolving artifacts
bash framework_data/artifacts/run_generate.sh

# Generate cross-session summaries
bash framework_data/cross-session/run_generate.sh

# Compose episodic and retrieved memories
bash framework_data/composed_memory/run_compose.sh
```

## Released Timeline Data

The `data/timeline/` directory contains **388 synthesized project trajectories** (194 research + 194 tutoring, each 16–25 steps). Each case includes:

- **`events.json`** — structured timeline with `event_id`, `time_index`, `event_type`, `description`, `required_artifacts`, `generated_artifacts`, and `reason`
- **`stats.json`** — generation-time statistics (token counts, number of steps)

Prompt-level generation traces (`interactions.json`) are excluded for privacy. See `data/timeline/README.md` for details.

## Evaluation

The evaluation scripts are included for transparency. To build downstream jobs from the released timelines:

```bash
python evaluation/make_test_data/make_jobs.py \
  --root data/timeline \
  --out /tmp/timeline_jobs.jsonl \
  --out_base evaluation/test_set
```

## Citation

```bibtex
@article{tian2026controllable,
  title={Controllable Memory Usage: Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction},
  author={Tian, Muzhao and Huang, Zisu and Wang, Xiaohua and Xu, Jingwen and Guo, Zhengkang and Qian, Qi and Song, Kaitao and Yuan, Jiakang and Lv, Changze and Zheng, Xiaoqing},
  year={2026}
}
```
