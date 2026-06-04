# artifact_prompts.py

import json
from typing import Any, Dict, List, Optional

# 目前 research domain 下支持的 artifact label
RESEARCH_ARTIFACT_LABELS = [
    "research_plan",
    "research_goals",
    "experiment_results",
    "method_scheme",
    "paper_paragraph",
]

TUTORING_ARTIFACT_LABELS = [
    "learning_objectives",
    "study_plan",
    "teaching_notes",
    "practice_record",
    "feedback_summary",
]


def _multiple_artifacts_schema_block_research(artifact_labels: List[str]) -> str:
    """
    Schema for multiple-artifacts generation.
    """
    labels_str = ", ".join([f'"{label}"' for label in artifact_labels])
    return f"""
Output schema (ONE JSON object):

The output is a single dictionary where:
- Each key is a artifact label you need to generate.
- Each value (the content) is a dictionary with the following structure:
  {{
    "artifact_text": "string",
    "reason": "string"
  }}

Example output structure:
{{
  "artifact_label_1": {{
    "artifact_text": "...",
    "reason": "..."
  }},
  "artifact_label_2": {{
    "artifact_text": "...",
    "reason": "..."
  }},
  ...
}}

Rules:
- Output MUST be valid JSON, with exactly the structure above.
- For each artifact label in the list [{labels_str}], you MUST provide an entry as a key in the output dictionary.
- Each artifact's "artifact_text" MUST be a non-empty string containing the VERSION of that artifact, after this event.
- Each artifact's "reason" MUST be a non-empty string containing the concise and clear reason for that artifact text.
- Do NOT output any additional keys or commentary.
- The generated artifact_text should be in natural language text, not JSON.
"""


def _format_prior_events_brief(
    prior_events: List[Dict[str, Any]],
    last_k: int = 5,
) -> str:
    """
    Keep context compact: include only last_k events, each with minimal fields.
    """
    if not prior_events:
        return "[]"
    slim: List[Dict[str, Any]] = []
    for e in prior_events[-last_k:]:
        slim.append(
            {
                "time_index": e.get("time_index"),
                "event_type": e.get("event_type"),
                "description": e.get("description"),
            }
        )
    return json.dumps(slim, ensure_ascii=False, indent=2)


def _format_prior_events_natural(
    prior_events: List[Dict[str, Any]],
    last_k: int = 5,
) -> str:
    """
    Format prior events as natural language text.
    """
    if not prior_events:
        return "No prior events."
    
    events_text = []
    for e in prior_events[-last_k:]:
        time_idx = e.get("time_index", "?")
        event_type = e.get("event_type", "unknown")
        description = e.get("description", "")
        events_text.append(f"Event {time_idx} ({event_type}): {description}")
    
    return "\n".join(events_text)


def _format_required_artifacts_natural(
    required_artifacts_recent_versions: Dict[str, List[Dict[str, Any]]],
) -> str:
    """
    Format required artifacts recent versions as natural language text.
    Only keeps the most recent version for each artifact.
    """
    if not required_artifacts_recent_versions:
        return "No required artifacts."
    
    artifacts_text = []
    for label, versions in required_artifacts_recent_versions.items():
        if not versions:
            continue
        # 只取最新的一个版本（列表中的最后一个）
        latest_version = versions[-1]
        text = latest_version.get("text", "")
        # Truncate long text for readability
        # text_preview = text[:200] + "..." if len(text) > 200 else text
        text_preview = text
        artifacts_text.append(f"\n===== Required artifact [{label}] begin =====")
        artifacts_text.append(f"{text_preview}")
        artifacts_text.append(f"===== Required artifact [{label}] end=====")
    
    return "\n".join(artifacts_text) if artifacts_text else "No required artifacts available."


def _artifact_label_guidance(label: str) -> str:
    """
    Give label-specific writing guidance to the model.
    """
    if label == "research_plan":
        return (
            "research_plan: a concise roadmap or next-steps plan for the project. "
            "It should focus on upcoming actions, priorities, rough timeline, and dependencies. "
            "Bullet lists are allowed. It should be actionable and realistic."
        )
    if label == "research_goals":
        return (
            "research_goals: the current research questions, hypotheses, and success criteria. "
            "Clarify what the project aims to demonstrate or measure, and how success will be evaluated."
        )
    if label == "method_scheme":
        return (
            "method_scheme: a concrete, executable description of the experimental or methodological setup. "
            "It should include key components, design choices, control baselines, evaluation metrics, "
            "and any important implementation details relevant for critique."
        )
    if label == "experiment_results":
        return (
            "experiment_results: a structured summary of the key findings from experiments. "
            "Highlight patterns in the results, successes and failures, unexpected behaviors, and known limitations. "
            "It should be consistent with the methods and not overclaim."
        )
    if label == "paper_paragraph":
        return (
            "paper_paragraph: a polished paragraph that could be inserted into a research paper. "
            "It should describe either the method or the results in a way that is logically coherent, "
            "technically accurate, and consistent with the current method_scheme and experiment_results."
        )
    # fallback
    return "Describe the artifact in a way that is precise, concise, and consistent with the project context."


def _format_recent_artifact_versions(
    recent_versions: List[Dict[str, Any]],
) -> str:
    """
    Format recent artifact versions for context.
    """
    if not recent_versions:
        return "[]"
    formatted = []
    for v in recent_versions:
        formatted.append({
            "event_id": v.get("event_id"),
            "time_index": v.get("time_index"),
            "text": v.get("text"),
        })
    return json.dumps(formatted, ensure_ascii=False, indent=2)



def build_research_multiple_artifacts_prompt(
    topic: str,
    subject: str,
    artifact_labels: List[str],
    event: Dict[str, Any],
    prior_events: List[Dict[str, Any]],
    required_artifacts_recent_versions: Dict[str, List[Dict[str, Any]]],
    current_artifact_state: Dict[str, Optional[str]],
) -> List[Dict[str, str]]:
    """
    Build messages for generating MULTIPLE artifact types' content at ONE event in a single call.

    Inputs:
    - topic, subject: from the timeline events.
    - artifact_labels: list of artifact labels to generate (subset of RESEARCH_ARTIFACT_LABELS).
    - event: current event dict (event_id, time_index, domain, event_type, description, reason, required_artifacts, generated_artifacts).
    - prior_events: list of dicts of previous events (for compact history, last 5 events).
    - required_artifacts_recent_versions: mapping label -> list of recent versions (most recent only).
    - current_artifact_state: mapping label -> latest text (or None).
    """
    
    # Validate all labels
    for label in artifact_labels:
        if label not in RESEARCH_ARTIFACT_LABELS:
            raise ValueError(f"Unsupported artifact_label for research: {label}")

    required_labels: List[str] = event.get("required_artifacts", []) or []
    event_type: str = event.get("event_type", "")

    # Get label guidance for all artifacts
    labels_guidance = []
    for label in artifact_labels:
        guidance = _artifact_label_guidance(label)
        labels_guidance.append(f"- {label}: {guidance}")

    system_prompt = f"""
You are generating MULTIPLE artifacts for a long-term research project in a single call.
You must be strictly faithful to the provided context and avoid any future leakage.

Domain: research
Project topic: {topic}
Project subject: {subject}

You are responsible for generating the following artifact labels: {", ".join(artifact_labels)}.

Artifact semantics:
{chr(10).join(labels_guidance)}

General writing constraints:
- Stay consistent with the project description and prior artifacts.
- Do NOT invent results, methods, or goals that contradict the given context.
- Do not generate overly long text.
- Each artifact text should be self-contained and understandable on its own.

{_multiple_artifacts_schema_block_research(artifact_labels)}
"""

    # Format information as natural language blocks
    prior_events_natural = _format_prior_events_natural(prior_events, last_k=5)
    required_artifacts_natural = _format_required_artifacts_natural(required_artifacts_recent_versions)
    
    # Build current event description
    current_event_desc = f"""Event Type: {event_type}
Description: {event.get("description", "")}
Reason: {event.get("reason", "")}
Required Artifacts: {", ".join(required_labels) if required_labels else "None"}
Generated Artifacts: {", ".join(artifact_labels)}"""

    # Build previous versions section for all artifacts
    previous_versions_sections = []
    for label in artifact_labels:
        prev_version_text = current_artifact_state.get(label)
        if prev_version_text:
            cur_ver_section = f"===== Previous version of [{label}] begin =====\n"
            cur_ver_section += f"{prev_version_text}\n"
            cur_ver_section += f"===== Previous version of [{label}] end ====="
            previous_versions_sections.append(cur_ver_section)
        else:
            # previous_versions_sections.append(f"{label}: None (this will be the first version)")
            cur_ver_section = f"===== Previous version of {label} begin =====\n"
            cur_ver_section += f"None (this will be the first version)\n"
            cur_ver_section += f"===== Previous version of {label} end ====="
            previous_versions_sections.append(cur_ver_section)

    previous_versions_section = "\n\n".join(previous_versions_sections)

    # Build the natural language user prompt
    user_prompt = f"""## Project Context
Topic: {topic}
Subject: {subject}

## Recent Event History (Last 5 Events)
{prior_events_natural}

## Current Event
{current_event_desc}

## Required Artifacts (The most recent version)
The following artifacts are required by the current event. Each artifact shows its most recent version (if available):
{required_artifacts_natural}

## Previous Version of Each Artifact to Generate

{previous_versions_section}

## Task Instructions
You need to generate or update the following artifacts: {", ".join(artifact_labels)}.

- For each artifact, if there is no previous version (shown as "None" above), create a sensible initial version based on the current event and the required artifacts shown above.
- If there is a previous version (shown above), refine or update it in light of the current event. Incorporate new results, clarify the plan, adjust goals, or make other appropriate updates.
- Generate ALL artifacts in a single JSON response following the schema provided.
- The output must follow the JSON schema exactly.
- The generated "artifact_text" field for each artifact should be in natural language text, not JSON.

## Additional Constraint
**IMPORTANT**:
- Each generated artifact must include enough specific, decision-relevant detail to plausibly support a realistic long-term research project, so that future events and queries can build on it coherently.
- If a previous version of an artifact is provided, update it by incorporating changes and improvements implied by the current event, while preserving the existing structure. Avoid unnecessary rewrites or large stylistic changes.
- Keep the generated artifacts succinct and easy to update across many iterations: avoid unnecessary exposition, avoid expanding into many subcomponents or tangential ideas, and prioritize a small set of high-signal details over exhaustive coverage.
"""

    # print("--------------- SYSTEM PROMPT -----------------")
    # print(system_prompt)
    # print("--------------- USER PROMPT -----------------")
    # print(user_prompt)

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]





def _artifact_label_guidance_tutoring(label: str) -> str:
    """
    Give label-specific writing guidance to the model (tutoring domain).
    """
    if label == "learning_objectives":
        return (
            "learning_objectives: the learner's current objectives, background, and constraints. "
            "Include target outcomes, current level, constraints (time/resources), and success signals."
        )
    if label == "study_plan":
        return (
            "study_plan: a staged plan with milestones and checkpoints. "
            "Specify sequencing, pacing, and what evidence will be used to decide progress (e.g., practice performance)."
        )
    if label == "teaching_notes":
        return (
            "teaching_notes: reusable teaching content tailored to the learner and objectives for **ONLY one lesson/event (not cumulative)**. "
            "Include key concepts/methods, common pitfalls, concise examples, and practical heuristics."
        )
    if label == "practice_record":
        return (
            "practice_record: a record of assigned practice and learner attempts for **ONLY one practice/event (not cumulative)**. "
            "Track what was assigned, what was attempted, outcomes."
        )
    if label == "feedback_summary":
        return (
            "feedback_summary: a concise, actionable summary of the learner’s current performance and gaps. "
            "Identify mistakes, explain corrections, and list concrete next actions (what to review + what to practice next)."
        )
    return "Describe the artifact precisely, concisely, and consistently with the tutoring context."


def build_tutoring_multiple_artifacts_prompt(
    topic: str,
    subject: str,
    artifact_labels: List[str],
    event: Dict[str, Any],
    prior_events: List[Dict[str, Any]],
    required_artifacts_recent_versions: Dict[str, List[Dict[str, Any]]],
    current_artifact_state: Dict[str, Optional[str]],
) -> List[Dict[str, str]]:
    """
    Build messages for generating MULTIPLE tutoring artifact types at ONE event in a single call.
    Reuses shared helpers:
      - _multiple_artifacts_schema_block_research()
      - _format_prior_events_natural()
      - _format_required_artifacts_natural()
    """
    for label in artifact_labels:
        if label not in TUTORING_ARTIFACT_LABELS:
            raise ValueError(f"Unsupported artifact_label for tutoring: {label}")

    required_labels: List[str] = event.get("required_artifacts", []) or []
    event_type: str = event.get("event_type", "")

    labels_guidance = [f"- {lbl}: {_artifact_label_guidance_tutoring(lbl)}" for lbl in artifact_labels]

    system_prompt = f"""
You are generating MULTIPLE artifacts for a long-term tutoring process in a single call.
You must be strictly faithful to the provided context and avoid any future leakage.

Domain: tutoring
Learning topic: {topic}
Subject: {subject}

You are responsible for generating the following artifact labels: {", ".join(artifact_labels)}.

Artifact semantics:
{chr(10).join(labels_guidance)}

General writing constraints:
- Stay consistent with the learner context, the current event, and prior artifacts.
- Do NOT invent learner achievements, assessments, or progress that contradict the given context.
- Do not generate overly long text.
- Each artifact text should be self-contained and understandable on its own.

{_multiple_artifacts_schema_block_research(artifact_labels)}
"""

    prior_events_natural = _format_prior_events_natural(prior_events, last_k=5)
    required_artifacts_natural = _format_required_artifacts_natural(required_artifacts_recent_versions)

    current_event_desc = f"""Event Type: {event_type}
Description: {event.get("description", "")}
Reason: {event.get("reason", "")}
Required Artifacts: {", ".join(required_labels) if required_labels else "None"}
Generated Artifacts: {", ".join(artifact_labels)}"""

    previous_versions_sections = []
    for label in artifact_labels:
        prev_version_text = current_artifact_state.get(label)
        if prev_version_text:
            cur_ver_section = f"===== Previous version of [{label}] begin =====\n"
            cur_ver_section += f"{prev_version_text}\n"
            cur_ver_section += f"===== Previous version of [{label}] end ====="
            previous_versions_sections.append(cur_ver_section)
        else:
            # previous_versions_sections.append(f"{label}: None (this will be the first version)")
            cur_ver_section = f"===== Previous version of {label} begin =====\n"
            cur_ver_section += f"None (this will be the first version)\n"
            cur_ver_section += f"===== Previous version of {label} end ====="
            previous_versions_sections.append(cur_ver_section)

    previous_versions_section = "\n\n".join(previous_versions_sections)

    user_prompt = f"""## Learning Context
Topic: {topic}
Subject: {subject}

## Recent Event History (Last 5 Events)
{prior_events_natural}

## Current Event
{current_event_desc}

## Required Artifacts (The most recent version)
The following artifacts are required by the current event. Each artifact shows its most recent version (if available):
{required_artifacts_natural}

## Previous Version of Each Artifact to Generate

{previous_versions_section}

## Task Instructions
You need to generate or update the following artifacts: {", ".join(artifact_labels)}.

- For each artifact, if there is no previous version (shown as "None" above), create a sensible initial version based on the current event and required artifacts.
- If there is a previous version (shown above), refine or update it in light of the current event.
- Generate ALL artifacts in a single JSON response following the schema provided.
- The output must follow the JSON schema exactly.
- The generated "artifact_text" field for each artifact should be in natural language text, not JSON.

## Additional Constraint
**IMPORTANT**:
- Each generated artifact must include enough specific, decision-relevant detail to plausibly support a realistic long-term research project, so that future events and queries can build on it coherently.
- If a previous version of an artifact is provided, update it by incorporating changes and improvements implied by the current event, while preserving the existing structure. Avoid unnecessary rewrites or large stylistic changes.
- Keep the generated artifacts succinct and easy to update across many iterations: avoid unnecessary exposition, avoid expanding into many subcomponents or tangential ideas, and prioritize a small set of high-signal details over exhaustive coverage.
- If generating "teaching_notes" or "practice_record", output a single-event snapshot only (non-cumulative): include ONLY the latest/current relevant content, and exclude all prior versions unless explicitly required.
- Do not use placeholder text. If some information is unavailable, either omit it or provide a reasonable concrete value based on the given context.
"""

    # print("--------------- SYSTEM PROMPT -----------------")
    # print(system_prompt)
    # print("--------------- USER PROMPT -----------------")
    # print(user_prompt)

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]



if __name__ == "__main__":
    # 写一个测试multiple的例子
    # build_research_multiple_artifacts_prompt(
    #     topic="Efficient training algorithms for large-scale transformer models",
    #     subject="CS-AI/ML",
    #     artifact_labels=["research_plan", "research_goals"],
    #     event={
    #         "event_id": "1",
    #         "time_index": "1",
    #         "domain": "research",
    #         "event_type": "research_plan",
    #         "description": "research_plan",
    #         "reason": "research_plan",
    #         "required_artifacts": [],
    #         "generated_artifacts": [],
    #     },
    #     prior_events=[],
    #     required_artifacts_recent_versions={
    #         "research_plan": [
    #             {
    #                 "event_id": "1",
    #                 "time_index": "1",
    #                 "text": "description 1",
    #             },
    #             {
    #                 "event_id": "2",
    #                 "time_index": "2",
    #                 "text": "description 2",
    #             },
    #         ],
    #         "research_goals": [
    #             {
    #                 "event_id": "1",
    #                 "time_index": "1",
    #                 "text": "description 1",    
    #             },
    #             {
    #                 "event_id": "2",
    #                 "time_index": "2",
    #                 "text": "description 2",
    #             },
    #         ],
    #     },
    #     current_artifact_state={
    #         "research_plan": "description 1——current",
    #         "research_goals": "description 1——current",
    #     },
    # )

    build_tutoring_multiple_artifacts_prompt(
        topic="Machine Learning",
        subject="CS-AI/ML",
        artifact_labels=["learning_objectives"],
        event={
            "event_id": "1",
            "time_index": "1",
            "domain": "tutoring",
            "event_type": "objective_clarification",
            "description": "description_current_event",
            "reason": "reason_current_event",
        },
        prior_events=[],
        required_artifacts_recent_versions={
            "learning_objectives": [
                {
                    "event_id": "1",
                    "time_index": "1",
                    "text": "learning_objectives",
                },
            ],
        },
        current_artifact_state={
            "learning_objectives": "learning_objectives",
        },
    )