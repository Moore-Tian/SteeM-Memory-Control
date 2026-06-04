"""
timeline_prompts.py

Domain-specific prompts for generating long-term task timelines.

Each domain (research, engineering, academic writing, teaching)
has its own prompt builder, with domain-specific stages, examples,
and coverage constraints.

The event JSON schema is shared across domains.
"""

import json
from typing import List, Dict, Any

# ============================================================
# Event Schema Block
# ============================================================

def _event_schema_block_research() -> str:
    """Description of the JSON event schema for RESEARCH domain."""
    return """
Event schema (JSON) for each item in the list:

{
  "event_id": "e_01",
  "time_index": 1,
  "domain": "research",
  "event_type": "proposal | scheme_design | pilot_experiments | main_experiments | analysis | writing",
  "description": "1–3 sentences describing what happened at this step.",
  "required_artifacts": [
    // one or more strings chosen ONLY from:
    // "research_plan", "research_goals", "experiment_results", "method_scheme", "paper_paragraph"
  ],
  "generated_artifacts": [
    // one or more strings chosen ONLY from:
    // "research_plan", "research_goals", "experiment_results", "method_scheme", "paper_paragraph"
  ]
  "reason": "1–3 sentences explaining why this event is a reasonable next step, given the previous events and available artifacts."
}

All events must follow this schema exactly. Use JSON-compatible values only.
Strings must be plain strings; null means “not provided or not updated in this event”.
- "required_artifacts" and "generated_artifacts" are lists of STRING LABELS, not full texts.
- allowed artifact labels are exactly:
  ["research_plan", "research_goals", "experiment_results", "method_scheme", "paper_paragraph"].
"""

def _event_schema_block_tutoring() -> str:
    """Description of the JSON event schema for TUTORING domain."""
    return """
Event schema (JSON) for ONE event object:

{
  "event_id": "e_01",
  "time_index": 1,
  "domain": "tutoring",
  "event_type": "diagnostic_session | plan_milestones | lesson | practice | review | materials_revision | end",
  "description": "1–3 sentences describing what happened at this step.",
  "required_artifacts": [
    // zero or more strings chosen ONLY from:
    // "learning_objectives", "study_plan", "teaching_notes", "practice_record", "feedback_summary"
  ],
  "generated_artifacts": [
    // zero or more strings chosen ONLY from:
    // "learning_objectives", "study_plan", "teaching_notes", "practice_record", "feedback_summary"
  ],
  "reason": "1–3 sentences explaining why this event is a reasonable next step, given the previous events and available artifacts."
}

Rules:
- All events must follow this schema exactly. Use JSON-compatible values only.
- "required_artifacts" and "generated_artifacts" are lists of STRING LABELS, not full texts.
- Allowed artifact labels are exactly:
  ["learning_objectives", "study_plan", "teaching_notes", "practice_record", "feedback_summary"].
- If event_type is "end", both "required_artifacts" and "generated_artifacts" must be empty lists: [].
"""



# ============================================================
# Research timeline prompt
# ============================================================


def build_research_next_event_prompt(
    topic: str,
    existing_events: List[Dict[str, Any]],
    next_index: int,
    end_available: bool,
) -> str:
    """
    Build a prompt for generating the NEXT research event, given all previously generated events.
    """

    header = f"""
You are generating the NEXT event in a long-term research project timeline.

Task domain: research
Project topic: {topic}

The timeline should reflect how a real research project actually progresses over time:
- starting from an initial proposal and vague ideas,
- going through multiple rounds of method design, pilot experiments, main experiments, analysis, and writing,
- possibly revisiting earlier steps (e.g., revising goals, redesigning methods, rerunning experiments) when results are unsatisfactory or new insights emerge.
"""

    stages_block = """
The research project uses the following HIGH-LEVEL EVENT TYPES:

1. "proposal"          – Define or revise research questions, goals, and a high-level plan.
2. "scheme_design"     – Design or revise the experimental / methodological scheme.
3. "pilot_experiments" – Run small-scale pilot experiments to sanity-check the idea.
4. "main_experiments"  – Run main experiments, refine the scheme, and collect results.
5. "analysis"          – Analyze results and summarize the current research status.
6. "writing"           – Draft or revise parts of the paper describing methods and results.

Important:
- These event types may appear MULTIPLE times and in an interleaved order. For example, after an "analysis" event, it is realistic to go back to "scheme_design" and then run new "main_experiments".
- It is also realistic to refine "proposal" (goals/plan) later in the project.
- Over the whole timeline, each of the six event types should appear at least once as the "event_type" of some event.
"""

    coverage_block = """
Artifact types for the project:

- "research_plan"      – high-level plan / roadmap for the project.
- "research_goals"     – current research questions and goals.
- "experiment_results" – results or findings from pilot/main experiments.
- "method_scheme"      – current design of the method / experimental setup.
- "paper_paragraph"    – text snippets describing methods or results (for the paper).

Across ALL events in the timeline, each of these artifact types should be generated at least once in some event's "generated_artifacts" list, so that later user queries are well-grounded.

An event should not require an artifact that has never been generated before.
"""

    schema_block = _event_schema_block_research()

    existing_block = (
        "Events generated so far (in order):\n\n"
        + json.dumps(existing_events, ensure_ascii=False, indent=2)
    )

    if not end_available:
        generation_instructions = f"""
You are now generating event number {next_index}.

1. Use the previous events above as context to decide what is a reasonable next step.
- Respect dependencies: "required_artifacts" of the new event must have been generated in at least one previous event.
- It is allowed (and realistic) to revisit earlier stages, such as refining the research plan, redesigning the method, or rerunning experiments.

2. Output format:
- Output ONE JSON object that follows the event schema exactly (not a list).
- Set "time_index" = {next_index}.
- Set "domain" = "research".
- Set "event_type" to one of: 
    ["proposal", "scheme_design", "pilot_experiments", "main_experiments", "analysis", "writing"].
- "required_artifacts" and "generated_artifacts" must each contain ONE OR MORE labels chosen ONLY from:
    ["research_plan", "research_goals", "experiment_results", "method_scheme", "paper_paragraph"].

3. Reasoning:
- The "reason" field should briefly explain why this event is a reasonable next step, referencing relevant previous events and artifacts when appropriate.

Now output ONLY the JSON object for this next event, with no extra text.
"""
    else:
        generation_instructions = f"""
You are now generating event number {next_index}.

The minimum required number of events for this timeline has already been reached. From this point on, you should briefly check whether the project is already in a natural end state. If it clearly looks complete, it is reasonable to terminate the timeline with an "end" event; if not, you should add a normal event that makes meaningful progress to the end.

1. Use the previous events above as context to decide what is a reasonable next step.
- Respect dependencies: "required_artifacts" of the new event must have been generated in at least one previous event.
- It is allowed (and realistic) to revisit earlier stages, such as refining the research plan, redesigning the method, or rerunning experiments.
- If you decide not to use "end" yet, the new event should have a clear purpose (for example, stabilizing the method_scheme, consolidating experiment_results, clarifying research_goals or research_plan, or improving paper_paragraph), rather than opening an entirely new direction without motivation.

2. Choice of event_type:
- There are SIX normal research event types:
  ["proposal", "scheme_design", "pilot_experiments", "main_experiments", "analysis", "writing"].
- There is ONE special event type used ONLY to terminate the timeline:
  "end".

Usage rules for "end":
- "end" must be used at most once in the entire timeline.
- After an event with event_type = "end", no further events will be generated.
- You should choose "end" only when the project already has a reasonably stable design, sufficiently informative results, and reasonably revised writing.
- If "end" is chosen for the current event, you do NOT need to generate any artifacts (both "required_artifacts" and "generated_artifacts" must be empty lists []).

3. Output format:
- Output ONE JSON object that follows the event schema exactly (not a list).
- Set "time_index" = {next_index}.
- Set "domain" = "research".
- Set "event_type" to one of: ["proposal", "scheme_design", "pilot_experiments", "main_experiments", "analysis", "writing", "end"].
- For NORMAL events (not "end"), "required_artifacts" and "generated_artifacts" must each contain ONE OR MORE labels chosen ONLY from: ["research_plan", "research_goals", "experiment_results", "method_scheme", "paper_paragraph"].
- For the TERMINATION event ("end"), "required_artifacts" and "generated_artifacts" must BOTH be empty lists: [].

4. Reasoning:
- For NORMAL events, the "reason" field should briefly explain why this event is a reasonable next step and how it contributes to the overall progress of the project.
- For the TERMINATION event ("end"), the "reason" should explain why the project can now be considered complete (for example, methods are stable, results are conclusive, and the writing has been sufficiently revised).

Now output ONLY the JSON object for this next event, with no extra text.
"""

    # 直接返回构造之后的 message（把除了 existing_block和generation instruction 之外的作为 system prompt）
    system_prompt = header + "\n" + stages_block + "\n" + coverage_block + "\n" + schema_block
    user_prompt = existing_block + "\n\n" + generation_instructions

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]



# ============================================================
# Tutoring timeline prompt
# ============================================================

def build_tutoring_next_event_prompt(
    topic: str,
    existing_events: List[Dict[str, Any]],
    next_index: int,
    end_available: bool,
) -> List[Dict[str, str]]:
    """
    Build messages for generating the NEXT tutoring event, given all previously generated events.
    """

    header = f"""
You are generating the NEXT event in a long-term tutoring timeline.

Task domain: tutoring
Learning topic: {topic}

The timeline should reflect how real tutoring progresses over time: it typically starts with an initial diagnostic session, then cycles through planning milestones, lessons, practice, review/feedback, and materials revision. The process may revisit earlier steps when progress is unstable or new difficulties emerge.
"""

    stages_block = """
The tutoring process uses the following HIGH-LEVEL EVENT TYPES:

1. "objective_clarification" – Clarify the learner’s learning objectives, constraints, and self-reported background, and produce an initial draft of the study plan when appropriate.
2. "plan_milestones"         – Define or revise milestones and checkpoints, and produce or update a staged study plan with clear sequencing and expectations.
3. "lesson"                  – Teach a concept/method/pattern aligned with the current objectives or recent difficulties, and update teaching notes for reuse (e.g., definitions, steps, examples, common pitfalls).
4. "practice"                – Assign or run targeted practice, and record attempts or intermediate steps in a structured way for later review.
5. "review"                  – Review practice attempts, identify mistakes and misconceptions, and produce actionable feedback, corrections, and next-step guidance.
6. "materials_revision"      – Revise teaching notes, study plan wording, or reusable templates/examples in response to learner needs and observed issues from practice/review, improving clarity, consistency, and reusability.

Important:
- These event types may appear MULTIPLE times and in an interleaved order. For example, after a "review" event, it is realistic to perform "materials_revision" and then run another "practice" event.
- It is also realistic to revisit "objective_clarification" or "plan_milestones" later to adjust objectives or pacing based on current learner needs and observed issues.
- Over the whole timeline, each of the six types should appear at least once as the "event_type" of some event.
"""

    coverage_block = """
Artifact types for the tutoring timeline:

- "learning_objectives" – explicit learning targets and constraints (scope, time horizon, priorities, success criteria).
- "study_plan"          – staged plan with milestones, methods, checkpoints, and resource usage guidance.
- "teaching_notes"      – reusable notes explaining concepts/methods/patterns with examples and common pitfalls.
- "practice_record"     – practice tasks plus a structured record of attempts/steps and outcomes.
- "feedback_summary"    – review feedback summarizing issues, corrections, and actionable next steps.

Coverage constraints:
- Across ALL events in the timeline, each artifact type above should be generated at least once in some event's "generated_artifacts" list.
- An event should not require an artifact that has never been generated before.
"""

    schema_block = _event_schema_block_tutoring()

    existing_block = (
        "Events generated so far (in order):\n\n"
        + json.dumps(existing_events, ensure_ascii=False, indent=2)
    )

    if not end_available:
        generation_instructions = f"""
You are now generating event number {next_index}.

1. Use the previous events above as context to decide what is a reasonable next step.
- Respect dependencies: labels in "required_artifacts" should have appeared in "generated_artifacts" of at least one previous event.
- Exception for the very beginning: if this is the first event or there are no prior artifacts yet, an "objective_clarification" event may use an empty required_artifacts list [].
- It is allowed (and realistic) to revisit earlier stages, such as revising objectives, adjusting the study plan, repeating practice after feedback, or refining teaching notes based on learner needs.

2. Output format:
- Output ONE JSON object that follows the event schema exactly (not a list).
- Set "time_index" = {next_index}.
- Set "domain" = "tutoring".
- Set "event_type" to one of:
     ["objective_clarification", "plan_milestones", "lesson", "practice", "review", "materials_revision"].
- "generated_artifacts" should contain ONE OR MORE labels chosen ONLY from: 
    ["learning_objectives", "study_plan", "teaching_notes", "practice_record", "feedback_summary"].
- For events after the first step, "required_artifacts" should normally contain ONE OR MORE labels chosen ONLY from: 
    ["learning_objectives", "study_plan", "teaching_notes", "practice_record", "feedback_summary"].

3. Reasoning:
- The "reason" field should briefly explain why this event is a reasonable next step, referencing relevant previous events and artifacts when appropriate.

Now output ONLY the JSON object for this next event, with no extra text.
"""
    else:
        generation_instructions = f"""
You are now generating event number {next_index}.

The minimum required number of events for this timeline has already been reached. From this point on, you should briefly check whether the tutoring process is already in a natural end state. If it clearly looks complete, it is reasonable to terminate the timeline with an "end" event; if not, add a normal event that makes meaningful progress to the end.

1. Use the previous events above as context to decide what is a reasonable next step.
- Respect dependencies: labels in "required_artifacts" should have appeared in "generated_artifacts" of at least one previous event.
- It is allowed (and realistic) to revisit earlier stages, such as revising objectives, updating milestones, refining teaching notes for recurring misconceptions, repeating targeted practice based on feedback, or revising materials in response to learner needs.
- If you decide not to use "end" yet, the new event should have a clear purpose (for example, validating progress with practice, consolidating feedback into teaching notes, improving the study plan wording, or addressing a remaining weakness), rather than expanding scope without motivation.

2. Choice of event_type:
- There are SIX normal tutoring event types: ["objective_clarification", "plan_milestones", "lesson", "practice", "review", "materials_revision"].
- There is ONE special event type used ONLY to terminate the timeline: "end".

Usage rules for "end":
- "end" must be used at most once in the entire timeline.
- After an event with event_type = "end", no further events will be generated.
- Choose "end" only when the learning objectives are substantially met, performance is stable, and remaining issues are minor or routine.
- If "end" is chosen for the current event, you do NOT need to generate any artifacts (both "required_artifacts" and "generated_artifacts" must be empty lists []).

3. Output format:
- Output ONE JSON object that follows the event schema exactly (not a list).
- Set "time_index" = {next_index}.
- Set "domain" = "tutoring".
- Set "event_type" to one of: ["objective_clarification", "plan_milestones", "lesson", "practice", "review", "materials_revision", "end"].
- For NORMAL events (not "end"), "generated_artifacts" must contain ONE OR MORE labels chosen ONLY from: ["learning_objectives", "study_plan", "teaching_notes", "practice_record", "feedback_summary"].
- For NORMAL events (not "end"), "required_artifacts" should normally contain ONE OR MORE labels chosen ONLY from: ["learning_objectives", "study_plan", "teaching_notes", "practice_record", "feedback_summary"].
- For the TERMINATION event ("end"), "required_artifacts" and "generated_artifacts" must BOTH be empty lists: [].

4. Reasoning:
- For NORMAL events, the "reason" field should briefly explain why this event is a reasonable next step and how it contributes to progress toward the learning objectives.
- For the TERMINATION event ("end"), the "reason" should explain why the tutoring process can now be considered complete (for example, objectives are met, performance is stable, and feedback has been incorporated into reusable notes and plans).

Now output ONLY the JSON object for this next event, with no extra text.
"""

    system_prompt = header + "\n" + stages_block + "\n" + coverage_block + "\n" + schema_block
    user_prompt = existing_block + "\n\n" + generation_instructions

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]



# ============================================================
# Engineering timeline prompt
# ============================================================

def build_engineering_timeline_prompt(
    topic: str,
    overall_goal: str,
    n_events_min: int = 8,
    n_events_max: int = 12,
) -> str:
    """
    Build a prompt for generating a long-term engineering/system timeline,
    e.g., for an LLM-based agent system or large-scale service.
    """

    header = f"""
You are generating a long-term engineering project timeline.

Task domain: engineering
System topic: {topic}
Overall goal: {overall_goal}

The timeline should describe how a realistic engineering or system project
progresses over iterations: from understanding the current system and bottlenecks,
through design, implementation, deployment, monitoring, and long-term evolution.
"""

    stages_block = """
The engineering project is divided into the following ordered STAGES:

1. "baseline_and_bottlenecks" – Measure current performance and identify bottlenecks.
2. "perf_targets" – Set performance targets and clarify constraints (latency, cost, reliability).
3. "design_optimization" – Design optimization strategies and system changes.
4. "implementation" – Implement changes in code, add tests, and refactor as needed.
5. "deployment_and_monitoring" – Roll out changes and monitor real-world performance.
6. "long_term_evolution" – Refine architecture and plan long-term evolution of the system.

Within each stage, generate 1–3 concrete events that would naturally occur.
Examples of event_type values: "baseline_measurement", "bottleneck_analysis",
"optimization_brainstorming", "design_doc_written", "implementation_event",
"deployment_event", "postmortem_meeting", etc.
"""

    coverage_block = """
Coverage constraints (very important for later queries):

Across ALL events, the following artifact fields MUST be non-null
at least once:

- "system_arch"   (for planning and tutoring about the current architecture)
- "perf_metrics"  (for understanding current performance and progress)
- "design_doc"    (for revising and critiquing the current design)
- "code_snippet"  (for revising or explaining specific code behavior)

If an artifact is not relevant for a given event, keep it as null.
Only fill fields that are naturally produced or updated by that event.
"""

    schema_block = _event_schema_block()

    generation_instructions = f"""
Generation instructions:

- Output a single top-level JSON list of events (no extra keys).
- Generate between {n_events_min} and {n_events_max} events in total.
- Events must be ordered by time_index starting at 1.
- time_label can be simple (e.g., "Sprint 1", "Sprint 2", ... or "Month 1").
- event_type must be a short snake_case label (e.g., "baseline_measurement").
- description should be concrete but concise (1–3 sentences).
- artifacts should only be filled when that event naturally produces or
  updates that piece of information. Otherwise keep it as null.

Later, we will use this timeline to support user queries such as:
- proposing system performance optimization directions,
- revising design documents or code for clarity,
- critiquing a design in terms of maintainability, extensibility, and risk,
- tutoring-style explanations of system architecture and engineering practices.

Therefore, ensure that architecture, metrics, designs, and code evolve coherently.
Now output ONLY the JSON list of events, with no extra text.
"""

    return (
        header
        + "\n"
        + stages_block
        + "\n"
        + schema_block
        + "\n"
        + coverage_block
        + "\n"
        + generation_instructions
    )


# ============================================================
# Academic writing timeline prompt
# ============================================================

def build_writing_timeline_prompt(
    topic: str,
    overall_goal: str,
    n_events_min: int = 8,
    n_events_max: int = 12,
) -> str:
    """
    Build a prompt for generating a long-term academic writing timeline,
    focusing on a single paper or closely related set of manuscripts.
    """

    header = f"""
You are generating a long-term academic writing timeline for a research paper.

Task domain: academic_writing
Paper topic: {topic}
Overall goal: {overall_goal}

The timeline should describe how the manuscript evolves over time:
from framing the research question and contributions, through outlining,
drafting, internal feedback, external review, and revisions.
"""

    stages_block = """
The academic writing process is divided into the following ordered STAGES:

1. "framing" – Clarify the research question, contributions, and target venue.
2. "outline" – Draft the overall structure and section outline.
3. "section_drafts" – Write first drafts of key sections (Intro, Method, Experiments, etc.).
4. "internal_feedback" – Collect feedback from advisors or co-authors and adjust the story.
5. "external_review" – Submit to a venue, receive reviews, and understand revision requirements.
6. "revision_rounds" – Perform major/minor revisions and prepare camera-ready version.

Within each stage, generate 1–3 concrete events that would naturally occur.
Examples of event_type values: "framing_meeting", "outline_creation",
"intro_draft_written", "advisor_feedback", "reviewer_comment_received",
"revision_round", etc.
"""

    coverage_block = """
Coverage constraints (very important for later queries):

Across ALL events, the following artifact fields MUST be non-null
at least once:

- "paper_outline"    (for planning structural changes in the manuscript)
- "paper_paragraph"  (for revising and critiquing specific paragraphs/sections)
- "venue_info"       (for aligning writing with target venue norms)

If an artifact is not relevant for a given event, keep it as null.
Only fill fields that are naturally produced or updated by that event.
"""

    schema_block = _event_schema_block()

    generation_instructions = f"""
Generation instructions:

- Output a single top-level JSON list of events (no extra keys).
- Generate between {n_events_min} and {n_events_max} events in total.
- Events must be ordered by time_index starting at 1.
- time_label can be simple (e.g., "Month 1", "Month 2", ...).
- event_type must be a short snake_case label (e.g., "outline_creation").
- description should be concrete but concise (1–3 sentences).
- artifacts should only be filled when that event naturally produces or
  updates that piece of information. Otherwise keep it as null.

Later, we will use this timeline to support user queries such as:
- planning the next revision or restructuring of the paper,
- rewriting a paragraph to improve clarity and logic,
- critiquing a section in terms of rigor and alignment with the main contributions,
- tutoring-style explanations of good academic writing patterns for a given section.

Therefore, ensure that the outline, sections, and venue alignment evolve coherently.
Now output ONLY the JSON list of events, with no extra text.
"""

    return (
        header
        + "\n"
        + stages_block
        + "\n"
        + schema_block
        + "\n"
        + coverage_block
        + "\n"
        + generation_instructions
    )