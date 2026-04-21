# concept_prompt.py

def build_concept_prompt(events: list, domain: str = "research") -> str:
	"""
	Build a prompt for generating knowledge-oriented concepts from a timeline of events.

	Args:
		events: List of event dictionaries, each containing event_id, description, etc.
		domain: Domain type, either "research" or "tutoring"

	Returns:
		A formatted prompt string for the LLM.
	"""
	# Determine project type description based on domain
	if domain == "research":
		project_type = "a long-term research project"
		project_context = "This is a research project where a researcher is working on a specific research topic, conducting experiments, analyzing results, and developing methods over time."
	elif domain == "tutoring":
		project_type = "a long-term tutoring/learning project"
		project_context = "This is a tutoring project where a learner is working with a tutor on a specific subject, practicing skills, receiving feedback, and progressing through learning objectives over time."
	else:
		raise ValueError("Invalid domain")
	events_text = ""
	for event in events:
		event_id = event.get("event_id", "")
		event_type = event.get("event_type", "")
		description = event.get("description", "")
		topic = event.get("topic", "")
		subject = event.get("subject", "")
		
		events_text += f"Event {event_id} ({event_type}):\n"
		events_text += f"  Topic: {topic}\n"
		events_text += f"  Subject: {subject}\n"
		events_text += f"  Description: {description}\n"
		if event.get("reason"):
			events_text += f"  Reason: {event.get('reason')}\n"
		events_text += "\n"

	prompt = f"""You are given a timeline of {project_type} with multiple events. {project_context}

Generate exactly 5 DISTINCT, KNOWLEDGE-ORIENTED concepts that a user would realistically ask to be explained at the current stage of this timeline.

Important:
- These concepts are NOT summaries of the event stage, workflow, or project-management notions.
- Do NOT output meta/process summaries (e.g., "manuscript preparation", "iterative refinement", "scaling experiments").
- Output domain concepts that are explainable as knowledge: definitions, mechanisms, rules, strategies or techniques tied to the timeline’s topic and what the user likely needs now.

What counts as a good concept:
- A concrete technical/subject-matter concept that can be explained independently.
- Plausibly triggered by the timeline context (what the user is doing/learning right now).
- Suitable for a "Concept Explanation" task.

Examples
- Research: "speculative decoding", "KV-cache"
- Tutoring: "unit rate", "k^2 vs k^3 scaling"

Constraints:
1) Generate exactly 5 concepts.
2) Each concept must be tied to ONE specific event_id that most motivates it.
3) All 5 concepts must be distinct and non-repeating.
4) Prefer concepts that are actionably relevant to the current stage (e.g., current confusion, next-step implied by the event).
5) Avoid overly broad concepts (e.g., "machine learning", "algebra"). Make them specific.

Timeline Events:
{events_text}

Output your response as a valid JSON array:
[
  {{
    "event_id": "e_01",
    "concept": "A specific knowledge concept (term/method/rule/strategy)",
    "reason": "Why this concept would be asked/explained now, grounded in the event and timeline context (not a summary of the event)."
  }},
  ...
]
"""
	return prompt



if __name__ == "__main__":
    events = [
        {
            "event_id": "e_01",
            "event_type": "Research",
            "description": "Formulating Research Objectives",
            "topic": "Research",
            "subject": "Research Objectives"
        },
        {
            "event_id": "e_02",
            "event_type": "Research",
            "description": "Iterative Algorithm Refinement",
            "topic": "Research",
            "subject": "Iterative Algorithm Refinement"
        },
        {
            "event_id": "e_03",
            "event_type": "Research",
            "description": "Scaling Research Experiments",
            "topic": "Research",
            "subject": "Scaling Research Experiments"
        },
    ]
    print(build_concept_prompt(events, domain="research"))