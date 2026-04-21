# cross_session_prompt.py

QUERY_RULES = {
	"research": {
		"Plan & Design": {
			"research_plan": {
				"query": "Propose the next-stage plan for the project.",
				"essentials": ["research_goals", "research_plan", "experiment_results", "method_scheme"],
				"targets": ["research_plan"],
			},
			"method_scheme": {
				"query": "Design the method for the project.",
				"essentials": ["research_goals", "experiment_results", "method_scheme"],
				"targets": ["method_scheme"],
			},
		},

		"Revise": {
			"paper_paragraph": {
				"query": "Refine the current paper paragraph.",
				"essentials": ["paper_paragraph", "experiment_results", "method_scheme", "research_goals"],
				"targets": ["paper_paragraph"],
			},
			"method_scheme": {
				"query": "Refine the current method.",
				"essentials": ["method_scheme", "research_goals", "experiment_results"],
				"targets": ["method_scheme"],
			},
			"research_plan": {
				"query": "Improve and refine the current research plan.",
				"essentials": ["research_plan", "research_goals", "experiment_results"],
				"targets": ["research_plan"],
			},
		},

		"Analyze & Critique": {
			"method_scheme": {
				"query": "Analyze the current method and offer a critique.",
				"essentials": ["method_scheme", "research_goals", "experiment_results"],
				"targets": ["method_scheme"],
			},
			"research_plan": {
				"query": "Analyze the current research plan and offer a critique.",
				"essentials": ["research_plan", "research_goals", "experiment_results"],
				"targets": ["research_plan"],
			},
		},

		"Concept Explanation": {
			"query": "Explaine [concept] to me.",
			"concept": "to be synthesized",
		},
	},

	"tutoring": {
		"Plan & Design": {
			"study_plan": {
				"query": "Propose the next-stage study plan for me.",
				"essentials": ["learning_objectives", "study_plan", "practice_record", "feedback_summary"],
				"targets": ["study_plan"],
			},
			"practice_record": {
				"query": "Propose the next practice for me.",
				"essentials": ["learning_objectives", "study_plan", "teaching_notes"],
				"targets": ["practice_record"],
			},
			"teaching_notes": {
				"query": "Draft the teaching notes for the current lesson.",
				"essentials": ["learning_objectives", "study_plan"],
				"targets": ["teaching_notes"],
			},
		},

		"Revise": {
			"teaching_notes": {
				"query": "Refine the current teaching notes.",
				"essentials": ["teaching_notes", "learning_objectives", "practice_record", "feedback_summary"],
				"targets": ["teaching_notes"],
			},
			"study_plan": {
				"query": "Improve and refine the current study plan.",
				"essentials": ["study_plan", "learning_objectives", "practice_record", "feedback_summary"],
				"targets": ["study_plan"],
			},
			"feedback_summary": {
				"query": "Refine the feedback to the current practice results for me.",
				"essentials": ["practice_record", "learning_objectives", "teaching_notes", "feedback_summary"],
				"targets": ["feedback_summary"],
			},
		},

		"Analyze & Critique": {
			"study_plan": {
				"query": "Analyze the current study plan and offer a critique.",
				"essentials": ["study_plan", "learning_objectives", "feedback_summary"],
				"targets": ["study_plan"],
			},
			"teaching_notes": {
				"query": "Analyze the current teaching notes and offer a critique.",
				"essentials": ["teaching_notes", "learning_objectives", "feedback_summary"],
				"targets": ["teaching_notes"],
			},
		},
		"Concept Explanation": {
			"query": "Explain [concept] to the learner.",
			"essentials": ["learning_objectives", "teaching_notes"],
			"concept": "to be synthesized",
		},
	},
}


def build_task_description(domain: str, task_name: str) -> str:
    """
    Build a task description based on QUERY_RULES.
    
    Args:
        domain: Domain type, either "research" or "tutoring"
        task_name: Name of the task (e.g., "Plan & Design", "Revise")
    
    Returns:
        A description string for the task
    """
    if domain not in QUERY_RULES:
        raise ValueError(f"Invalid domain: {domain}")
    
    domain_rules = QUERY_RULES[domain]
    if task_name not in domain_rules:
        raise ValueError(f"Invalid task name: {task_name} for domain {domain}")
    
    task_rules = domain_rules[task_name]
    
    # Handle Concept Explanation separately
    if task_name == "Concept Explanation":
        description = f"""The "{task_name}" task involves explaining concepts to the user.
The user may request explanations of various concepts that arise during the project.
The assistant should provide clear, contextual explanations that help the user understand these concepts in relation to their current work."""
        return description
    
    # For other tasks, build description from available targets
    target_artifacts = []
    example_queries = []
    
    for target_name, target_info in task_rules.items():
        if isinstance(target_info, dict) and "query" in target_info:
            target_artifacts.append(target_name)
            example_queries.append(f'"{target_info["query"]}" (targeting {target_name})')
    
    if not target_artifacts:
        description = f"""The "{task_name}" task involves helping the user with various project-related activities."""
    else:
        artifacts_list = ", ".join(target_artifacts)
        queries_list = "\n  - ".join(example_queries)
        
        description = f"""The "{task_name}" task involves helping the user with various project-related activities.

Possible target artifacts for this task include: {artifacts_list}

Example user requests for this task:
  - {queries_list}

The user may request help with any of these targets at different stages of the project. The interactions should naturally vary across different target artifacts to reflect the diverse needs that arise during the timeline."""
    
    return description


def build_cross_session_prompt(
    events: list,
    domain: str,
    task_name: str,
    task_description: str,
    preference_summary: str,
    num_interactions: int = 5,
) -> str:
    """
    Build a prompt for generating cross-session context summary.
    
    Args:
        events: List of event dictionaries from the timeline
        domain: Domain type, either "research" or "tutoring"
        task_name: Name of the task (e.g., "Plan/Design", "Revise")
        task_description: Description of the task
        preference_summary: Summary of user preference for this task
        num_interactions: Number of interactions to generate (default 5)
    
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
    
    # Format events timeline
    events_text = ""
    for event in events:
        event_id = event.get("event_id", "")
        event_type = event.get("event_type", "")
        description = event.get("description", "")
        topic = event.get("topic", "")
        subject = event.get("subject", "")
        time_index = event.get("time_index", "")
        
        events_text += f"Event {event_id} ({event_type}):\n"
        events_text += f"  Topic: {topic}\n"
        events_text += f"  Subject: {subject}\n"
        events_text += f"  Description: {description}\n"
        if event.get("reason"):
            events_text += f"  Reason: {event.get('reason')}\n"
        events_text += "\n"
    
    prompt = f"""You are simulating interactions between a user and an AI assistant during {project_type}. {project_context}

You are given:
1. A complete timeline of events that occurred during this project
2. A task type and its description
3. The user's preference for this task type

Given Information:
task type: {task_name}
task description: {task_description}
task-related user preferences: {preference_summary}

Your task:
1. Simulate {num_interactions} realistic interactions where the user requests help with the "{task_name}" task at different stages of the timeline.
2. Each interaction should:
   - Occur at a specific event_id in the timeline (choose events that naturally trigger this type of request)
   - Include the user's request (which should implicitly reflect the user's preference)
   - Include the assistant's response (which should subtly embody the user's preference in its approach, structure, and reasoning)
3. The interactions should be distributed across different stages of the timeline (early, middle, late stages) to show how the task evolves.
4. After generating all interactions, provide a summary that captures the overall pattern of how the user and assistant engaged with this task throughout the timeline.

Important constraints:
- The user's preference should be IMPLICITLY reflected in the interactions, not explicitly stated.
- The interactions should feel natural and realistic, as if they occurred during the actual project timeline.
- The assistant's responses should be substantive and helpful, demonstrating the preference through their reasoning and structure.
- The summary should provide a concise overview of the task-related interactions, including:
  * A brief summary of when and how the "{task_name}" task was requested at different stages or events without mentioning event IDs, timestamps, or any other explicit event identifiers.
  * Key examples or patterns from the interactions that illustrate the engagement approach
  * How the user's preference manifested implicitly across these interactions (without explicitly stating the preference)

Timeline Events:
{events_text}

Output your response as a valid JSON object with the following structure:
{{
  "interactions": [
    {{
      "event_id": "e_01",
      "user_request": "The user's request text (implicitly reflecting preference)",
      "assistant_response": "The assistant's response text (subtly embodying preference)"
    }},
    ...
  ],
  "summary": "A comprehensive summary of how the user and assistant engaged with the '{task_name}' task throughout the timeline, highlighting how the preference manifested implicitly across interactions."
}}

Generate exactly {num_interactions} interactions, distributed across the timeline stages.
"""
    return prompt


if __name__ == "__main__":
    # Example usage
    events = [
        {
            "event_id": "e_01",
            "time_index": 1,
            "event_type": "proposal",
            "description": "Initial project proposal",
            "topic": "Test Topic",
            "subject": "Test Subject",
        },
        {
            "event_id": "e_02",
            "time_index": 2,
            "event_type": "proposal",
            "description": "Initial project proposal",
            "topic": "Test Topic",
            "subject": "Test Subject",
        },
    ]
    task_name = "Plan & Design"
    task_description = build_task_description("research", task_name)
    print(build_cross_session_prompt(
        events=events,
        domain="research",
        task_name=task_name,
        task_description=task_description,
        preference_summary="The user wants the next-stage plan to advance along the project's existing trajectory.",
    ))

