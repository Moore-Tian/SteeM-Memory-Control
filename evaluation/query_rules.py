QUERY_RULES = {
	"research": {
		"Plan & Design": {
			"research_plan": {
				"query": "Propose a next-stage project plan, including key milestones, required resources, main risks and their reasons.",
				"essentials": ["research_goals", "research_plan", "experiment_results", "method_scheme"],
				"targets": ["research_plan"],
			},
			"method_scheme": {
				"query": "Design the project method, outlining core components, assumptions, and how it will be evaluated.",
				"essentials": ["research_goals", "experiment_results", "method_scheme"],
				"targets": ["method_scheme"],
			},
		},

		"Revise": {
			"paper_paragraph": {
				"query": "Refine the current paper paragraph to improve clarity, technical precision, coherence, and academic rigor.",
				"essentials": ["paper_paragraph", "experiment_results", "method_scheme", "research_goals"],
				"targets": ["paper_paragraph"],
			},
			"method_scheme": {
				"query": "Refine the current method to make the key components, assumptions, and evaluation procedure explicit and internally consistent.",
				"essentials": ["method_scheme", "research_goals", "experiment_results"],
				"targets": ["method_scheme"],
			},
			"research_plan": {
				"query": "Improve the current research plan to enhance clarity, coherence, feasibility, and alignment with the project goals.",
				"essentials": ["research_plan", "research_goals", "experiment_results"],
				"targets": ["research_plan"],
			},
		},

		"Analyze & Critique": {
			"method_scheme": {
				"query": "Analyze the current research plan and offer a critique, focusing on significance, coherence, feasibility, and risk.",
				"essentials": ["method_scheme", "research_goals", "experiment_results"],
				"targets": ["method_scheme"],
			},
			"research_plan": {
				"query": "Analyze the current research plan and offer a critique, focusing on coherence, feasibility, and risk.",
				"essentials": ["research_plan", "research_goals", "experiment_results"],
				"targets": ["research_plan"],
			},
		},

		"Concept Explanation": {
			"query": "Explain [concept] in clear terms, including intuition, key mechanism/steps, and a fewexamples.",
			"essentials": ["research_goals", "research_plan", "method_scheme"],
		},
	},


	"tutoring": {
		"Plan & Design": {
			"study_plan": {
				"query": "Propose a next-stage study plan, including key milestones, recommended practice focus, common pitfalls to watch for, and clear mastery criteria.",
				"essentials": ["learning_objectives", "study_plan", "practice_record", "feedback_summary"],
				"targets": ["study_plan"],
			},
			"practice_record": {
				"query": "Design the next practice set, specifying the target skills, expected solution patterns, and a simple rubric for checking answers.",
				"essentials": ["learning_objectives", "study_plan", "teaching_notes", "feedback_summary"],
				"targets": ["practice_record"],
			},
			"teaching_notes": {
				"query": "Draft the teaching notes for the current lesson, making the key rules/steps explicit and highlighting likely mistakes with brief fixes.",
				"essentials": ["learning_objectives", "study_plan", "feedback_summary"],
				"targets": ["teaching_notes"],
			},
		},

		"Revise": {
			"teaching_notes": {
				"query": "Refine the current teaching notes to improve clarity, coherence, and instructional usefulness, making key steps and common mistakes explicit.",
				"essentials": ["teaching_notes", "learning_objectives", "practice_record", "feedback_summary"],
				"targets": ["teaching_notes"],
			},
			"study_plan": {
				"query": "Improve the current study plan to enhance clarity, pacing, feasibility.",
				"essentials": ["study_plan", "learning_objectives", "practice_record", "feedback_summary"],
				"targets": ["study_plan"],
			},
			"feedback_summary": {
				"query": "Refine the feedback summary to be accurate, actionable, and well-calibrated, clearly stating strengths, key gaps, and the most important next steps.",
				"essentials": ["practice_record", "learning_objectives", "teaching_notes", "feedback_summary"],
				"targets": ["feedback_summary"],
			},
		},

		"Analyze & Critique": {
			"study_plan": {
				"query": "Analyze the current study plan and offer a critique, focusing on learning value, coherence, pacing, feasibility, and risk of common misconceptions.",
				"essentials": ["study_plan", "learning_objectives", "practice_record", "feedback_summary"],
				"targets": ["study_plan"],
			},
			"teaching_notes": {
				"query": "Analyze the current teaching notes and offer a critique, focusing on conceptual correctness, clarity, coverage of key pitfalls, and suitability for the learner’s level.",
				"essentials": ["teaching_notes", "learning_objectives", "practice_record", "feedback_summary"],
				"targets": ["teaching_notes"],
			},
		},

		"Concept Explanation": {
			"query": "Explain [concept] in clear terms, including intuition, key rules/steps, and a few examples.",
			"essentials": ["learning_objectives", "study_plan", "teaching_notes"],
		},
	}
}

