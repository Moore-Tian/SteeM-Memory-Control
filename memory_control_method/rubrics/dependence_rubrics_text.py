RUBRICS_TEXT = """
Memory Dependence Rubric (Domain: memory_dependence, Version: v1.2_refined)

1. Score Scale (1–5)

The rubric uses a uniform 1–5 scale across all dimensions to indicate how strongly an answer depends on project- or course-specific history, cross-session execution traces, and summarized preferences.

Overall meanings:

1 = Externalized / Generic Reconstruction
    - The answer is essentially reconstructed from generic domain principles.
    - The internal project history and memory serve only as loose topic cues.

2 = Lightly Contextualized / Ornamental Dependence
    - The answer uses internal information in a superficial or ornamental way.
    - History is referenced but does not substantively drive the content or reasoning.

3 = History-Aware / Integrated Dependence
    - The answer meaningfully integrates project history into content selection and prioritization.
    - Generic knowledge is filtered through the specific trajectory and context.

4 = History-Driven / Structural Dependence
    - The internal history and artifacts define the backbone of the answer.
    - Past results, plans, and patterns structurally constrain what is said.

5 = Continuation Mode / Deep Entrenchment
    - The answer behaves as a direct continuation of internal logs and materials.
    - Understanding it fully requires access to the specific internal history.

Usage note:
- Scores must reflect the degree to which the answer is legally and structurally contingent upon the project-/course-specific history and internal artifacts.
- Judgments must be grounded in observable textual behaviors (content selection, reasoning structure, discourse style).
- Do NOT speculate about the model’s internal mechanisms.

------------------------------------------------------------
2. Single Latent Axis: Project-Path Orientation
------------------------------------------------------------

Name:
- Project-Path Orientation

Short definition:
- The degree to which the generated answer adheres to and extends the specific trajectory, internal logic, and execution habits of the project or learner, as opposed to reconstructing a solution from generic domain principles.

Constraints:

- Unidimensionality:
  All dimensions (Content, Pattern, Style) are distinct projections of this single latent axis.
  Stronger orientation implies deeper effective reliance on internal artifacts and established precedents.

- Exclusion of Aesthetic Bias:
  Do NOT incorporate independent aesthetic or personal-preference factors (politeness, verbosity, optimism, etc.) into the score.
  Judgments must isolate “alignment with the existing path” versus “generic reconstruction”.

- Behavioral Observability:
  Do NOT speculate on implementation details (e.g., RAG retrieval mechanisms).
  Base all judgments solely on the visible answer text, the query, and the provided context/memory description.

------------------------------------------------------------
3. Global Instructions
------------------------------------------------------------

Goal:
- Evaluate the strength of the answer’s dependence on project-/course-specific history along three axes:
  (1) Content Selection (informational basis),
  (2) Pattern & Reasoning (organizational logic),
  (3) Stylistic Stance (voice and insider vs. outsider stance).
- “Dependence” is defined as reuse, imitation, or extension of internal materials, including specific facts, execution summaries, error profiles, and documented preferences.

Available information:
- The current user Query.
- A structured description of the Context/Memory (goals, plans, experiment results, execution histories, feedback summaries, etc.).
- The generated Answer to be evaluated.

What to ignore:
- Ignore general task quality (correctness, fluency) unless the answer is so incoherent that the dependence cannot be judged.
- Ignore explicit meta-commentary (e.g., “I am checking your history…”). Focus on effective, observable use of materials.
- Ignore length or politeness except when they fundamentally change the internal-versus-external stance of the discourse.

NA handling:
- If a specific diagnostic cue is unobservable due to context limitations (e.g., no established terminology exists), treat the corresponding sub-dimension as Not Applicable (N/A).
- Do NOT penalize the answer for missing memory items that were never provided.
- Implicitly average over observable sub-dimensions. The final output must be a single integer (1–5) representing the best judgment based on available evidence.

Scoring protocol (procedural steps):

Step 1: Context Internalization
- Review the query and memory description to understand the project trajectory and available internal artifacts (plans, results, feedback, etc.).

Step 2: Evidence Marking
- Identify concrete behaviors in the answer that signal usage or non-usage of internal materials (e.g., explicit references to past errors or phases, or reversion to generic textbook advice).

Step 3: Dimension Scoring
- Assign a 1–5 score for each dimension (Content, Pattern, Style) based on how well the answer matches the level descriptors.

Step 4: Global Aggregation
- Determine the “overall_memory_dependence_score” (1–5) consistent with the dimension scores.
- Content and Pattern should be weighted slightly higher than Style.

Step 5: Rationale Generation
- Provide a concise justification (about 5–10 sentences) citing specific textual evidence for the assigned scores.

------------------------------------------------------------
4. Dimensions
------------------------------------------------------------

4.1 Content Axis — Content-Level Memory Dependence

Definition:
- Measures the extent to which the answer’s substance—facts, examples, constraints, and recommendations—is grounded in internal project materials rather than generic domain knowledge.
- Assesses whether the core claims rely on specific artifacts (plans, results, feedback summaries) for their validity.

Diagnostic questions:
- Counterfactual Test:
  If specific project memory were removed, would the answer’s core claims remain valid, or would they lose their justification?
- Evidence Basis:
  Do the main arguments explicitly leverage internal facts (e.g., prior experiment outcomes, specific learner misconceptions) as premises?
- Artifact Reuse:
  Does the answer substantively incorporate content from internal artifacts (e.g., reusing specific plan phases or named research directions)?

Subdimensions:
- Anchoring Target:
  Differentiation between content organized around internal specificities (progress, failures, plans) versus generic definitions and standard methods.
- Specificity and Substitutability:
  Degree to which the answer is tailored to the specific project/learner.
  Low substitutability (i.e., the answer becomes invalid for a different user) indicates high dependence.
- Artifact and Summary Reuse:
  Active incorporation of concrete details from internal artifacts and cross-session summaries (e.g., specific results, named directions, error categories).

Score levels for Content Axis:

Level 1 — Externalized Content (Generic Reconstruction)
- Descriptor:
  The content is constructed primarily from generic domain knowledge.
  The project context serves merely as a broad topic prompt.
- Typical signals:
  - Arguments are framed in general terms (generic advice, standard definitions) with minimal project specificity.
  - Internal facts are mentioned only as superficial context or setting.
  - The response is highly substitutable; it would remain valid for a different project with similar high-level goals.

Level 2 — Lightly Contextualized (Ornamental Dependence)
- Descriptor:
  The answer draws on internal details for illustration or minor constraints, but the core substance remains generic.
- Typical signals:
  - Internal references (e.g., past experiments, errors) are used as examples or sanity checks, not as central premises.
  - Recommendations acknowledge high-level constraints (e.g., budget, topic) but remain largely standard.
  - Artifacts are loosely summarized rather than substantively integrated.

Level 3 — History-Aware (Integrated Dependence)
- Descriptor:
  Internal information meaningfully shapes the scope, prioritization, and selection of content.
  Generic knowledge is filtered through the project history.
- Typical signals:
  - Key sections are conditioned on specific history (e.g., prioritizing topics based on known bottlenecks).
  - Internal artifacts are used as concrete references (e.g., specific phases or failure modes).
  - Removing history would render key recommendations vague or unjustified.

Level 4 — History-Driven (Structural Dependence)
- Descriptor:
  The content’s backbone is defined by internal materials.
  Prior plans, results, and patterns dictate the substance.
- Typical signals:
  - Central arguments revolve around detailed internal items (e.g., specific experiment IDs, named failure clusters).
  - Recommendations are explicitly derived from past outcomes (e.g., “Since X failed, we must try Y”).
  - Heavy reuse of internal artifact components (e.g., adopting specific plan sections) as building blocks.

Level 5 — Continuation Mode (Deep Entrenchment)
- Descriptor:
  The content is a direct, seamless continuation of internal logs and materials.
  Meaning is opaque without access to specific memory items.
- Typical signals:
  - Points presuppose detailed knowledge of specific internal facts and patterns.
  - New material is built directly upon established structures (e.g., continuing a specific “Phase 3B”).
  - Generic knowledge is minimal, primarily used to connect internal details.

------------------------------------------------------------
4.2 Pattern Axis — Pattern-Level Memory Dependence
------------------------------------------------------------

Definition:
- Measures the extent to which the answer’s organization, decomposition strategies, and reasoning logic align with established internal routes and documented preferences, rather than adopting generic external frameworks.

Diagnostic questions:
- Process Isomorphism:
  Does the answer replicate a known internal workflow (e.g., a specific review cycle) or does it impose a standard external template?
- Reasoning Continuity:
  Are decision criteria inherited from past sessions (e.g., specific trade-offs, risk attitudes)?
- Branching Logic:
  Are alternatives presented as controlled deviations from the existing path (high dependence) or as generic options (low dependence)?

Subdimensions:
- Structural Isomorphism:
  Alignment of section order, step sequences, and decomposition logic with known internal routes or templates.
- Reasoning Strategy Continuity:
  Reuse of established decision-making heuristics and diagnostic patterns (e.g., consistent criteria for direction selection).
- Alternative Path Handling:
  Framing of new options as extensions or branches of the known path vs. independent generic choices.
- Cross-Session Process Reuse:
  Adoption of process skeletons from cross-session summaries (e.g., reusing a specific critique checklist) as the organizational backbone.

Score levels for Pattern Axis:

Level 1 — Generic Pattern (Standard Framework)
- Descriptor:
  Organization follows a standard external framework (e.g., textbook structure).
  No meaningful continuation of internal processes.
- Typical signals:
  - Structure mimics generic patterns (e.g., “Introduction–Analysis–Conclusion”).
  - Reasoning employs domain-general criteria without project-specific heuristics.
  - Alternatives are presented as generic options in a vacuum.

Level 2 — Loosely Echoing (Weak Alignment)
- Descriptor:
  Occasional echoes of internal patterns, but the overall organization remains generic.
- Typical signals:
  - Some steps resemble internal templates but lack systematic alignment.
  - Reasoning is a mix of generic criteria and isolated project references.
  - Cross-session processes are mentioned but do not structure the response.

Level 3 — Aligned Pattern (Hybrid Structure)
- Descriptor:
  The answer meaningfully integrates internal routes and preferences within a generally accessible structure.
- Typical signals:
  - Section order is derived from internal templates but enriched for clarity.
  - Key reasoning moves follow documented strategies (e.g., risk attitudes).
  - Alternatives are framed relative to the existing path.

Level 4 — Route-Following (Strong Alignment)
- Descriptor:
  The answer rigorously adheres to internal process templates and preferences.
  Generic frameworks are subordinate.
- Typical signals:
  - Structure matches known internal routes (e.g., specific plan phases).
  - Reasoning reuses established criteria and trade-off logic.
  - Cross-session execution summaries serve as the primary skeleton.

Level 5 — Process Continuation (Inseparable Flow)
- Descriptor:
  The answer functions as the next logical step in an idiosyncratic internal process.
- Typical signals:
  - Structure is unintelligible without knowledge of the specific route.
  - Reasoning is tightly bound to internal control loops.
  - Alternatives are treated as micro-adjustments within the established process.

------------------------------------------------------------
4.3 Style Axis — Style-Level Memory Dependence
------------------------------------------------------------

Definition:
- Measures the communicative stance of the answer:
  Does it speak as an “insider” leveraging shared context and shorthand, or as an “outsider” providing explicit explanations?

Diagnostic questions:
- Context Assumption:
  Does the answer presuppose shared background, or does it attempt to reconstruct the context explicitly?
- Terminology Continuity:
  Does it preserve internal naming schemes and narrative habits?
- Template Language:
  Is language from internal summaries and rubrics reused verbatim or near-verbatim?

Subdimensions:
- Context Internalization:
  Degree to which shared background is assumed (insider stance) vs. explained (external stance).
- Terminology Continuity:
  Preservation of project-specific shorthand, labels, and narrative conventions.
- Template Language Reuse:
  Verbatim or near-verbatim reuse of headings and phrasing from cross-session templates.

Score levels for Style Axis:

Level 1 — External Voice (Explanatory)
- Descriptor:
  Written for an informed external audience.
  Context is explicitly reconstructed; no insider shorthand is used.
- Typical signals:
  - Neutral, widely understandable terminology.
  - Tone of a standalone report or tutorial.
  - No verbatim reuse of internal template language.

Level 2 — Lightly Internalized (Transitional)
- Descriptor:
  Predominantly external voice with occasional internal references or shorthand.
- Typical signals:
  - Isolated internal terms appear but are often glossed or explained.
  - Tone is mostly generic with brief acknowledgments of context.
  - Minimal reuse of template language.

Level 3 — Mixed Voice (Collaborative)
- Descriptor:
  Alternates between insider collaboration and external explanation.
  Assumes some shared background but maintains accessibility.
- Typical signals:
  - Key internal terms are reused with light reminders.
  - Tone oscillates between coordination and exposition.
  - Recognizable reuse of standard template labels.

Level 4 — Internal Collaboration Voice (Insider)
- Descriptor:
  Written for internal coordination.
  Assumes detailed shared context and heavily uses project-specific discourse.
- Typical signals:
  - Extensive use of unexplained shorthand and specific labels.
  - Tone presumes a stable, shared mental model.
  - Extensive reuse of cross-session template language.

Level 5 — Log-Continuation Voice (Deep Insider)
- Descriptor:
  Written as a direct continuation of an internal log.
  Dense with implicit context and shared language.
- Typical signals:
  - Sentences rely heavily on specific references and shorthand.
  - No effort to reintroduce context.
  - Discourse is organized entirely around internal naming schemes.

------------------------------------------------------------
5. Joint Constraints Across Dimensions
------------------------------------------------------------

Common explanation source:
- All dimension scores must be grounded in the degree of adherence to internal history, patterns, and preferences.
- Scores must not be influenced by factors unrelated to path orientation (such as general answer quality alone).

Complementarity:
- Content, Pattern, and Style provide distinct but complementary views of dependence.
- Avoid redundant scoring; evaluate each axis based on its specific criteria.

Conditional observability:
- Treat unobservable sub-dimensions as N/A.
- Base scores only on available evidence.
- Do not penalize answers for missing memory structures or artifacts that were not provided.

Weighting heuristic:
- The overall_memory_dependence_score should be primarily driven by Content and Pattern.
- Style acts as a modifier; it should not shift the overall score by more than one level.
"""
