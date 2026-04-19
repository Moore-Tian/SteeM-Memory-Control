

ALL_PERSONAS = {
  "research": {
    "CS-AI/ML": [
      "A machine learning researcher focused on improving the efficiency and scalability of large models. They are interested in optimization strategies, training stability, and resource-aware algorithms, and often think about how design choices affect real-world deployment constraints.",
      "An applied AI scientist working on alignment and robustness of LLM-based systems. They care about evaluation protocols, reward modeling, and how agent behavior changes under different training regimes and memory settings in practical applications.",
      "A theoretical-minded ML researcher interested in generalization, optimization landscapes, and overparameterized models. They tend to question assumptions behind empirical methods and look for principled explanations that connect theory to observed behavior.",
      "A practitioner building multi-agent or memory-augmented systems who is concerned with long-horizon reasoning, coordination, and failure modes in complex environments. They routinely analyze emergent behaviors and edge cases in agent interactions.",
      "A systems-aware ML engineer who bridges modeling and infrastructure, focusing on distributed training reliability, hyperparameter tuning at scale, and performance trade-offs. They frequently collaborate with platform teams to make large models production-ready."
    ],

    "CS-NLP": [
      "An NLP researcher focused on evaluation and benchmarking of language models, especially for long-context reasoning, summarization, and question answering tasks. They design datasets and metrics to stress-test model limits.",
      "A practitioner working on retrieval-augmented and knowledge-grounded NLP systems, interested in robustness, factuality, and error analysis. They care about how retrieval choices and knowledge sources affect downstream behavior.",
      "A researcher studying dialogue systems and conversational agents, with particular interest in personalization, safety, and memory-sensitive behaviors. They monitor how long-term context changes user experience and risk.",
      "An applied NLP scientist working on multilingual or low-resource settings, concerned with data augmentation, cross-lingual transfer, and fairness across languages. They frequently evaluate models across diverse linguistic and cultural settings."
    ],

    "CS-CV": [
      "A computer vision researcher focused on robustness and generalization, particularly under domain shift, weak supervision, or adversarial conditions. They design challenging benchmarks and stress tests for visual models.",
      "An applied CV practitioner working on real-time or edge deployment, balancing accuracy, latency, and resource constraints. They routinely optimize architectures and pipelines for limited compute and tight SLAs.",
      "A researcher interested in vision-language models and multimodal reasoning, exploring how visual and textual representations align. They often probe cross-modal failures and ambiguous cases.",
      "A scientist working on 3D understanding or video analysis, thinking about temporal structure and spatial consistency. They care about long-term tracking, motion patterns, and stability of predictions over time."
    ],

    "CS-HCI": [
      "An HCI researcher studying how people interact with AI systems, with a focus on trust, usability, and evaluation methodologies. They conduct user studies and analyze how interface design shapes mental models.",
      "A designer or researcher interested in adaptive interfaces and accessibility, aiming to improve inclusiveness and user experience. They think carefully about diverse user needs and edge-case scenarios.",
      "An applied scientist analyzing behavioral logs to understand user patterns, feedback loops, and system misalignment. They use quantitative and qualitative signals to refine AI-driven products over time."
    ],

    "CS-Systems": [
      "A systems researcher optimizing distributed training and large-scale infrastructure, focusing on reliability and fault tolerance. They design experiments to measure throughput, resilience, and cost under realistic loads.",
      "An engineer working on efficient serving and deployment of foundation models, concerned with latency and cost. They tune batching, caching, and model variants to meet production SLAs.",
      "A researcher studying pipeline orchestration and resource scheduling in cloud and edge environments. They explore how to coordinate data, compute, and workloads across heterogeneous clusters."
    ],

    "CS-Security": [
      "A security researcher focused on privacy-preserving ML and federated learning, interested in threat models and auditability. They analyze how data, models, and protocols can leak or protect sensitive information.",
      "An applied security engineer working on adversarial robustness and jailbreak prevention for LLM systems. They design red-teaming workflows and safeguards against misuse.",
      "A cryptography-aware ML researcher exploring secure computation and data exchange protocols. They think about how to combine cryptographic guarantees with practical learning systems."
    ],

    "Mathematics": [
      "A mathematically inclined researcher interested in optimization theory and constrained systems, often connecting theory to computational practice. They derive bounds, convergence guarantees, and conditions for algorithms used in ML.",
      "A scientist using probabilistic and graph-theoretic models to analyze complex systems. They care about structure, randomness, and how network properties influence behavior.",
      "A researcher working on symbolic or automated reasoning, curious about formal structure and abstraction. They design logics, proof systems, or tools for mechanized reasoning."
    ],

    "Physics": [
      "A computational physicist working on simulation pipelines, concerned with numerical stability, discretization choices, and model fidelity. They routinely balance accuracy against compute limitations.",
      "An experimental physicist designing and calibrating instrumentation, interested in data interpretation workflows. They think carefully about noise, measurement error, and reproducible analysis.",
      "A researcher modeling complex physical systems such as turbulence or plasma, balancing theory and computation. They iteratively refine models based on both simulation and experimental evidence."
    ],

    "Chemistry": [
      "A computational chemist using ML to predict material or molecular properties, interested in model accuracy and interpretability. They compare predictions against simulations and experimental data.",
      "A researcher focused on reaction mechanisms and synthesis planning, combining domain knowledge with algorithmic tools. They design workflows that suggest plausible reaction paths and conditions.",
      "An applied scientist analyzing spectroscopy or electrochemical data, concerned with signal interpretation. They build pipelines to denoise, fit, and explain complex experimental measurements."
    ],

    "Biology": [
      "A computational biologist analyzing protein structures and functional annotations, interested in evaluation metrics and biological relevance. They regularly compare algorithms on realistic biological benchmarks.",
      "A systems biologist modeling regulatory or metabolic networks, focused on inference robustness. They integrate multi-omics data and assess how perturbations propagate through networks.",
      "An experimental biologist designing protocols and workflows, concerned with reproducibility and interpretation. They want analysis tools that respect practical lab constraints and noise."
    ],

    "Engineering": [
      "An engineer using simulation and modeling to assess system safety and reliability. They study failure modes, stress conditions, and regulatory requirements when evaluating designs.",
      "A robotics researcher focused on perception, motion planning, and evaluation under uncertainty. They run experiments in both simulation and real-world environments to probe robustness.",
      "An applied engineer optimizing sensing and control pipelines in real-world systems. They care about latency, fault tolerance, and maintainability of deployed solutions."
    ],

    "Medical": [
      "A medical data scientist analyzing clinical or imaging data, focused on evaluation and reliability. They examine metrics, calibration, and bias before deploying decision-support tools.",
      "A researcher working on disease modeling and outbreak forecasting, concerned with uncertainty and policy impact. They simulate scenarios and communicate limitations to stakeholders.",
      "An applied scientist integrating ML tools into healthcare workflows. They think about usability, safety, and integration with existing clinical processes."
    ],

    "Economics": [
      "An economist using causal inference and modeling to evaluate policy interventions. They are careful about identification strategies and robustness checks.",
      "A researcher analyzing market behavior and trade-offs using empirical data. They build models to interpret price dynamics, incentives, and strategic interactions.",
      "An applied economist focused on communicating uncertainty and assumptions clearly. They prepare reports and briefings for non-technical decision-makers."
    ],

    "PoliticalScience": [
      "A political scientist studying text-based analysis of ideology and discourse. They construct corpora, annotate concepts, and build models to track framing over time.",
      "A researcher focused on survey design, forecasting, and policy evaluation. They care about sampling, measurement error, and how public opinion changes.",
      "An analyst interested in how institutions and incentives shape political outcomes. They synthesize qualitative and quantitative evidence for comparative studies."
    ],

    "Psychology": [
      "A cognitive scientist designing experiments to study behavior and decision making. They carefully plan tasks, measurements, and analyses to test specific hypotheses.",
      "A psychologist working on longitudinal or behavioral modeling studies. They track participants over time and analyze trajectories of change.",
      "A researcher interested in measurement validity and psychometric scale construction. They refine questionnaires, factor structures, and reliability estimates."
    ],

    "Humanities": [
      "A digital humanities scholar analyzing historical texts and archival data. They combine computational methods with close reading to study patterns and narratives.",
      "A researcher studying thematic or linguistic evolution across corpora. They examine how concepts, styles, or genres shift over time.",
      "A historian modeling long-term trends using structured data. They link quantitative indicators to qualitative historical accounts."
    ]
  },

  "tutoring": {
    "Math": [
      "A student who understands basic procedures but struggles with translating word problems into mathematical expressions. They want help turning natural language into equations and inequalities.",
      "A learner preparing for exams who wants to improve multi-step problem-solving and systematic error checking. They are looking for step-by-step strategies and common pitfall warnings.",
      "A student interested in building intuition behind formulas, units, and scaling rather than relying on rote memorization. They appreciate visual explanations and real-world examples.",
      "A learner who often makes small logical or arithmetic mistakes and wants more structured solution habits and sanity checks. They are motivated to develop checklists and verification routines."
    ],

    "Programming": [
      "A beginner programmer learning Python who needs help understanding basic syntax, control flow, and how to interpret and fix error messages. They often ask for simple examples and guided debugging.",
      "A self-taught learner aiming to write cleaner, more maintainable code, with better structure, naming, and documentation. They care about code that others can read and extend.",
      "A student who knows basic Python and wants to practice core algorithmic patterns such as searching, sorting, and aggregation, and to develop systematic problem-solving strategies.",
      "A learner interested in code review and understanding trade-offs in design choices, including readability, performance, and extensibility. They enjoy comparing alternative implementations."
    ],

    "DataScience": [
      "A learner transitioning from data collection to analysis, struggling with evaluation, overfitting, and choosing appropriate baselines. They want guidance on what to measure and how to interpret results.",
      "A student focused on interpreting results and communicating findings clearly, including limitations and assumptions. They want help structuring reports and narratives around their analysis.",
      "A beginner data analyst learning how to structure experiments, split data correctly, and avoid leakage. They seek concrete patterns and checklists for good practice."
    ],

    "LifeSkills": [
      "A student managing multiple responsibilities and trying to build better planning and prioritization habits. They want concrete routines, schedules, and ways to adjust when plans slip.",
      "A learner interested in decision-making frameworks for everyday choices and trade-offs. They like simple mental models and step-by-step approaches to hard decisions.",
      "Someone working on consistency, motivation, and reflective improvement in their routines. They want to track progress, review setbacks, and gradually adjust habits."
    ],

    "Health": [
      "A learner trying to build sustainable routines around sleep, exercise, and stress management. They want realistic plans that fit into a busy schedule.",
      "Someone seeking to better interpret health information and make informed decisions from different sources. They are cautious about misinformation and conflicting advice.",
      "A student aiming to balance wellbeing with demanding academic or work schedules. They want strategies to handle peak stress periods without burning out."
    ],

    "Writing": [
      "A student working to improve clarity and coherence in academic writing. They need support with paragraph flow, topic sentences, and transitions.",
      "A learner revising drafts and learning how to respond to feedback constructively. They want help turning comments into concrete revision plans.",
      "Someone focused on professional tone and structure in reports, essays, and short documents. They care about sounding confident, precise, and appropriate for the audience."
    ],

    "Physics": [
      "A student who understands formulas but struggles with conceptual explanations and verbal reasoning in physics word problems. They want help connecting equations to physical intuition.",
      "A learner practicing multi-step reasoning, unit consistency, and sanity checks in physics calculations. They appreciate worked examples that highlight common mistakes."
    ],

    "Chemistry": [
      "A student learning to explain reaction mechanisms clearly rather than memorizing steps. They want to understand why reactions proceed in certain directions.",
      "A learner practicing stoichiometry and equilibrium reasoning through text-based problems and explanations. They seek guidance on setting up balanced equations and interpreting shifts."
    ],

    "History": [
      "A student learning to structure historical explanations using cause-and-effect reasoning. They want help linking events, actors, and consequences coherently.",
      "A learner practicing comparative historical arguments across periods, themes, or regions. They need support choosing relevant evidence and framing comparisons."
    ],

    "PoliticalScience": [
      "A student learning to analyze political arguments and policy trade-offs in a structured way. They want to break complex debates into clear claims and evidence.",
      "A learner practicing evidence-based political writing with clear assumptions and reasoning. They care about fairness, nuance, and acknowledging uncertainty."
    ],

    "Economics": [
      "A student learning to reason about incentives and trade-offs in everyday and policy scenarios. They want intuitive examples that link theory to real life.",
      "A learner practicing short, assumption-aware economic explanations for qualitative questions. They need help explicitly stating models, constraints, and caveats."
    ],

    "Biology": [
      "A student learning to explain biological systems and experimental reasoning clearly and step by step. They want to connect mechanisms to observable outcomes.",
      "A learner practicing structured biological explanations in text form, including diagrams translated into words. They care about precise terminology and logical flow."
    ]
  }
}