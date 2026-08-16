# Related Work Draft (English) — LLM → IRL → RL Evacuation

> For use as a draft of the paper's Related Work section. Citation keys refer to `paper/references.bib`.
> Please verify final versions, author lists, and page numbers before submission, especially entries marked TODO.

## 1. Crowd Evacuation Simulation: From Physical Models to LLM Agents

Conventional crowd evacuation simulation is dominated by physics-based models, most notably the social force model (SFM) \citep{helbing1995social}, which represents pedestrian motion as the superposition of several forces. These models are computationally efficient and support large-scale simulation, but their individual decision rules are simplistic and struggle to capture behavioral heterogeneity caused by age, personality, training background, and other factors. Cellular automata (CA) methods face the same "rules-as-behavior" bottleneck.

Large language models (LLMs) offer a new path for behavior modeling. Wu et al. proposed Smart Agent-Based Modeling (SABM), embedding LLMs as the decision-making cores of agents within traditional ABM and validating the approach in an emergency evacuation case study \citep{wu2023sabm}. Since then, LLM-driven evacuation simulation has grown rapidly: Dang et al. combined LLM agents with a CA fire environment, endowing agents with personalized memory, cognition, and decision-making abilities to simulate human decisions and behavior during fire emergencies \citep{dang2025llmfire}; Yang et al. embedded LLMs as the decision cores of individual agents in ABM and validated the framework in a real-world disaster evacuation case study \citep{yang2026think}; Calzolari et al. injected OCEAN personality traits into LLM agents and found that personality significantly affects both individual and collective evacuation outcomes \citep{calzolari2026personality}; Mendoza et al. proposed persona-driven agents with a three-level cognitive hierarchy (high-level goals, mid-level route reasoning, and low-level navigation), calibrated against empirical evacuation data \citep{mendoza2026hierarchical}. On scale and utility, Li et al. scaled LLM agent simulations to 13,000 agents for emergency-preparedness policy making \citep{li2025policy}; Sultimov et al. demonstrated that an LLM-guided commander agent with episodic memory improves evacuation coordination in wildfire scenarios \citep{sultimov2026llmguided} and further built RESPOND, a platform coupling natural-hazard forecasting with LLM-driven human behavior \citep{sultimov2026respond}.

These works demonstrate that LLMs can generate diverse, human-like evacuation behavior, but they generally stop at "driving behavior directly with LLMs": the decisions are opaque, difficult to optimize formally, and often lack rigorous validation of behavioral realism. Lee et al. evaluated the predictability of LLM-based evacuation simulations against real-world mobility data \citep{lee2025predictability}, and Larooij and Törnberg's critical review argues that generative ABM research rarely moves beyond subjective "believability" toward operational validation \citep{larooij2025critical}. Translating the behavioral wisdom of LLMs into an interpretable, optimizable, and verifiable decision system therefore remains a key gap in the field.

## 2. Learning Rewards from Demonstrations: IRL and LLMs

Inverse reinforcement learning (IRL) aims to recover the implicit reward function behind expert behavior. Ziebart et al. introduced Maximum Entropy IRL (MaxEnt IRL), which assumes that behavior distributions are maximally entropic subject to feature-matching constraints; it remains a canonical framework for diverse human behavior \citep{ziebart2008maxent}. Recent work combining IRL and LLMs falls into two main lines.

First, **using LLMs to improve the interpretability and sample efficiency of IRL**. GRACE uses LLM-driven evolutionary search to reverse-engineer executable, code-form reward functions from expert trajectories \citep{sapora2025grace}. Masked IRL leverages LLMs to infer state-relevance masks from language instructions, mitigating spurious correlations and instruction ambiguity in reward learning \citep{hwang2026maskedirl}.

Second, **recovering dense rewards from LLM/expert behavior and using them for downstream policy optimization**. Scherer et al. learn a dense reward from expert demonstrations via coherent imitation learning to improve large behavior-model policies \citep{scherer2026csil}. Li et al. prove that supervised fine-tuning (SFT) is equivalent to an implicit IRL reward-recovery process and propose Dense-Path REINFORCE \citep{li2025beyond}. Fanconi et al. propose R-AIRL, which recovers process-level rewards from expert chain-of-thought and uses them for training and inference-time reranking \citep{fanconi2025rairl}. Li et al. further prove that a generalist reward model already exists inside any LLM trained with standard next-token prediction, equivalent to offline IRL \citep{li2025generalist}. On the interpretability side, IR$^3$ reconstructs the implicit objectives of RLHF-tuned models via contrastive IRL and decomposes them into interpretable features \citep{beigi2026ir3}.

These works provide the methodological foundation for our approach: learning rewards from LLM-generated demonstrations is theoretically grounded. However, they mostly target robot control, LLM alignment, or general decision tasks, and none forms a complete chain of "LLM behavior → IRL value weights → downstream scheduling policy" in the crowd-evacuation setting.

## 3. Distilling LLM Behavior into Downstream Policies: Same-Track Methods and Our Differences

The closest works to ours fall into three categories.

**(1) Behavior distillation (BC).** These methods train a classifier or policy directly on (state, action) samples from LLMs. Zhang et al. propose DFD, which distills rule-based, interpretable decision functions from LLM-driven evacuation crowds and shows they outperform both classical methods and a state-of-the-art LLM-based symbolic-regression baseline \citep{zhang2026star}. BC methods essentially imitate the surface of LLM behavior: they cannot surpass the teacher policy and generalize poorly to out-of-distribution states.

**(2) LLM direct reward design.** These methods prompt LLMs to state their preferences over objectives directly as reward weights. They rely on stated preferences, which psychology research shows can deviate systematically from the revealed preferences reflected in actual behavior, as discussed in the introduction.

**(3) Parallel LLM+RL / prediction frameworks.** FLARE combines behavioral theory, LLM reasoning, and memory-based RL to predict real human evacuation decisions during wildfires \citep{chen2025flare}. RESPOND drives population behavior with LLM agents on a disaster-simulation platform \citep{sultimov2026respond}. These works either target prediction rather than policy optimization or let LLMs make decisions directly without forming an interpretable distillation.

**(4) Safety constraints for LLMs.** A related line of work addresses the safe and reliable deployment of LLMs in safety-critical decisions. Constitutional AI constrains the self-improvement and alignment process of LLMs with an explicit list of principles \citep{bai2022constitutional}. SayCan grounds LLM proposals in the value functions of low-level skills, preventing suggestions that are infeasible for the given embodiment \citep{ahn2022saycan}. KnowNo calibrates the uncertainty of LLM-based planners with conformal prediction, deciding between execution and help-seeking under statistical guarantees \citep{ren2023knowno}. Unlike these training-time or planning-time constraints, our safety guard is an inference-time symbolic layer of hard constraints: the LLM acts as an advisor, while seven verifiable physical and physiological thresholds arbitrate every decision in real time and produce an auditable intervention log. This provides interpretable, verifiable safety guarantees in evacuation scenarios.

The essential differences between our three-tier LLM → IRL → RL cascade and the above works are summarized below.

| Dimension | BC distillation (e.g., DFD) | LLM direct reward | Parallel LLM+RL (e.g., FLARE) | Ours |
|-----------|----------------------------|-------------------|-------------------------------|------|
| What is transferred from LLM behavior | State→action mapping | Stated weights | Advice / prediction signal | Revealed value preferences (via IRL) |
| Theoretical basis | Supervised learning | Prompt self-report | Parallel fusion | MaxEnt IRL feature matching |
| Downstream artifact | Black-box / rule policy | Hand-crafted reward | Predictive model / advice | Interpretable weights → zone scheduling policy |
| Closed loop | No | No | One-way | RL advice injected back into LLM prompts in natural language |
| Auditability | Weak | Weak | Medium | Weight visualization + guardrail audit |

In summary, no existing work to our knowledge implements the complete pipeline of "LLM evacuation behavior → MaxEnt IRL value weights → zone-level RL scheduling → natural-language feedback into LLM decisions." Our paper fills this gap; the ablation study and same-track baseline comparisons (BC distillation, LLM direct reward, heuristic RL) further demonstrate the necessity of IRL as the distillation bridge.

## 4. Remaining To-Dos

- [ ] Verify the complete author lists of `hwang2026maskedirl` and `li2025generalist` (currently marked `others` in the BibTeX).
- [ ] Add full bibliographic details for the Chinese paper 《基于大语言模型智能体的多主体协同应急仿真系统研究》(CQVIP/CNKI).
- [x] Safety-constraint references (Constitutional AI, SayCan, KnowNo) added; see Section 3(4).
- [ ] After the main experiments are complete, tie the "lack of rigorous validation" argument in this section to the empirical results in the paper's experiment section.
