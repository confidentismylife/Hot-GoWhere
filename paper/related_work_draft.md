# Related Work 草稿（LLM → IRL → RL 疏散仿真）

> 用途：作为论文 `Related Work` 章节的初稿。引用键对应 `paper/references.bib`。
> 请在使用前核实每条引用的最终版本、作者列表与页码，尤其是标注了 TODO 的条目。

## 1. 人群疏散仿真：从物理模型到 LLM 智能体

传统人群疏散仿真以物理模型为主，最具代表性的是社会力模型（social force model, SFM）\citep{helbing1995social}，其将行人运动建模为若干力的叠加，计算高效、便于大规模仿真，但个体决策规则单一，难以刻画年龄、性格、培训背景等异质性导致的行为差异。元胞自动机（cellular automata, CA）方法同样面临"规则即行为"的表达瓶颈。

大语言模型（LLM）的兴起为行为建模提供了新路径。Wu 等人提出 Smart Agent-Based Modeling（SABM）框架，将 LLM 作为智能体的决策核心嵌入传统 ABM，并首次在紧急疏散案例中验证了可行性\citep{wu2023sabm}。此后，LLM 驱动的疏散仿真迅速成为热点：Dang 等人将 LLM 智能体与 CA 火灾环境结合，赋予智能体个性化记忆、认知与决策能力，模拟火灾疏散中的人类决策与行为\citep{dang2025llmfire}；Yang 等人将 LLM 作为个体决策核心嵌入 ABM，并在真实灾害疏散案例中验证其真实性、适应性与可靠性\citep{yang2026think}；Calzolari 等人将 OCEAN 人格特质注入 LLM 智能体，发现人格显著影响个体与群体疏散结果\citep{calzolari2026personality}；Mendoza 等人提出三层认知层级（高层目标、中层路径推理、底层导航）的人格化智能体框架，并用真实疏散数据进行校准\citep{mendoza2026hierarchical}。在规模与实用性方面，Li 等人将 LLM 智能体仿真扩展到 13,000 个智能体，用于大型集会应急预案制定\citep{li2025policy}；Sultimov 等人则在野火场景中验证了带记忆的 LLM 调度智能体对疏散协调的有效性\citep{sultimov2026llmguided}，并进一步构建了耦合自然灾害预测与人类行为的 RESPOND 平台\citep{sultimov2026respond}。

上述工作证明了"LLM 可以生成多样化、类人的疏散行为"，但普遍停留在"用 LLM 直接驱动行为"这一层：决策过程不可解释、难以形式化优化，且缺少对 LLM 行为真实性的严格验证。Lee 等人用真实移动数据评估 LLM 疏散模拟的可预测性\citep{lee2025predictability}，Larooij 与 Törnberg 的批判性综述更指出，生成式 ABM 普遍缺乏对"可信度"之外的操作性验证\citep{larooij2025critical}。因此，如何把 LLM 的行为智慧转化为可解释、可优化、可验证的决策系统，成为本领域的关键缺口。

## 2. 从示范行为中学习奖励：IRL 与 LLM

逆强化学习（IRL）旨在从专家行为中恢复隐含的奖励函数。Ziebart 等人提出的最大熵 IRL（MaxEnt IRL）以"行为分布满足特征匹配约束下的最大熵"为核心假设，是处理多样化人类行为的经典框架\citep{ziebart2008maxent}。近年来，IRL 与 LLM 的结合出现两条主线：

第一，**用 LLM 改进 IRL 的可解释性与样本效率**。GRACE 用 LLM 驱动进化搜索，从专家轨迹中反演出可执行的代码形式奖励函数\citep{sapora2025grace}；Masked IRL 利用 LLM 从语言指令推断状态相关性掩码，缓解奖励学习中的伪相关与指令歧义问题\citep{hwang2026maskedirl}。

第二，**从 LLM/专家行为中恢复稠密奖励并用于下游策略优化**。Scherer 等人通过相干模仿学习（coherent imitation learning）从专家演示中学得稠密奖励，用于提升大规模行为模型策略\citep{scherer2026csil}；Li 等人证明监督微调（SFT）等价于一种隐式 IRL 奖励恢复，并据此提出 Dense-Path REINFORCE\citep{li2025beyond}；Fanconi 等人提出 R-AIRL，从专家思维链中恢复过程级奖励，用于推理模型的训练与推理期重排\citep{fanconi2025rairl}；Li 等人进一步从理论上证明标准下一词预测训练的 LLM 内部已蕴含一个等价于离线 IRL 的内生奖励模型\citep{li2025generalist}。在可解释性方向上，IR$^3$ 用对比 IRL 重构 RLHF 隐式目标并分解为可解释特征\citep{beigi2026ir3}。

这些工作为本项目提供了方法论依据：从 LLM 生成的示范行为中学习奖励是有理论根基的。但它们大多面向机器人控制、LLM 对齐或通用决策任务，尚未在人群疏散场景中形成"LLM 行为 → IRL 价值权重 → 下游调度策略"的完整链路。

## 3. LLM 行为蒸馏到下游策略：同赛道方法与本文差异

与本文最接近的工作可分为三类：

**（1）行为蒸馏（BC）类**。直接以 LLM 的（状态，动作）样本训练分类器/策略。Zhang 等人提出的 DFD 方法从 LLM 驱动的疏散人群中蒸馏出规则形式的可解释决策函数，并验证其优于经典方法与 LLM 符号回归基线\citep{zhang2026star}。BC 类方法本质上是对 LLM 行为的表面模仿，难以超越教师策略，且对分布外状态泛化能力有限。

**（2）LLM 直接奖励设计类**。通过 Prompt 让 LLM 直接陈述对各目标的重视程度作为奖励权重。这类方法依赖"陈述偏好"（stated preference），而心理学研究表明其与实际行为中体现的"显示偏好"（revealed preference）存在系统性偏差，本文引言已展开论述。

**（3）LLM+RL 并联/预测类**。FLARE 将行为理论、LLM 推理与记忆增强 RL 结合，用于预测真实人群的野火疏散决策\citep{chen2025flare}；RESPOND 在灾害仿真平台上用 LLM 驱动人群行为\citep{sultimov2026respond}。这些工作或面向预测而非策略优化，或仍由 LLM 直接承担决策，未形成可解释的策略蒸馏。

**（4）LLM 的安全约束类工作**。让 LLM 安全可靠地参与安全关键决策是另一个相关方向。Constitutional AI 通过一组显式原则约束 LLM 的自我改进与对齐过程\citep{bai2022constitutional}；SayCan 用底层技能的价值函数约束 LLM 提出"可行且合语境"的动作，避免模型输出脱离实际能力的建议\citep{ahn2022saycan}；KnowNo 基于保形预测对 LLM 规划器的不确定性进行校准，在统计保证下决定"执行还是求助"\citep{ren2023knowno}。与这些训练期或规划期的约束不同，本文的安全护栏是推理时的符号化硬约束：LLM 仅作为建议者，7 条可由领域专家验证的物理/生理阈值对每条决策进行实时仲裁，且拦截过程可审计——在安全关键疏散场景中提供了可解释、可验证的安全保障。

本文提出的 LLM → IRL → RL 三级联级与上述工作的本质区别在于：

| 维度 | BC 蒸馏（DFD 等） | LLM 直接奖励 | LLM+RL 并联（FLARE 等） | 本文 |
|------|------------------|--------------|------------------------|------|
| 从 LLM 行为中迁移什么 | 状态→动作映射 | 陈述性权重 | 决策建议/预测信号 | 显示性价值偏好（IRL 恢复） |
| 理论依据 | 监督学习 | Prompt 自述 | 并行融合 | MaxEnt IRL 特征匹配 |
| 下游产物 | 黑盒/规则策略 | 人工设计奖励 | 预测模型/决策建议 | 可解释权重 → 区域调度策略 |
| 闭环 | 无 | 无 | 单向 | RL 建议以自然语言回注 LLM Prompt |
| 可审计性 | 弱 | 弱 | 中 | 权重可视化 + 护栏审计 |

综上，现有文献中尚未出现"从 LLM 疏散行为出发，经 MaxEnt IRL 恢复多角色价值权重，再训练区域级 RL 调度策略并以自然语言反馈注入 LLM 决策闭环"的完整方案。本文填补的正是这一缺口；消融实验与同赛道基线对比（BC 蒸馏、LLM 直接奖励、启发式 RL）进一步论证了 IRL 作为蒸馏桥梁的必要性。

## 4. 待补充与核对事项

- [ ] 核实 `hwang2026maskedirl`、`li2025generalist` 的完整作者列表（BibTeX 中暂以 `others` 占位）。
- [ ] 补充中文文献《基于大语言模型智能体的多主体协同应急仿真系统研究》的完整出处（维普/CNKI）。
- [x] 已补充安全约束类文献（Constitutional AI、SayCan、KnowNo），见 §3 第 (4) 点。
- [ ] 主实验数据跑通后，把 Related Work 中"现有方法缺乏验证"的论述与本文第 X 节的实证结果呼应起来。
