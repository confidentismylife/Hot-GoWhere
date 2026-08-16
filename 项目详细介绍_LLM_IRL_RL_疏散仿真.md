# 基于 LLM→IRL→RL 三级联级的人群疏散多智能体仿真系统

**项目详细介绍 · 2026-08-05 生成 · 对应仓库版本 v2.1（分支 v2.0-multimodal-diffusion）**

> 说明：本文档是项目的中文综合介绍，用于存档、组会汇报与论文写作参考。文中所有实验数字均标注来源；未经真实 LLM 实验验证的指标明确标注“未验证/待补”，请勿直接引用到论文正文。

---

## 1. 项目定位

这是一个面向单张 RTX 4090（24GB）消费级 GPU 的 **LLM 驱动的多角色智能体灾害人群疏散仿真系统**。系统用大语言模型（LLM）生成多样化的人类疏散决策，用逆强化学习（IRL）从行为中恢复人类价值权重，再用多智能体强化学习（RL）做区域级出口调度，形成“LLM → IRL → RL”三级联级的知识迁移闭环。

一句话概括：**让 LLM 教 RL 什么是好的疏散决策，而不是简单地把两个技术拼在一起。**

## 2. 核心创新：LLM → IRL → RL 三级联级

传统“LLM + RL”融合方案通常是把两者并行加权，审稿人可以批评为“两个已有技术的组合”。本项目的关键洞察是三级递进的知识蒸馏：

1. **LLM 行为数据生成**：600 个智能体（5 种角色、5 种人设）在商场/地铁站等场景中由 LLM 驱动疏散决策（出口选择、速度、协作模式），生成大量行为轨迹。
2. **MaxEnt IRL 价值权重恢复**：从 LLM 轨迹中恢复 5 维奖励权重（安全、效率、社交、从众、舒适）× 5 种人设，输出可解释的价值向量。
3. **分区独立 PPO（Zone-PPO） 区域调度**：把场景划分为 4 个区域，每个区域一个 MLP 调度器，用 IRL 学到的权重构造奖励，训练出口偏好策略；调度建议再以中文自然语言注入 LLM Prompt，保留 LLM 的最终决策权。

与主流方案的本质区别：IRL 作为理论桥梁，使 RL 的奖励函数不再是人工设计，而是从 LLM 行为中学到的“显示性偏好”；RL 建议回注 LLM 形成闭环，且权重可可视化、拦截可审计。

## 3. 系统总体架构

### 3.1 三层认知架构

- **感知层（perception/）**：CA 元胞自动机灾害仿真（火/震/洪，0.5m 网格、4 通道）、VLM 视觉感知（Qwen-VL-7B-AWQ，可选）、YOLO 行人检测（可选）、数值→中文描述转换器。
- **决策层（decision/）**：vLLM 批量推理引擎（Qwen2.5-3B-AWQ / 7B-AWQ）、15 套角色×灾害 Prompt 模板、7 条安全护栏、ChromaDB+BGE 中文逃生知识库 RAG。
- **执行层（execution/）**：主仿真循环、Numba 批量社会力物理引擎、IRL 轨迹采集与权重恢复、分区独立 PPO（Zone-PPO） 区域调度、奖励权重分析。

### 3.2 数据流（每个 tick）

灾害演进 → 环境快照 → RL 区域调度推理 → LLM 决策提交（含 RL 建议）→ 安全护栏仲裁 → 应用决策与轨迹记录 → VLM/YOLO 感知（可选）→ 群体智能（信息传播/恐惧/体力）→ 批量物理引擎 → 统计与可视化。

## 4. 代码模块清单

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 入口 | main.py | CLI 参数解析、模式选择（本地/Web/Gradio/录帧） |
| 感知层 | perception/environment.py | CA 灾害仿真器（火/震/洪） |
| 感知层 | perception/nl_converter.py | 数值网格 → 中文场景描述（含 RL 建议注入） |
| 感知层 | perception/vlm_perceiver.py | Qwen-VL-7B-AWQ 视觉语义（支持 Mock） |
| 感知层 | perception/yolo_detector.py | YOLOv8n 行人检测与密度热点 |
| 感知层 | perception/floorplan.py | 商场/场馆平面图加载（武汉保利广场等） |
| 决策层 | decision/cognitive_engine.py | vLLM 批量推理、Prefix Caching、异步流水线 |
| 决策层 | decision/prompt_manager.py | 5 角色 × 3 灾害 Prompt 模板与 JSON 解析 |
| 决策层 | decision/safety_guard.py | 7 条硬约束安全护栏（双阶段验证） |
| 决策层 | decision/knowledge_base.py | ChromaDB + BGE 知识库 RAG、角色路由 |
| 决策层 | decision/agent_state.py、agent_roles.py | Agent 数据模型与 5 角色体系 |
| 执行层 | execution/orchestrator.py | 主仿真循环（约 1258 行） |
| 执行层 | execution/batched_physics.py | Numba JIT 批量社会力模型（800 人约 3ms） |
| 执行层 | execution/irl_recovery.py | TrajectoryCollector + MaxEnt IRL（GC 组约束） |
| 执行层 | execution/rl_scheduler.py | 分区独立 PPO（Zone-PPO） 区域调度、启发式回退、中文建议生成 |
| 执行层 | execution/reward_analysis.py | 权重雷达图/热力图/KL 散度报告 |
| 执行层 | execution/diffusion_policy.py、diffusion_trainer.py | 扩散模型轨迹生成（实验性，默认关闭） |
| 群体智能 | group_intel/propagation.py | 信息传播、恐惧传染、体力、记忆 |
| 训练 | training/ | QLoRA 数据生成/微调/三模型对比 |
| 实验 | experiments/ | 主实验、消融、KS 验证、压力测试、基线对比 |
| 可视化 | visualization/ | Pygame / Matplotlib 录帧 / Flask Web / Gradio |
| 测试 | tests/ | 物理、安全护栏、IRL、RL、VLM 集成等 |

## 5. 关键算法与工程优化

### 5.1 MaxEnt IRL

- 5 维奖励特征：安全、效率、社交、从众、舒适。
- 5 种人设：未培训老人、未培训年轻人、已培训店员、引导员、消防员。
- 离散化特征 → 软值迭代 + 梯度下降，最小化专家特征期望与策略特征期望之差。
- GC-MaxEnt 扩展：跨人设拉普拉斯正则化，共享统计强度，防止小样本过拟合。

### 5.2 分区独立 PPO（Zone-PPO） 区域调度

- 4 区域 × 双头 MLP（策略头输出各出口偏好 [-1,+1]，价值头输出状态值）。
- PPO + GAE（γ=0.99, λ=0.95）离线训练，训练用快速规则模拟器加速。
- 推理 <1ms/tick；未训练时用 IRL 权重驱动的启发式回退。

> 实现说明：当前实现为 MAPPO 风格的 CTDE —— 各分区独立策略头（分散执行）+
> 共享集中式价值网络（仅训练时使用）；联合策略为因子化高斯策略。
> 早期文档中的“P-MAPPO”统一更名为“分区独立 PPO（Zone-PPO）”。

### 5.3 安全护栏（7 条硬约束）

出口烟雾封锁、路径火焰阻挡、体力-速度匹配、受伤限制、站位着火、出口距离合理性、危险中等待拦截。LLM 是建议者，护栏是仲裁者，每条拦截可审计。

### 5.4 单卡工程优化

AWQ 4-bit 量化、vLLM Prefix Caching、异步流水线、知识库缓存、O(1) Agent 查找、批量 Numba 物理引擎。README 声称 800 Agent 约 120ms/tick（该性能数字未在本次复核中重测）。

## 6. 运行与部署

### 6.1 本地/云端运行

```bash
# 安装依赖
pip install -r requirements.txt

# 基础运行（本地渲染）
python main.py --agents 200 --duration 120

# 无头录帧（服务器）
python main.py --no-viz --record --agents 500

# 云端 Web 监控
python main.py --config config/default.yaml --web --port 8080

# Gradio 交互仪表板
python main.py --gradio --gradio-port 8081

# 三级联级实验：采集轨迹 → IRL → RL
python main.py --config config/mall_floorplan.yaml --no-viz --agents 200
python -m execution.irl_recovery --trajectory_dir data/trajectories --output data/irl_weights.json
python -m execution.reward_analysis --weights data/irl_weights.json --output data/analysis/
python -m execution.rl_scheduler --mode init --irl_weights data/irl_weights.json --output data/rl_policy.json
```

云平台部署脚本：`setup_cloud.sh`（支持 AutoDL、恒源云、Vast.ai、矩池云等），`setup_and_run.sh` 为 7B 模型验证脚本。

### 6.2 主要配置文件

| 配置 | 场景 | 要点 |
| --- | --- | --- |
| config/default.yaml | 地铁站 800 人 | 3B-AWQ 模型、4 出口 |
| config/mall_floorplan.yaml | 武汉保利广场 600 人 | 7B-AWQ、6 出口、IRL/RL 开关 |
| config/mall_floorplan_hard.yaml | 商场困难版（未提交，待确认） | 难度升级配置 |
| config/diffusion_train.yaml | 扩散模型训练 | 实验性 |

## 7. 实验体系与当前状态

### 7.1 实验脚本与产出

| 实验 | 产出 | 状态 |
| --- | --- | --- |
| 主实验（7 条件 × 20 runs） | data/experiments/summary.md、results.json | 已完成，但为**合成模式（pipeline test）** |
| 深度消融 | deep_ablation_report.md | 已完成（合成模式） |
| 通道消融 | ablation_report.md | 已完成（合成模式） |
| 压力测试（23 场景） | stress_test_report.md | 17/23 通过 |
| BC 基线 | bc_baseline_report.md | 已完成（合成模式） |
| KS 策略一致性验证 | ks_report.md | **空表（0/0），未实际完成** |
| 真实 LLM 小规模验证 | real_llm_validation.json | **仅 SFM 成功，LLM 条件因缺 vllm 全部失败** |

### 7.2 数据文件现状

- `data/trajectories/`：20 个 oracle_run JSONL（约 1.9MB/个），由合成 oracle 脚本生成，**不是真实 LLM 行为轨迹**。
- `data/irl_weights.json`：5 人设权重；其中 guide/firefighter 为预设示例值，其余 3 个人设权重高度趋同、效率权重接近 0，**需用真实轨迹重学后复核**。
- `data/experiments/`：各报告与 results.json。

### 7.3 重要提醒（务必注意）

1. `summary.md` 中“Ours 91.7% 疏散率”等数字来自**合成管线测试**，论文引用前必须用真实 LLM 实验复现。
2. 论文稿（llm_irl_rl_evacuation.md）中的数字（如 90.1%）与合成报告口径不一致，需要统一。
3. KS 报告为空表，IRL 策略与 LLM 行为一致性的关键证据尚未补齐。
4. README 中“12 万条轨迹”“31% 拦截”“伤亡率下降 57%”等为项目自述，需以可复现数据为准。

## 8. 论文与文献工作流（2026-08-05 更新）

### 8.1 paper/ 目录

| 文件 | 用途 |
| --- | --- |
| llm_irl_rl_evacuation.md | 论文主稿（中文 Markdown，已有完整方法与实验章节草稿） |
| references.bib | 26 条文献（Zotero + Better BibTeX 可导入并自动维护） |
| related_work_draft.md | 相关工作中文完整草稿 |
| related_work_en.md | 相关工作英文版 |
| main.tex | 最小可编译 LaTeX 骨架（ctex + biblatex + 图表结构） |

### 8.2 Zotero 工作流

建议流程：Zotero 导入 references.bib → Better BibTeX 自动导出 .bib → LaTeX/Overleaf 引用 → Zotero 批注与笔记 → 更新相关工作草稿。

## 9. 仓库当前状态

- 当前分支：`v2.0-multimodal-diffusion`
- 已修改未提交：config/mall_floorplan.yaml、decision/cognitive_engine.py、execution/irl_recovery.py、execution/orchestrator.py、execution/rl_scheduler.py、perception/floorplan.py
- 未跟踪文件：config/mall_floorplan_hard.yaml、experiments/generate_oracle_trajectories.py、experiments/paper_experiment.py、论文汇报文档.md/.pdf、paper/ 下新增的文献与写作文件
- 核心自测（2026-08-05 本机运行）：物理引擎、安全护栏（11/11）、IRL（9/9）、RL 调度器（10/10）均通过；pytest 未安装，LLM/vllm 依赖未在本机安装（项目在云平台运行）。

## 10. 待办清单

1. 在云平台安装 vllm 并跑通真实 LLM 小规模验证（pure_llm / ours）。
2. 用真实 LLM 轨迹重新生成数据、重学 IRL 权重，检查人设权重是否分化。
3. 补齐 KS 策略一致性验证报告。
4. 统一论文稿与实验报告的数字口径，注明实验模式。
5. 补全 BibTeX 中 `others` 占位作者（Masked IRL、Generalist Reward Models）。
6. 提交当前未提交的代码改动，并考虑为论文写作开独立分支。
