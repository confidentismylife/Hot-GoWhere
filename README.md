# 基于LLM的多角色智能体灾害人群疏散仿真系统

## LLM-Powered Multi-Role Agent Crowd Evacuation Simulation

### Single GPU (RTX 4090 24GB) Edition · v2.1

---

> **架构说明（v2.2 路线A 修订）**：本系统采用 **LLM 认知仿真 + 安全护栏 + RL 区域调度辅助** 三层架构。RL 的奖励权重为**基于疏散行为文献手工设计的 5 维价值权重**（见 `data/irl_weights.json` 的 `_provenance` 字段）；仓库中保留的 MaxEnt IRL 管线为实验性组件，其局限性（奖励不可辨识性、轨迹来源）已在 `execution/irl_recovery.py` 头部注释中如实说明。

---

## 目录

- [1. 项目定位与核心创新](#1-项目定位与核心创新)
- [2. 架构：LLM 认知仿真 + RL 区域调度辅助](#2-架构llm-认知仿真--rl-区域调度辅助)
- [3. 三层认知架构总览](#3-三层认知架构总览)
- [4. 模块详解](#4-模块详解)
  - [4.1 感知层 (perception/)](#41-感知层-perception)
  - [4.2 决策层 (decision/)](#42-决策层-decision)
  - [4.3 执行层 (execution/)](#43-执行层-execution)
  - [4.4 群体智能 (group_intel/)](#44-群体智能-group_intel)
  - [4.5 IRL/Reward管线 (execution/)](#45-irlreward-管线-execution)
  - [4.6 训练管线 (training/)](#46-训练管线-training)
  - [4.7 可视化层 (visualization/)](#47-可视化层-visualization)
- [5. 与主流方案的系统性对比](#5-与主流方案的系统性对比)
- [6. 性能数据与优化实证](#6-性能数据与优化实证)
- [7. 创新点详析](#7-创新点详析)
- [8. 学术与职业价值](#8-学术与职业价值)
- [快速开始](#快速开始)

---

## 1. 项目定位与核心创新

本项目回答一个递进的学术问题链：

| 层面 | 核心问题 | 本项目的回答 |
|------|---------|------------|
| 建模层 | LLM能否模拟多样化人类疏散行为？ | 5角色×5人设的 Agent 体系，LLM生成有人类心理特征的决策 |
| 安全层 | LLM决策在安全关键场景中是否可靠？ | 7条硬约束安全护栏，LLM是"建议者"、护栏是"仲裁者" |
| **价值层** | **调度奖励如何体现人类疏散偏好？** | 基于文献手工设计的 5 维价值权重（safety/efficiency/social/conformity/comfort），经敏感性分析验证；MaxEnt IRL 作为实验性对照保留 |
| **调度层** | **价值权重能否指导全局调度？** | 分区独立 PPO（Zone-PPO）多智能体RL，训练环境内置"建议采纳率"建模以对齐部署因果链 |
| 工程层 | 单卡消费级GPU能否支撑全链路？ | AWQ量化 + vLLM + 异步流水线（具体性能数字以实测为准，见 results/ 下实验记录） |

---

## 2. 架构：LLM 认知仿真 + RL 区域调度辅助

### 2.1 架构动机与诚实声明

本项目最初设想为 "LLM → IRL → RL" 三级级联（LLM 行为 → IRL 学权重 → RL 用权重调度）。经过对仓库证据的审查后，**v2.2 路线A 修订为如下诚实叙事**：

- 现有轨迹数据（`data/trajectories/`）由规则 Oracle 生成，**不是 LLM 行为**；
- MaxEnt IRL 在线性特征下存在奖励不可辨识性（同一策略兼容多个奖励函数，参见 Skalse & Abate 2024），且特征存在共线性（comfort 是 safety 的别名，social/conformity 同源于 cooperation 字段），学习结果退化为 safety 独大或回退硬编码默认值；
- 因此**部署权重为手工设计**（基于疏散行为文献），IRL 管线保留为实验性组件。

当前架构的数据流：

```
文献/专家知识 ──手工设计──→ 5维价值权重 ──注入RL奖励──→ 区域调度策略
                                  │                          ↓
LLM认知仿真（多角色Agent） ←── 自然语言建议注入Prompt ── 出口偏好
        │
        ↓
安全护栏（7条硬约束，仲裁者）──→ 最终行动
```

### 2.2 各层详解

**LLM 行为层**
- 多 Agent 由 LLM 驱动疏散决策（角色：平民/引导员/消防员/指挥员）
- 每次决策包含：出口选择、速度(跑/走/爬/等)、协作模式(帮家人/跟人群/带路人)
- 决策记录由 `TrajectoryCollector` 收集，可用于未来的 IRL 实验

**价值权重层（手工设计，非 IRL 学习）**
- 5个奖励特征：**安全**(远离火/烟)、**效率**(近出口快速移动)、**社交**(帮家人带路人)、**从众**(跟人群听指挥)、**舒适**(走熟悉路线避免劳累)
- 5种人设：未培训老人、未培训年轻人、已培训店员、引导员、消防员
- 权重设计依据与注意事项见 `data/irl_weights.json` 的 `_provenance` 字段
- MaxEnt IRL 管线（`execution/irl_recovery.py`）保留，用于未来真实 LLM 轨迹充足后的对照实验

**Zone-PPO 调度层**
- 将商场划分为4个象限区域(NW/NE/SW/SE)
- 每个区域一个分区独立 PPO（Zone-PPO）调度Agent (3层MLP, 观测→隐藏→出口偏好[-1,+1])
- 奖励函数使用上述 5 维价值权重 + 结果导向 shaping 项
- **训练对齐部署（路线A）**：`FastTrainingSimulator` 内置 `advice_accept_rate` 参数（默认 0.6），模拟部署时"建议注入 LLM prompt → 可能被忽略 → 护栏拦截约 31%"的真实因果链，避免学到的策略假设一个不存在的动作→结果通路
- RL输出转为中文自然语言建议，注入LLM Prompt：`【西北区调度中心建议】✅ 强烈推荐 出口3 ⚠️ 避免前往 出口2`
- LLM仍做最终决策——RL只是"建议"，不替代LLM的人类判断力；护栏是最终仲裁者

> 实现说明：当前实现为 MAPPO 风格的 CTDE —— 各分区独立策略头（分散执行）+
> 共享集中式价值网络（仅训练时使用）；联合策略为因子化高斯策略
> （联合 log-prob 为各分区之和），并非完整联合动作模型。
> 早期文档中的“P-MAPPO”统一为“分区独立 PPO（Zone-PPO）”。

### 2.3 技术细节

```
观测空间 (ZoneObservation):
  - 各出口烟雾浓度 [0-1]
  - 各出口拥堵人数
  - 各出口火源距离 (m)
  - 区域内Agent数量/平均恐惧/平均体力
  - 已培训人员比例/老年人比例
  - 前一时刻出口使用量

动作空间 (ZoneAction):
  - 各出口偏好分数 [-1, +1]
  - +1=强烈推荐, -1=避免前往

奖励函数 (手工设计的价值权重):
  R_zone = w_safety·f_safety + w_efficiency·f_efficiency
         + w_social·f_social + w_conformity·f_conformity
         + w_comfort·f_comfort
  其中w为基于文献手工设计的各人设权重均值（见 data/irl_weights.json）
```

### 2.4 与已有方案的定位区别

| 维度 | 纯物理仿真（SFM/CA） | LLM 认知仿真（Berkeley 等） | 本项目 |
|------|:---:|:---:|:---:|
| 行为建模 | 理性粒子 | LLM persona 驱动 | LLM 多角色 + 护栏仲裁 |
| 调度机制 | 无 | 无 | Zone-PPO 区域建议（辅助，非控制） |
| 奖励来源 | — | — | 文献手工设计 5 维权重 + 敏感性分析 |
| 训练/部署对齐 | — | — | 训练内置采纳率建模（advice_accept_rate） |
| 验证方式 | 流量对比 | 真实数据校准 | 真实 LLM 仿真对比（固定 seed + 采纳率指标） |

---

## 3. 三层认知架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                     感知层 (Perception)                           │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ CA灾害仿真        │  │ VLM视觉感知       │  │ YOLO行人检测  │  │
│  │ 火/震/洪 三灾种   │  │ Qwen-VL-7B-AWQ   │  │ yolov8n       │  │
│  │ 0.5m网格, 4通道   │  │ CCTV画面→中文描述  │  │ 密度热点+异常  │  │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘  │
│           └──────────────────────┼──────────────────────┘         │
│                         自然语言描述 + 检测结果                     │
├──────────────────────────────────────────────────────────────────┤
│                     决策层 (Cognition)                            │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ vLLM批量推理引擎  │  │ 安全护栏(7约束)   │  │ 知识库RAG     │  │
│  │ Qwen2.5-3B-AWQ   │  │ LLM建议→护栏仲裁  │  │ ChromaDB+BGE  │  │
│  │ Prefix Caching   │  │ 每条拦截可审计    │  │ 22条逃生规则  │  │
│  │ Async Pipeline   │  │ 31%决策被拦截纠正  │  │ 角色路由      │  │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘  │
│           └──────────────────────┼──────────────────────┘         │
│           5种角色 × 3种灾害 = 15种Prompt模板                       │
│           + RL区域调度建议注入 (v2.1)                              │
│           决策输出: {出口, 速度, 协作模式, 推理链}                  │
├──────────────────────────────────────────────────────────────────┤
│                     执行层 (Execution)                            │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ 批量社会力模型    │  │ 分区独立 PPO（Zone-PPO） RL调度    │  │ 群体智能       │  │
│  │ Numba JIT单次调用 │  │ 4 Zone MLP推理   │  │ 信息传播+恐惧  │  │
│  │ 800人×1调用≈3ms  │  │ 手工价值权重驱动奖励  │  │ +体力+记忆    │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│                 价值权重 + RL 调度辅助 (v2.2 路线A)                  │
│                                                                   │
│  文献手工权重 ──→ 5维奖励 ──→ Zone-PPO 训练(含采纳率建模) ──→ 中文建议→Prompt │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. 模块详解

### 4.1 感知层 (perception/)

#### `environment.py` — CA灾害仿真器

用2D元胞自动机模拟火灾、地震、洪水三类灾害。网格分辨率0.5m，每格4个通道：烟雾浓度(0-1)、温度(°C)、结构完整性(0-1)、是否着火(bool)。

**设计选择——CA而非CFD（计算流体力学）**：
1. 单GPU的CPU预算有限，CFD每个时间步计算量远超LLM推理
2. 疏散仿真的精度瓶颈在决策层（人类行为建模），物理精度不是主要误差来源
3. 与经典疏散论文（Helbing 2000, Zheng 2009）保持可比性

#### `nl_converter.py` — 数值→自然语言转换器

将网格浮点数翻译为LLM能理解的中文描述。定义了阈值映射：
- 烟雾 <0.1 → "几乎无烟"，>0.6 → "浓烟弥漫"
- 温度 <35°C → "正常"，>150°C → "极高，有生命危险"

**v2.1 新增**：`rl_context()` 方法和 `full_context()` 的 `rl_preference` 参数，支持RL调度建议注入到LLM Prompt中。

#### `vlm_perceiver.py` — VLM视觉感知通道

Qwen2.5-VL-7B-AWQ (~5GB)，将CCTV监控画面转化为中文场景描述。30 tick（3秒）调用一次。支持Mock模式用于消融实验。

#### `yolo_detector.py` — YOLO行人检测通道

YOLOv8n (~6MB)，检测行人位置、数量和密度热点。与VLM互补：VLM提供语义理解，YOLO提供精确数据。

---

### 4.2 决策层 (decision/)

#### `agent_state.py` — 智能体状态数据模型

`AgentProfile`（静态：年龄/性别/熟悉度/性格/角色）和 `AgentDynamic`（动态：位置/速度/体力/恐惧/记忆/目标出口）。dataclass实现，无运行时校验开销。

#### `agent_roles.py` — 多角色体系定义

5种角色构成完整应急指挥链：

| 角色 | 职责 | 特有决策字段 |
|------|------|------------|
| Civilian (平民) | 自主逃生 | 出口/速度/协作模式 |
| Global Commander | 全局态势评估 | 广播消息/资源调度 |
| Area Commander | 片区管理 | 片区优先级 |
| Firefighter | 灭火+搜救 | 目标位置/灭火点 |
| Guide | 引导人群 | 出口引导/召集跟随者 |

#### `cognitive_engine.py` — vLLM批量推理引擎

**核心优化**：
- AWQ 4-bit量化：模型6GB→2GB
- Prefix Caching：按`{role}_{disaster_type}`分组，共享前缀命中率~80%，节省66%编码量
- 异步流水线：LLM后台推理不阻塞物理引擎
- 决策采样：~2% Agent/tick重新决策

**v2.1 新增**：`submit_batch()` 的 `rl_preferences` 参数，支持每Agent独立的RL调度建议。

#### `prompt_manager.py` — 多角色提示词管理

5角色×3灾害=15套Prompt模板。强制JSON输出格式。`parse_response()` 鲁棒解析（处理Markdown代码块、截断JSON、LLM附加文字）。

**v2.1 新增**：`build_user()` 的 `rl_preference` 参数，透传到NL Converter。

#### `safety_guard.py` — 安全护栏（7条硬约束）

| # | 约束 | 阈值 | 动作 |
|---|------|------|------|
| 1 | 出口烟雾封锁 | smoke > 0.6 | 切换到最佳可用出口 |
| 2 | 路径火焰阻挡 | 路径经过火源 | 切换出口 |
| 3 | 体力-速度匹配 | stamina < 20 | RUN→WALK |
| 4 | 受伤限制 | injured=True | 强制WALK |
| 5 | 站位着火 | 在火上 | 强制RUN |
| 6 | 距离合理性 | 出口>150m | 切换到最近出口 |
| 7 | 危险中等待 | smoke>0.5或temp>60 + WAIT | WAIT→WALK |

实测：拦截31%的LLM原始决策，伤亡率下降57%。

#### `knowledge_base.py` — 逃生知识库RAG

22条中文逃生规则，ChromaDB+BGE-small-zh-v1.5语义检索。**角色路由**：专业人员获取领域知识，平民获取通用安全常识——防止"所有市民突然变成消防专家"的不真实行为。

---

### 4.3 执行层 (execution/)

#### `orchestrator.py` — 主仿真循环 (1051行)

每tick执行流程：
```
① 灾害演进 (disaster.step)
② 环境快照 (snapshot)
③ RL区域调度 (rl_scheduler.infer) — v2.1新增
④ LLM决策提交 (submit_batch + RL建议 + 非阻塞)
⑤ LLM结果收集 (collect_results + 安全护栏 + 应用决策)
⑥ 轨迹记录 (trajectory_collector.record_decision) — v2.1新增
⑦ VLM + YOLO感知
⑧ 群体智能 (信息传播 + 恐惧 + 体力)
⑨ 物理引擎 (step_all — 批量JIT)
⑩ 统计 + 可视化 + 终止检查
```

**v2.1 新增开关**：
- `enable_irl_collection: true/false` — 是否采集IRL训练轨迹
- `enable_rl_scheduling: true/false` — 是否启用RL区域调度
- `_rl_preferences_cache` — 只对需要决策的Agent生成RL建议(优化后)

#### `batched_physics.py` — 批量社会力模型

Numba JIT实现，所有Agent数据→平铺数组→一次JIT调用：800人×1调用≈3ms。5种力：目标引力、社交排斥力、障碍排斥力、家庭吸引力、边界约束。空间哈希网格O(1)邻居查找。

#### `diffusion_policy.py` — 扩散模型轨迹生成（实验性）

MID架构的条件扩散模型，生成Agent未来轨迹。DDIM 100步采样，单条轨迹≈50ms。需大量数据预训练，默认关闭。

---

### 4.4 群体智能 (group_intel/)

#### `propagation.py` — 信息传播与群体动力学

3种信息传播机制（官方广播/P2P交流/恐惧传染）、体力系统（跑-5/s、走-0.5/s、等+2/s）、记忆系统（最近20条事件）。

---

### 4.5 IRL/Reward 管线 (execution/)

> **状态说明（v2.2 路线A）**：本模块中的 MaxEnt IRL 为**实验性组件**。当前部署的奖励权重为手工设计（见 `data/irl_weights.json` 的 `_provenance`）；轨迹数据目前由规则 Oracle 生成，且 MaxEnt 在线性特征下存在奖励不可辨识性，详见 `irl_recovery.py` 头部注释。

#### `irl_recovery.py` — 轨迹采集 + MaxEnt IRL（实验性，~500行）

**TrajectoryCollector**：挂在Orchestrator上，实时采集每个Agent的决策轨迹。
- 记录字段：(tick, 位置, 烟雾浓度, 火源距离, 出口距离, 速度选择, 协作模式, 恐慌水平, 是否被安全护栏拦截)
- 输出：JSONL文件

**IRLRecovery**：Maximum Entropy IRL算法。
- 输入：从JSONL加载的AgentTrajectory列表
- 算法：梯度下降最小化 expert_fe - policy_fe（专家特征期望 - 当前策略特征期望）
- 5个奖励特征从轨迹中提取：
  - **safety**: (1-烟雾) × min(火源距离/50, 1)
  - **efficiency**: 1/(1+出口距离/50) × 速度系数
  - **social**: 协作模式映射(帮家人/带路人→1, 跟人群→0.5, 无→0.1)
  - **conformity**: 跟人群→1, 无→0.3, 其他→0.5
  - **comfort**: (1-恐惧/10) × 速度舒适度(走→1, 跑→0.3)
- 输出：5种人设×5维权重向量

**使用方式**：
```bash
# 采集轨迹 (仿真时设置 irl.enabled: true)
python main.py --config config/mall_floorplan.yaml --agents 200 --no-viz

# 从轨迹学习权重
python -m execution.irl_recovery --trajectory_dir data/trajectories --output data/irl_weights.json
```

#### `rl_scheduler.py` — 分区独立 PPO（Zone-PPO）区域调度器 (~560行)

**ZoneDefinition**：150m×80m商场4象限：
| Zone | 名称 | 范围 | 主要出口 |
|------|------|------|---------|
| 0 | 西北区 | x∈[0,75), y∈[40,80] | 0,1,4,5 |
| 1 | 东北区 | x∈[75,150], y∈[40,80] | 1,2,3,7 |
| 2 | 西南区 | x∈[0,75), y∈[0,40] | 4,5,6 |
| 3 | 东南区 | x∈[75,150], y∈[0,40] | 3,6,7 |

**RLZoneScheduler**：
- 3层MLP策略网络：(约47维观测→128→128→8出口) + tanh → [-1, +1]
- EMA平滑：`0.7×新偏好 + 0.3×历史偏好`，防止出口推荐震荡
- 推理速度：<1ms (4 zones × MLP forward pass)
- Heuristic回退：未训练时用IRL权重驱动的启发式评分

**inject_rl_preferences()**：查找Agent所在zone，将RL调度建议转为中文：
```
【西北区调度中心建议】
  ✅ 强烈推荐 出口3、建议考虑 出口1
  ⚠️ 避免前往 出口2（严重拥堵或危险）
```

**使用方式**：
```bash
# 初始化策略权重
python -m execution.rl_scheduler --mode init --output data/rl_policy.json

# 加载IRL权重 + 推理 (在仿真中)
# 设置 rl_scheduling.enabled: true 和 rl_scheduling.irl_weights: "data/irl_weights.json"
```

#### `reward_analysis.py` — IRL权重分析+可视化 (~390行)

**RewardAnalyzer**：生成论文级别的分析工件：
- **权重对比表**：Markdown表格，5人设×5特征的权重矩阵
- **雷达图** (`radar.png`)：5人设的权重分布
- **热力图** (`heatmap.png`)：人设×特征的权重热力图
- **KL散度矩阵** (`divergence.png`)：人设对之间的价值分歧
- **关键发现**：自动提取5条定量结论（最大差异特征、培训效果、消防员利他性等）

**使用方式**：
```bash
python -m execution.reward_analysis --weights data/irl_weights.json --output data/analysis/
```

---

### 4.6 训练管线 (training/)

完整的QLoRA微调闭环：`generate_data.py`（安全护栏自动标注76K样本）→ `train_lora.py`（4-bit QLoRA, 15MB adapter）→ `compare_models.py`（基座vs微调vs规则，11项指标）。

---

### 4.7 可视化层 (visualization/)

4种模式覆盖全场景：Pygame本地实时渲染、Matplotlib无头录帧、Flask Web远程监控、Gradio交互式仪表板（Natural Language查询 + Plotly图表 + 手动指挥官干预）。

---

## 5. 与主流方案的系统性对比

| 维度 | 传统ABM<br>(Helbing 1995) | RL疏散<br>(Lee 2022) | 纯LLM Agent<br>(GenAgents 2023) | 本项目 |
|------|:---:|:---:|:---:|:---:|
| 决策模型 | if-else规则 | 策略网络 | LLM (GPT-4) | LLM + 安全护栏 |
| 人类心理建模 | 无 | 间接(reward) | Prompt | **直接(恐惧/信任/利他)** |
| 安全性保证 | 规则安全 | 黑盒 | **无** | **7条可审计约束** |
| 奖励函数来源 | N/A | 人工设计 | N/A | **文献手工设计 5 维权重（IRL 为实验性对照）** |
| 知识迁移 | N/A | N/A | N/A | **价值权重 → RL 奖励 → 调度建议** |
| 未见场景泛化 | 无法 | 需重训练 | **零样本** | **零样本** |
| 可解释性 | 低 | 极低 | 高(有幻觉) | **高(推理链+约束日志)** |
| 多角色协同 | 需手动编码 | 需每角色训练 | 仅单一角色 | **5角色+指挥链** |
| GPU需求 | 无 | 训练需GPU | 多卡A100 | **单卡4090** |
| 规模上限 | 百万级 | 十万级 | <100 agents | **800-2000 agents** |

---

## 6. 性能数据与优化实证

### 6.1 单卡4090实测

| 配置 | 200 agents | 500 agents | 800 agents | 1500 agents |
|------|:---:|:---:|:---:|:---:|
| 纯物理 | 2ms | 5ms | 8ms | 15ms |
| +LLM (v2.0优化) | 35ms | 70ms | 120ms | 250ms |
| +YOLO | 42ms | 85ms | 140ms | 300ms |
| +VLM (mock) | 55ms | 110ms | 180ms | 380ms |
| +VLM (real) | 180ms | 350ms | 520ms | — |

### 6.2 v2.0优化拆解

| 优化项 | 原始耗时 | 优化后 | 加速比 |
|--------|---------|--------|:---:|
| KB查询缓存 | 800ms | 26ms | 30× |
| O(1) Agent查找 | 150ms | 0.001ms | 100000× |
| 批量物理引擎 | 525ms | 8ms | 65× |
| **综合** | **999ms/tick** | **50-150ms/tick** | **7-20×** |

### 6.3 RL调度推理开销 (v2.1)

| 操作 | 耗时 |
|------|------|
| Agent分区 (600→4 zones) | <0.1ms |
| 4×ZoneObservation构建 | ~0.2ms |
| 4×MLP前向传播 | ~0.3ms |
| 决策Agent RL建议生成 (~12个) | ~0.1ms |
| **总计** | **<1ms/tick** |

### 6.4 GPU显存分配 (24GB)

```
Qwen2.5-3B-AWQ (vLLM):      ████ 2.0 GB
KV Cache (prefix caching):  ████ 1.5 GB
Qwen-VL-7B-AWQ (VLM, 可选): ██████████ 5.0 GB
YOLOv8n:                    █ 0.04 GB
ChromaDB Embedding (BGE):   ██ 0.5 GB
扩散模型 (可选):             ████ 2.0 GB
RL调度器 (4×MLP):           █ <0.01 GB
───────────────────────────────────────
合计 (全开):                ~11.0 GB
安全余量:                   ~13.0 GB
```

---

## 7. 创新点详析

### 创新点1：LLM → IRL → RL 三级知识迁移范式 ⭐ 核心

**问题**：如何让审稿人相信"LLM + RL"不只是两个技术的简单拼接？

**方案（v2.2 修订后）**：价值层采用基于疏散行为文献手工设计的 5 维权重，经敏感性分析验证后注入 RL 奖励；MaxEnt IRL 管线保留为实验性组件，待真实 LLM 轨迹充足后作为对照。训练环境内置采纳率建模（advice_accept_rate），对齐部署时"建议可能被忽略/被护栏拦截"的真实因果链。

**与已有工作的区别**：
- 不同于纯物理仿真（SFM/CA）——本项目用 LLM 建模认知层（犹豫/折返/从众），并用安全护栏保证安全关键约束可审计
- 不同于RLHF（用RL训练LLM对齐人类偏好）——本项目是反过来用LLM教RL什么是好的
- 不同于LLM+RL并行融合（两个组件独立输出后加权）——本项目是递进的知识迁移

### 创新点2：LLM+符号化安全护栏的"建议-仲裁"架构

LLM是"建议者"，安全护栏是"仲裁者"。7条约束全部由领域专家可验证的物理/生理阈值定义。31%决策被拦截，伤亡率下降57%。与SayCan、Constitutional AI的区别：推理时实时校验、规则可随时调整、每条拦截可审计。

### 创新点3：多角色LLM智能体的指挥链协同建模

5种角色构成灾害应急指挥链。涌现现象：当官方广播与个人观察矛盾时，"信任+从众"权衡产生复杂群体分化。

### 创新点4：VLM+YOLO双通道感知融合

VLM提供语义理解（"左侧有浓烟"），YOLO提供精确数据（"右上角15人拥挤"）。两通道通过自然语言拼接融合，LLM本身作为"融合器"。

### 创新点5：单卡消费级GPU的全栈优化

五层优化组合（AWQ+Prefix Caching+异步流水线+KB缓存+批量物理）使800 Agent在RTX 4090上以120ms/tick运行。降低了LLM Agent研究的硬件门槛。

### 创新点6：完整的实验闭环与量化分析

76K自动标注→QLoRA微调→三模型对比11项指标→IRL权重分析雷达图+KL散度→RL调度策略对比。从数据到分析的全自动化实验管线。

---

## 8. 学术与职业价值

### 8.1 论文建议

**标题（建议，v2.2 口径）**：《基于LLM认知仿真与价值权重RL调度的灾害人群疏散多智能体系统》

**英文标题（建议）**：*LLM-Driven Crowd Evacuation Simulation with Value-Weighted Zone-Level RL Scheduling Advice*

**主要贡献声明（诚实版）**：
1. 多角色 LLM 认知疏散仿真框架（5 角色 + 7 条可审计安全护栏）
2. 基于文献的 5 维价值权重设计与敏感性分析（附 MaxEnt IRL 实验性对照及其局限性分析）
3. Zone-PPO 区域调度辅助层，训练环境内置建议采纳率建模以对齐部署因果链
4. 真实 LLM 仿真下的对照实验设计（固定 seed + 采纳率/护栏修改率指标）

### 8.2 面试话术（2分钟版）

> "我的毕业设计实现了一个LLM驱��的灾害疏散仿真系统，核心定位是 LLM 认知仿真 + 安全护栏 + Zone-PPO 区域调度辅助。认知层用多角色 LLM Agent 模拟人类疏散决策（犹豫/折返/从众），配 7 条可审计的安全护栏作为最终仲裁者；调度层的奖励函数采用基于疏散行为文献手工设计的 5 维价值权重，训练环境内置建议采纳率建模以对齐部署时的真实因果链（MaxEnt IRL 管线作为实验性对照保留）。
>
> 工程上，我通过AWQ量化、Prefix Caching和异步流水线把多 Agent 推理压到单张4090可跑的水平，还搭了QLoRA微调管线，用安全护栏自动标注训练数据。
>
> 这个项目让我完整走了一遍从仿真建模→RL训练→部署验证→问题复盘的全链路，涵盖了Prompt设计、RAG、微调、部署优化和论文级可视化。"

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com

# 2. 验证物理引擎 (无需GPU)
python tests/test_physics_only.py

# 3. 基础运行
python main.py --agents 200 --duration 120                    # 本地Pygame渲染
python main.py --no-viz --record --agents 500                  # 服务器录帧

# 4. 训练与运行 RL 区域调度（v2.2 路线A 流程）
# Step 1: 离线训练 Zone-PPO（训练环境内置采纳率建模 advice_accept_rate）
python main.py --config config/mall_floorplan_rl_extreme.yaml --train-rl \
  --train-rl-episodes 500 --irl-weights data/irl_weights.json \
  --rl-output data/rl_policy_v6.json

# Step 2: 权重敏感性分析（验证 5 维权重有区分度）
python -m execution.reward_analysis --weights data/irl_weights.json --output data/analysis/

# Step 3: RL vs 启发式对照实验（固定 seed）
python -m experiments.compare_rl_heuristic \
  --rl-config config/mall_floorplan_rl_extreme.yaml \
  --heuristic-config config/mall_floorplan_extreme.yaml \
  --agents 200 --duration 360 --seeds 42 43 44 45 46 \
  --output results/compare_rl_extreme_v6

# Step 4（实验性）: MaxEnt IRL 对照 —— 仅在真实 LLM 轨迹充足后使用
# python -m execution.irl_recovery --trajectory_dir data/trajectories --output data/irl_weights_irl.json

# 5. 多模态感知
python main.py --vlm --vlm-mock --yolo --agents 200             # Mock VLM
python main.py --vlm --yolo --agents 100                        # 真实VLM (~5GB VRAM)

# 6. Gradio交互式仪表板
python main.py --gradio --gradio-port 8081

# 7. LoRA微调管线
python -m training.generate_data --num-scenarios 50 --output data/train_lora.jsonl
python -m training.train_lora --data data/train_lora.jsonl --val-data data/train_lora_val.jsonl --epochs 3 --output-dir output/lora_evac
python -m training.compare_models --lora-path output/lora_evac/final --num-scenarios 10

# 8. 使用微调模型
python main.py --lora output/lora_evac/final --agents 200 --gradio
```

## 项目结构

```
single_gpu_evacuation/
├── main.py                         # 入口: CLI参数解析 + 模式选择
├── config/
│   ├── default.yaml                #   默认配置 (800 agents, 300s, fire)
│   ├── mall_floorplan.yaml         #   商场场景配置 + IRL + RL调度配置
│   └── diffusion_train.yaml        #   扩散模型训练超参
├── requirements.txt                 # Python依赖
│
├── perception/                      # 感知层 — 环境→自然语言
│   ├── environment.py               #   CA灾害仿真 (火/震/洪, 四通道)
│   ├── nl_converter.py              #   数值→中文描述转换 (v2.1: +RL调度注入)
│   ├── vlm_perceiver.py             #   VLM视觉感知 (Qwen-VL-7B-AWQ)
│   └── yolo_detector.py             #   YOLO行人检测 (yolov8n)
│
├── decision/                        # 决策层 — LLM认知引擎
│   ├── agent_state.py               #   Agent数据模型 (Profile+Dynamic)
│   ├── agent_roles.py               #   5种角色定义 + 专用决策结构
│   ├── cognitive_engine.py          #   vLLM批量推理引擎 (v2.1: +RL建议接入)
│   ├── prompt_manager.py            #   多角色Prompt模板 (v2.1: +RL建议注入)
│   ├── safety_guard.py              #   7条硬约束安全护栏
│   └── knowledge_base.py            #   ChromaDB+BGE知识库RAG + 角色路由
│
├── execution/                       # 执行层 — 物理仿真 + IRL + RL
│   ├── orchestrator.py              #   主仿真循环 (v2.1: +IRL采集+RL调度)
│   ├── batched_physics.py           #   批量社会力模型 (Numba JIT)
│   ├── irl_recovery.py              #   MaxEnt IRL实验性管线（部署权重为手工设计，见文件头注释）
│   ├── rl_scheduler.py              #   ★ 分区独立 PPO（Zone-PPO）区域调度+采纳率建模 [v2.2]
│   ├── reward_analysis.py           #   ★ IRL权重分析+雷达图+KL散度 [v2.1]
│   ├── diffusion_policy.py          #   扩散模型轨迹生成 (实验性v2.0)
│   └── diffusion_trainer.py         #   扩散模型训练脚本
│
├── group_intel/                     # 群体智能 — 信息+情绪+体力
│   └── propagation.py               #   信息传播/恐惧传染/体力/记忆
│
├── training/                        # LoRA微调管线
│   ├── generate_data.py             #   Oracle数据自动标注
│   ├── train_lora.py                #   QLoRA微调 (4-bit NF4)
│   └── compare_models.py            #   三模型对比评估 (11项指标)
│
├── visualization/                   # 可视化层 (4种模式)
│   ├── renderer.py                  #   Pygame本地实时渲染
│   ├── headless_renderer.py         #   Matplotlib无头录帧
│   ├── web_server.py                #   Flask Web远程监控
│   └── gradio_app.py                #   Gradio交互式仪表板
│
├── tests/                           # 测试
│   ├── test_safety_guard.py         #   安全护栏单元测试 (7项)
│   ├── test_v2_pipeline.py          #   VLM+YOLO集成测试 (6项)
│   ├── test_safety_integration.py   #   安全集成测试
│   └── test_physics_only.py         #   物理引擎性能测试
│
└── data/                            # 运行时数据
    ├── disaster_kb/                 #   ChromaDB知识库持久化
    ├── trajectories/                #   IRL轨迹采集 (.jsonl)
    ├── irl_weights.json             #   IRL学习到的权重
    ├── rl_policy.json               #   RL策略网络权重
    └── analysis/                    #   IRL权重分析输出 (图表+报告)
```

## 技术栈总览

| 类别 | 技术 | 选型理由 |
|------|------|---------|
| LLM推理 | vLLM 0.6+ | Continuous Batching + Prefix Caching |
| 基座模型 | Qwen2.5-3B-Instruct-AWQ | 中文好 + 2GB显存 + 3B足够推理 |
| VLM | Qwen2.5-VL-7B-Instruct-AWQ | 原生中文视觉理解 |
| 检测 | YOLOv8n | 6MB显存, 实时检测 |
| 知识库 | ChromaDB + BGE-small-zh | 轻量中文语义检索 |
| 物理引擎 | Numba JIT | 批量编译, 不离开Python |
| **IRL算法** | **MaxEnt IRL** | **最大熵原则, 适合多样化人类行为** |
| **RL框架** | **分区独立 PPO（Zone-PPO） (轻量MLP)** | **4 zone <1ms推理, 共享参数** |
| 微调 | QLoRA (PEFT + BnB) | 4GB显存训练, 15MB adapter |
| 可视化 | Matplotlib + Plotly | 论文图表 + 交互式分析 |
| 前端 | Gradio 6.0 + Flask | 交互式Dashboard + Web监控 |
| 硬件 | RTX 4090 24GB | 消费级GPU, 可复现 |

## 引用

```bibtex
@software{llm_evacuation_2025,
  title     = {From LLM Behavior to Optimal Scheduling:
               A Three-Tier Cascade Architecture with IRL for
               Multi-Role Agent Crowd Evacuation Simulation},
  year      = {2025},
  note      = {Single GPU (RTX 4090) Edition.
               LLM→IRL→RL cascade: LLM generates behavior data →
               MaxEnt IRL recovers value weights (5 personas × 5 features) →
               分区独立 PPO（Zone-PPO） RL optimizes zone-level scheduling.
               Three-layer perception-cognition-execution architecture
               with 7 hard safety constraints, VLM+YOLO dual perception,
               and QLoRA fine-tuning pipeline.},
  keywords  = {LLM Agents, Inverse Reinforcement Learning,
               Multi-Agent RL, Crowd Evacuation, Safety Guardrails}
}
```

---

*本项目的核心发现可以用一句话概括：**在安全关键场景中，LLM是好的"行为示范者"，IRL是好的"价值翻译器"，RL是好的"调度执行者"。三者不是并行竞争关系，而是递进的知识蒸馏关系。** 这一架构的意义远超疏散仿真本身——它为"人机协同决策"提供了一种可复制、可审计的方法论。*
