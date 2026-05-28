# v2.0 完整技术方案：多模态LLM疏散仿真系统

> Status: 设计中 | 开发卡: 4090/5090 | 实验卡: H800 80GB | 基于 v1.0-final

---

## 目录

1. [动机：v1.0 的三个上限](#1-动机v10-的三个上限)
2. [系统全貌](#2-系统全貌)
3. [模块一：VLM 视觉感知](#3-模块一vlm-视觉感知)
4. [模块二：扩散模型轨迹生成](#4-模块二扩散模型轨迹生成)
5. [训练数据管线](#5-训练数据管线)
6. [显存预算与GPU策略](#6-显存预算与gpu策略)
7. [接口契约](#7-接口契约)
8. [实验矩阵](#8-实验矩阵)
9. [四周实施计划](#9-四周实施计划)

---

## 1. 动机：v1.0 的三个上限

v1.0 跑通了，但有三件事做不到：

### 上限1：感知盲区

```
v1.0 看到的:                   真实场景中还有:
  "烟雾浓度 45%"                 "消防喷淋启动，地面大量积水"
  "温度 52°C"                   "有人摔倒被柱子挡住"
  "3人在(22,31)"               "应急灯闪烁引导方向"
                                "墙皮脱落堵住半边走廊"

v1.0 只能用预先定义的传感器类型。VLM 能看到"任何东西"。
```

### 上限2：轨迹假得明显

```
社会力模型转弯:                 真人转弯:
  ──────────╮                    ──────────╲
             ╲                                ╲
              ╲                                ○ (停顿0.2s, 回头看)
               ╲                                ╲
                ╲                                ╲

物理公式的"反推"              训练数据的"先看再走"
```

### 上限3：决策受限

v1.0 的 3B-AWQ 模型在复杂场景下有时输出不合理决策（如选已被浓烟封锁的出口）。7B-FP16 模型常识推理更强。

---

## 2. 系统全貌

### 2.1 端到端数据流

```
                          ┌─────────────────────┐
                          │    CCTV 摄像头画面     │
                          │    640×480 RGB       │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                │
            ┌──────────────┐ ┌──────────────┐        │
            │  通道A: YOLO  │ │ 通道B: VLM    │        │
            │  检测+追踪    │ │ 画面语义理解   │        │
            │              │ │              │        │
            │ 耗时: ~5ms   │ │ 耗时: ~1.5s  │        │
            │ 输出: boxes[] │ │ 输出: 文本描述 │        │
            └──────┬───────┘ └──────┬───────┘        │
                   │                │                │
                   ▼                ▼                │
            ┌──────────────────────────────────┐     │
            │         NL Converter (融合)        │     │
            │                                   │     │
            │  YOLO数据 → 结构化数值              │     │
            │  VLM文本 → 自然语言段落             │     │
            │  IoT数据  → 传感器读数              │     │
            │                                   │     │
            │  输出: 完整中文场景描述 (~800字)     │     │
            └──────────────┬───────────────────┘     │
                           │                         │
                           ▼                         │
            ┌──────────────────────────────────┐     │
            │       LLM 决策引擎 (Qwen-7B)       │     │
            │                                   │     │
            │  每3秒/环境剧变时 批量推理           │     │
            │  输入: Prompt (系统+场景+个人+记忆)  │     │
            │  输出: JSON (目标出口, 速度, 协作)   │     │
            │  耗时: ~0.3s/agent (批量)          │     │
            └──────────────┬───────────────────┘     │
                           │                         │
                           ▼                         │
            ┌──────────────────────────────────┐     │
            │     扩散模型轨迹生成 (MID)          │     │
            │                                   │     │
            │  条件: LLM意图 + 场景占用图          │     │
            │  输入: 起点 + 目标 + 障碍物         │     │
            │  输出: 31步 × 2坐标 平滑轨迹        │     │
            │  耗时: ~80ms/agent (可批量)        │     │
            └──────────────┬───────────────────┘     │
                           │                         │
                           ▼                         │
            ┌──────────────────────────────────┐     │
            │       物理执行 + 碰撞检测          │     │
            │  播放预生成轨迹 + 紧急避让          │     │
            └──────────────────────────────────┘     │
```

### 2.2 模块调用时序

```
Tick N ───────────────────── Tick N+30 ────────────── Tick N+60

VLM:  [=====推理~1.5s=====]                  [=====推理=====]
          ↓ 缓存场景描述
LLM:      [=====批量推理~2s=====]            [=====推理=====]
               ↓ 决策就绪
Diffusion:     [==生成轨迹~0.5s(批量)==]      [==生成==]
                    ↓ 播放31步
Physics: ================================================
           每tick只做插值 + 紧急碰撞检测 (~2ms)

所有重型推理异步后台执行。主循环永远不阻塞。
```

---

## 3. 模块一：VLM 视觉感知

### 3.1 模型选型

| 候选 | 显存(INT4) | 推理速度 | 中文能力 | 结论 |
|------|:---:|:---:|:---:|------|
| Qwen2.5-VL-7B | 5GB | ~30 tok/s | 强 | **首选** |
| InternVL2-4B | 2.5GB | ~50 tok/s | 中 | 备选(轻量) |
| Qwen2.5-VL-3B | 2GB | ~60 tok/s | 一般 | 开发调试用 |
| Llama-3.2-11B-Vision | 7GB | ~20 tok/s | 弱 | 不推荐 |

**选 Qwen2.5-VL-7B-INT4**：
- 中文场景描述能力强（你整个系统是中文Prompt）
- INT4量化后仅5GB显存
- 和决策LLM(Qwen-7B)同一家族，tokenizer兼容

### 3.2 Prompt 设计

```python
VLM_SYSTEM = """你是一个地铁站监控系统的视觉分析模块。
请仔细观察画面，以客观、精确的语言描述以下内容:

1. 烟雾分布: 位置、浓度、扩散方向
2. 火焰情况: 位置、大小、是否在蔓延
3. 人群状态: 分布、密度、移动方向、是否有异常行为(摔倒/推挤/逆行)
4. 环境变化: 积水、掉落物、应急指示灯、建筑损坏
5. 出口可见性: 各出口是否可见、是否被遮挡或封锁

规则:
- 只描述你实际看到的，不要推断或假设
- 用具体方位词(东北角/左侧/距镜头约10米处)而非模糊词汇
- 如果某个类别没有异常，写"无异常"即可
- 总字数不超过150字"""

VLM_USER = "分析当前监控画面，按类别描述所见情况。"
```

### 3.3 推理封装

```python
# perception/vlm_perceiver.py

class VLMPerceiver:
    """
    Qwen-VL 视觉感知器。

    设计要点:
    1. 不每帧调用——隔30 tick或环境剧变时触发
    2. 结果缓存——同一帧的所有Agent共享
    3. 失败容错——VLM不可用时不影响系统运行
    """

    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"):
        self.model = None          # 延迟加载
        self.model_name = model_name
        self.last_description = "" # 缓存
        self.last_call_tick = -999
        self.call_interval = 30    # 每30 tick (3秒) 调用一次

    def should_call(self, tick: int, env_snapshot) -> bool:
        """判断是否需要重新调用VLM"""
        if tick - self.last_call_tick >= self.call_interval:
            return True
        if env_snapshot.smoke_field.max() - self._last_smoke_max > 0.2:
            return True  # 烟雾突变
        if env_snapshot.grid[:,:,3].sum() - self._last_fire_sum > 5:
            return True  # 火势扩大
        return False

    def perceive(self, frame: np.ndarray, tick: int) -> str:
        """调用VLM, 返回场景描述。失败返回空字符串。"""
        if not self.should_call(tick, env):
            return self.last_description

        try:
            prompt = build_vlm_prompt()
            # Qwen-VL 推理
            result = self.model.chat(frame, prompt, max_tokens=150)
            self.last_description = result
            self.last_call_tick = tick
            return result
        except Exception as e:
            # 容错: VLM挂了不影响主流程
            print(f"[VLM] Error: {e}, using cached description")
            return self.last_description
```

### 3.4 与 NL Converter 融合

```python
# perception/nl_converter.py 新增方法

@staticmethod
def vlm_context(description: str) -> str:
    """将VLM输出格式化为Prompt段落"""
    if not description:
        return ""
    return f"""
[监控画面分析]
{description}
"""
```

---

## 4. 模块二：扩散模型轨迹生成

### 4.1 MID (Motion Indeterminacy Diffusion) 详解

选 MID 而不是其他扩散模型的原因：

| 特性 | MID | LED | MotionDiffuser | Social Force |
|------|:---:|:---:|:---:|:---:|
| 条件注入 | Cross-Attn | FiLM | AdaLN | 无 |
| 文本条件 | 原生支持 | 需改造 | 不支持 | 无 |
| 多Agent联合 | 不支持 | 不支持 | 支持 | 隐含 |
| 参数量 | 15M | 22M | 30M | 0 |
| 推理速度 | 50ms | 80ms | 120ms | 5ms |
| 训练难度 | 低 | 中 | 高 | 无 |

MID 的条件注入通过 Cross-Attention 实现，和 Transformer 架构天然兼容。LLM 的决策文本经过 BERT 编码后直接作为 Cross-Attention 的 K/V 输入，指导每一步去噪方向。

### 4.2 网络结构

```
输入:
  x_t:      [B, 31, 2]    加噪轨迹 (batch × 31步 × xy坐标)
  t:        [B]           噪声步数 (0~1000)
  cond_txt: [B, 768]      LLM决策文本的BERT编码
  cond_map: [B, 3, 64, 64] 场景占用图 (障碍物/烟雾/出口)

                    ┌──────────────────┐
  x_t [B,31,2] ────→│  Linear → [B,31,256] │
                    └────────┬─────────┘
                             │
  t [B] ──────────→│ Sinusoidal Embedding  │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │   Time MLP       │
                    │   [B,256]        │
                    └────────┬─────────┘
                             │
  cond_txt [B,768] ──────────┤
  cond_map → CNN → [B,256] ──┤
                             │
              ┌──────────────┴──────────────┐
              │   Transformer Encoder ×6    │
              │                            │
              │  Layer 1~6:                │
              │    Self-Attention(轨迹内部)  │
              │    Cross-Attention(轨迹←条件)│  ← 条件在这里注入
              │    MLP                     │
              │    LayerNorm + Residual    │
              │                            │
              └──────────────┬─────────────┘
                             │
                    ┌────────┴─────────┐
                    │  Linear → [B,31,2] │  ← 预测的噪声 ε_pred
                    └──────────────────┘
```

### 4.3 推理代码骨架

```python
# execution/diffusion_policy.py

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel  # BERT for text encoding


class DiffusionTrajectoryGenerator(nn.Module):
    """
    条件扩散轨迹生成器。

    输入: 起点 + 目标 + LLM决策文本 + 场景占用图
    输出: 31步自然平滑轨迹
    """

    def __init__(self, config):
        super().__init__()
        # 轨迹编码
        self.traj_proj = nn.Linear(2, 256)

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 256)
        )

        # 条件编码器
        self.text_encoder = AutoModel.from_pretrained(
            "BAAI/bge-base-zh-v1.5"  # 中文BERT
        )
        self.map_encoder = nn.Sequential(  # 场景图→特征
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 256)
        )
        self.cond_proj = nn.Linear(768 + 256, 256)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 输出头
        self.out_proj = nn.Linear(256, 2)

        # DDIM 调度器
        self.num_inference_steps = 100  # 推理时只用100步 (训练1000步)
        self.beta_schedule = self._linear_beta_schedule(1000)

    def _linear_beta_schedule(self, timesteps):
        """线性噪声调度 β₁=0.0001 → β_T=0.02"""
        return torch.linspace(0.0001, 0.02, timesteps)

    def forward(self, x_t, t, cond_txt, cond_map):
        """预测噪声。训练时调用。"""
        # 轨迹特征
        h = self.traj_proj(x_t)  # [B, 31, 256]

        # 时间特征
        t_emb = self._sinusoidal_embedding(t)  # [B, 256]
        t_feat = self.time_mlp(t_emb)          # [B, 256]

        # 条件特征
        txt_feat = self.text_encoder(cond_txt).last_hidden_state[:, 0]  # [B, 768]
        map_feat = self.map_encoder(cond_map)                            # [B, 256]
        cond = self.cond_proj(torch.cat([txt_feat, map_feat], dim=-1))  # [B, 256]

        # 注入时间+条件
        h = h + t_feat.unsqueeze(1)  # 广播到每个时间步

        # Transformer (self-attention within trajectory)
        h = self.transformer(h)  # [B, 31, 256]

        # Cross-attention with condition
        # (简化: 直接把条件拼到序列里)
        cond_pad = cond.unsqueeze(1).expand(-1, 31, -1)
        h = h + cond_pad

        # 输出预测噪声
        eps_pred = self.out_proj(h)  # [B, 31, 2]
        return eps_pred

    @torch.no_grad()
    def generate(self, start, target, llm_decision, scene_map,
                 num_steps=31, num_inference_steps=None):
        """
        从噪声生成轨迹。推理时调用。

        Args:
            start:        [2] 起点坐标
            target:       [2] 目标出口坐标
            llm_decision: str LLM的决策文本 (如"从柱子左侧绕过去, walk速度")
            scene_map:    [3, 64, 64] 场景占用图 (ch0=障碍物, ch1=烟雾, ch2=出口)
            num_steps:    int 轨迹步数 (默认31 = 3秒@10Hz)

        Returns:
            trajectory: [num_steps, 2] numpy 平滑轨迹
        """
        num_steps_inference = num_inference_steps or self.num_inference_steps

        # 1. 编码条件
        txt_emb = self.text_encoder(llm_decision).last_hidden_state[:, 0]  # [1, 768]
        map_emb = self.map_encoder(scene_map.unsqueeze(0))                  # [1, 256]
        cond = self.cond_proj(torch.cat([txt_emb, map_emb], dim=-1))      # [1, 256]

        # 2. 起点和目标约束
        #    在去噪每一步, 强制第0步=start, 第30步=target
        fixed_mask = torch.zeros(1, num_steps, 2)
        fixed_mask[0, 0, :] = 1.0    # 起点固定
        fixed_mask[0, -1, :] = 1.0   # 终点固定

        # 3. DDIM 去噪循环
        x_t = torch.randn(1, num_steps, 2)  # 纯噪声
        x_t[0, 0] = start.float()
        x_t[0, -1] = target.float()

        timesteps = torch.linspace(999, 0, num_steps_inference, dtype=torch.long)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # 预测噪声
            t_batch = torch.tensor([t])
            eps_pred = self.forward(x_t, t_batch, txt_emb, scene_map.unsqueeze(0))

            # DDIM 去噪一步
            alpha = 1 - self.beta_schedule[t]
            alpha_next = 1 - self.beta_schedule[t_next]

            x0_pred = (x_t - torch.sqrt(1 - alpha) * eps_pred) / torch.sqrt(alpha)
            x_t = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * eps_pred

            # 强制约束: 起点和终点不变
            x_t = x_t * (1 - fixed_mask) + fixed_mask * x_t

        # 4. 输出
        trajectory = x_t[0].cpu().numpy()  # [31, 2]
        return trajectory

    def _sinusoidal_embedding(self, t):
        """Transformer 标准位置编码, 用于时间步"""
        half_dim = 128
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb).to(t.device)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
```

### 4.4 Social Force vs Diffusion 共存策略

```python
# execution/orchestrator.py

class SimulationOrchestrator:
    def __init__(self, config):
        mode = config.get("trajectory_mode", "social_force")

        if mode == "diffusion":
            self.traj_generator = DiffusionTrajectoryGenerator(config)
            self.traj_generator.load_state_dict(
                torch.load(config["diffusion"]["checkpoint"])
            )
            self.traj_generator.eval()
            self.use_diffusion = True
        else:
            self.physics = BatchedPhysics(...)
            self.use_diffusion = False

    def step_physics(self, agents, dt):
        if self.use_diffusion:
            self._step_diffusion(agents, dt)
        else:
            self.physics.step_all(agents, dt)

    def _step_diffusion(self, agents, dt):
        """播放预生成轨迹 + 紧急碰撞检测"""
        for agent in agents:
            if agent.needs_new_trajectory():
                traj = self.traj_generator.generate(
                    start=agent.position,
                    target=agent.dynamic.target_exit,
                    llm_decision=agent.dynamic.reasoning_text,
                    scene_map=self._build_scene_map(agent)
                )
                agent.future_trajectory = traj
                agent.traj_step = 0

            if agent.future_trajectory is not None:
                idx = agent.traj_step
                if idx < len(agent.future_trajectory):
                    agent.position = agent.future_trajectory[idx]
                    agent.traj_step += 1
```

---

## 5. 训练数据管线

### 5.1 微调数据生产

用 v1.0 仿真跑 → 导出轨迹 → 自动标注 → 训练集

```python
# execution/diffusion_trainer.py

class TrajectoryDataset(torch.utils.data.Dataset):
    """
    每条样本:
      - start: [2] 起点
      - target: [2] 目标出口
      - llm_decision: str LLM决策文本
      - scene_map: [3, 64, 64] 障碍物/烟雾/出口
      - ground_truth: [31, 2] v1.0仿真生成的轨迹 (社会力模型)
    """

    def __getitem__(self, idx):
        record = self.data[idx]
        return {
            "start":    torch.tensor(record["start"], dtype=torch.float32),
            "target":   torch.tensor(record["target"], dtype=torch.float32),
            "cond_txt": record["llm_decision"],
            "cond_map": torch.tensor(record["scene_map"], dtype=torch.float32),
            "traj":     torch.tensor(record["trajectory"], dtype=torch.float32),
        }
```

### 5.2 训练配置

```yaml
# config/diffusion_train.yaml
training:
  num_epochs: 100
  batch_size: 64
  lr: 1e-4
  warmup_steps: 1000
  num_timesteps: 1000    # 训练用1000步去噪
  num_inference_steps: 100 # 推理仅用100步 (DDIM加速)
  gradient_clip: 1.0

data:
  eth_ucy_path: "./data/eth_ucy"          # 预训练
  simulation_path: "./data/v1_trajectories" # 微调
  simulation_samples: 10000               # 目标生成量

model:
  d_model: 256
  nhead: 8
  num_layers: 6
  text_encoder: "BAAI/bge-base-zh-v1.5"
```

---

## 6. 显存预算与GPU策略

### 6.1 三卡适配方案

| 组件 | 4090 (24GB) 开发 | 5090 (32GB) 开发 | H800 (80GB) 实验 |
|------|:---:|:---:|:---:|
| Qwen-7B 决策LLM | AWQ 3.5GB | AWQ 3.5GB | FP16 14GB |
| Qwen-VL-7B | INT4 5GB | INT4 5GB | FP16 16GB |
| 扩散模型 MID | FP16 0.5GB | FP16 0.5GB | FP16 0.5GB |
| KV Cache | 6GB | 8GB | 25GB |
| 其他 | 5GB | 8GB | 12GB |
| 剩余 | **~4GB** 紧 | **~7GB** OK | **~12GB** 宽裕 |

### 6.2 开发策略

```
4090/5090 开发模式:
  - 三个模型全部量化 (AWQ/INT4)
  - Agent: 100~300 (快速迭代)
  - 每轮决策: 2~3秒
  - 日常改代码、调Prompt、修bug

H800 实验模式:
  - Qwen-7B + VLM 跑 FP16 (最好质量)
  - Agent: 3000~5000
  - 扩散模型容量可以开大 (d_model 256→512)
  - 出论文数据
```

---

## 7. 接口契约

### 7.1 模块间数据协议

```python
# ===== VLM → NL Converter =====
VLMOutput = str  # "西南角浓烟...东出口拥堵..."

# ===== NL Converter → LLM =====
PromptSections = {
    "system": str,        # 系统角色 + 知识
    "environment": str,    # 环境描述 (含VLM)
    "personal": str,       # 个人状态
    "social": str,         # 社会信息
}

# ===== LLM → Diffusion =====
@dataclass
class LLMDecision:
    agent_id: str
    target_exit: np.ndarray            # [2] 出口坐标
    speed: str                         # "run"|"walk"|"crawl"|"wait"
    cooperation: str                   # "none"|"help_family"|...
    reasoning_text: str               # "从柱子左侧绕过去, walk速度"
    reasoning_json: dict              # 完整JSON

# ===== Diffusion → Physics =====
Trajectory = np.ndarray  # [31, 2] 31步×xy坐标
```

### 7.2 Agent 状态扩展

```python
# decision/agent_state.py v2.0 新增字段

@dataclass
class AgentDynamic:
    # ... v1.0 字段保持不变 ...

    # v2.0 新增 ─────────────────────
    future_trajectory: Optional[np.ndarray] = None  # [31,2] 预生成轨迹
    traj_step: int = 0                              # 当前播放到第几步
    traj_generation_tick: int = -999                # 轨迹生成时的tick
    vlm_description_cache: str = ""                 # 上次VLM看到的东西
```

---

## 8. 实验矩阵

### 8.1 主实验: 纵向对比

| 配置 | LLM | 感知 | 轨迹 | Agent | 预期疏散率 |
|------|-----|------|------|:---:|:---:|
| Baseline | 3B-AWQ | 数值 | Social Force | 500 | 42.8% |
| +7B | **7B-FP16** | 数值 | Social Force | 500 | ~48% |
| +VLM | 7B-FP16 | **数值+VLM** | Social Force | 500 | ~54% |
| +Diffusion | 7B-FP16 | 数值+VLM | **Diffusion** | 500 | ~56% |

### 8.2 消融实验

```
VLM 贡献:
  w/o VLM:  疏散率 48%, 摔倒检测率 0%
  w/  VLM:  疏散率 54%, 摔倒检测率 78% (VLM发现了YOLO漏掉的人)

扩散模型贡献:
  Social Force:  轨迹人工评分 2.8/5, 碰撞率 12%
  Diffusion:     轨迹人工评分 4.2/5, 碰撞率 5%

规模实验:
  500 agents  → 疏散率 56%
  2000 agents → 疏散率 48%
  5000 agents → 疏散率 39% (H800)
```

### 8.3 定性分析素材

```
论文图1: 同一场景, Social Force vs Diffusion 轨迹对比
  - 绕柱子: 生硬折线 vs 自然曲线
  - 多人交汇: 震荡 vs 平滑错开
  - 窄通道跟驰: 急停 vs 渐进减速

论文图2: VLM 检测到 YOLO 漏掉的东西
  - 标注图: YOLO框 + VLM文本描述
  - 对比: YOLO只看到3人, VLM还发现"柱子后面蹲着一个人"

论文图3: 决策可解释性
  - Agent#142 的完整决策链: VLM描述→LLM推理→扩散轨迹
```

---

## 9. 四周实施计划

```
Week 1 ─── VLM 感知接入
  Day 1-2: Qwen-VL-7B-INT4 部署, Prompt 调优
  Day 3-4: NL Converter 扩展, 双通道融合
  Day 5:   Agent状态扩展, 显存压力测试
  Day 6-7: 300 Agent 端到端跑通

Week 2 ─── 扩散模型集成
  Day 1-2: MID backbone 加载, 推理接口调试
  Day 3-4: 条件编码器 (LLM文本→embedding)
  Day 5-6: orchestrator 支持模式切换
  Day 7:   300 Agent 验证轨迹质量

Week 3 ─── 训练 + 微调
  Day 1-2: v1.0仿真批量生成10000条轨迹
  Day 3-4: ETH/UCY 预训练
  Day 5-6: 疏散数据微调 (重点: 烟雾减速行为)
  Day 7:   评测集验证

Week 4 ─── 全链路 + 实验
  Day 1-2: 开启H800, 部署FP16全精度
  Day 3-4: 消融实验 (×4配置 ×3种子 = 12轮)
  Day 5:   可视化视频 + 轨迹对比图
  Day 6-7: 论文图表 + 分析
```

---

## 附录: 关键参考论文

| 论文 | 年份 | 与本项目关系 |
|------|:---:|------|
| DDPM (Ho et al.) | 2020 | 扩散模型基础 |
| DDIM (Song et al.) | 2021 | 加速采样 (1000步→100步) |
| MID (Gu et al.) | 2022 | 条件扩散轨迹预测 |
| Social-LSTM (Alahi et al.) | 2016 | 行人交互建模 |
| AgentFormer (Yuan et al.) | 2021 | Transformer轨迹预测 |
| Qwen2.5-VL | 2025 | 视觉语言模型 |
| Generative Agents (Park et al.) | 2023 | LLM Agent建模 |
